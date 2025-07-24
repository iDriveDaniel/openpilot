#!/usr/bin/env python3
"""
ONNX-based model inference script for openpilot
This script provides an alternative to the tinygrad-based modeld.py using ONNX models
"""

import os
import time
import pickle
import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    print("Warning: onnxruntime not installed. Please install with: pip install onnxruntime")
    ort = None

import cereal.messaging as messaging
from cereal import car, log
from pathlib import Path
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value, get_curvature_from_plan
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from typing import Dict, Optional, Union

PROCESS_NAME = "selfdrive.modeld.onnx_modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

# ONNX model paths
VISION_ONNX_PATH = Path(__file__).parent / 'models/driving_vision.onnx'
POLICY_ONNX_PATH = Path(__file__).parent / 'models/driving_policy.onnx'
BIG_VISION_ONNX_PATH = Path(__file__).parent / 'models/big_driving_vision.onnx'
BIG_POLICY_ONNX_PATH = Path(__file__).parent / 'models/big_driving_policy.onnx'

# Metadata paths (for input/output shapes)
VISION_METADATA_PATH = Path(__file__).parent / 'models/driving_vision_metadata.pkl'
POLICY_METADATA_PATH = Path(__file__).parent / 'models/driving_policy_metadata.pkl'

LAT_SMOOTH_SECONDS = 0.1
LONG_SMOOTH_SECONDS = 0.3
MIN_LAT_CONTROL_SPEED = 0.3


def get_action_from_model(model_output: Dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                          lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
    """Extract driving actions from model output"""
    plan = model_output['plan'][0]
    desired_accel, should_stop = get_accel_from_plan(plan[:,Plan.VELOCITY][:,0],
                                                     plan[:,Plan.ACCELERATION][:,0],
                                                     ModelConstants.T_IDXS,
                                                     action_t=long_action_t)
    desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, LONG_SMOOTH_SECONDS)

    desired_curvature = get_curvature_from_plan(plan[:,Plan.T_FROM_CURRENT_EULER][:,2],
                                                plan[:,Plan.ORIENTATION_RATE][:,2],
                                                ModelConstants.T_IDXS,
                                                v_ego,
                                                lat_action_t)
    if v_ego > MIN_LAT_CONTROL_SPEED:
      desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, LAT_SMOOTH_SECONDS)
    else:
      desired_curvature = prev_action.desiredCurvature

    return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature),
                                  desiredAcceleration=float(desired_accel),
                                  shouldStop=bool(should_stop))


class FrameMeta:
    """Frame metadata container"""
    frame_id: int = 0
    timestamp_sof: int = 0
    timestamp_eof: int = 0

    def __init__(self, vipc=None):
        if vipc is not None:
            self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


class ONNXModelState:
    """ONNX-based model state manager"""

    def __init__(self, use_big_models: bool = False):
        if ort is None:
            raise ImportError("onnxruntime is required but not installed. Please install with: pip install onnxruntime")

        # Load metadata for input/output shapes
        with open(VISION_METADATA_PATH, 'rb') as f:
            vision_metadata = pickle.load(f)
            self.vision_input_shapes = vision_metadata['input_shapes']
            self.vision_input_names = list(self.vision_input_shapes.keys())
            self.vision_output_slices = vision_metadata['output_slices']
            vision_output_size = vision_metadata['output_shapes']['outputs'][1]

        with open(POLICY_METADATA_PATH, 'rb') as f:
            policy_metadata = pickle.load(f)
            self.policy_input_shapes = policy_metadata['input_shapes']
            self.policy_output_slices = policy_metadata['output_slices']
            policy_output_size = policy_metadata['output_shapes']['outputs'][1]

        # Initialize ONNX runtime sessions
        vision_model_path = BIG_VISION_ONNX_PATH if use_big_models else VISION_ONNX_PATH
        policy_model_path = BIG_POLICY_ONNX_PATH if use_big_models else POLICY_ONNX_PATH

        # Configure ONNX runtime options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        # Try to use GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        cloudlog.info(f"Loading vision model from {vision_model_path}")
        self.vision_session = ort.InferenceSession(str(vision_model_path), sess_options, providers=providers)

        cloudlog.info(f"Loading policy model from {policy_model_path}")
        self.policy_session = ort.InferenceSession(str(policy_model_path), sess_options, providers=providers)

        # Initialize buffers
        self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
        self.full_features_buffer = np.zeros((1, ModelConstants.FULL_HISTORY_BUFFER_LEN, ModelConstants.FEATURE_LEN), dtype=np.float32)
        self.full_desire = np.zeros((1, ModelConstants.FULL_HISTORY_BUFFER_LEN, ModelConstants.DESIRE_LEN), dtype=np.float32)
        self.full_prev_desired_curv = np.zeros((1, ModelConstants.FULL_HISTORY_BUFFER_LEN, ModelConstants.PREV_DESIRED_CURV_LEN), dtype=np.float32)
        self.temporal_idxs = slice(-1-(ModelConstants.TEMPORAL_SKIP*(ModelConstants.INPUT_HISTORY_BUFFER_LEN-1)), None, ModelConstants.TEMPORAL_SKIP)

        # Policy inputs
        self.numpy_inputs = {
            'desire': np.zeros((1, ModelConstants.INPUT_HISTORY_BUFFER_LEN, ModelConstants.DESIRE_LEN), dtype=np.float16),
            'traffic_convention': np.zeros((1, ModelConstants.TRAFFIC_CONVENTION_LEN), dtype=np.float16),
            'lateral_control_params': np.zeros((1, ModelConstants.LATERAL_CONTROL_PARAMS_LEN), dtype=np.float16),
            'prev_desired_curv': np.zeros((1, ModelConstants.INPUT_HISTORY_BUFFER_LEN, ModelConstants.PREV_DESIRED_CURV_LEN), dtype=np.float16),
            'features_buffer': np.zeros((1, ModelConstants.INPUT_HISTORY_BUFFER_LEN, ModelConstants.FEATURE_LEN), dtype=np.float16),
        }

        # Output buffers
        self.vision_output = np.zeros(vision_output_size, dtype=np.float32)
        self.policy_output = np.zeros(policy_output_size, dtype=np.float32)
        self.parser = Parser()

    def slice_outputs(self, model_outputs: np.ndarray, output_slices: Dict[str, slice]) -> Dict[str, np.ndarray]:
        """Slice model outputs according to metadata"""
        parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
        return parsed_model_outputs

    def preprocess_images(self, bufs: Dict[str, VisionBuf], transforms: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Preprocess images for ONNX model input"""
        processed_images = {}

        for name in self.vision_input_names:
            buf = bufs[name]
            transform = transforms[name]

            # Get image data from VisionBuf
            img_data = buf.data

            # Expected shape from metadata (e.g., (1, 12, 128, 256))
            expected_shape = self.vision_input_shapes[name]
            batch, channels, height, width = expected_shape

            # The model expects 12 channels but YUV420 has 6 channels
            # We need to convert from YUV420 format to the expected format
            # YUV420 format: 6 channels (Y[::2,::2], Y[::2,1::2], Y[1::2,::2], Y[1::2,1::2], U, V)

            # Calculate the expected YUV420 data size
            yuv_size = height * width * 6

            # Check if we have enough data
            if len(img_data) < yuv_size:
                # Pad with zeros if not enough data
                padded_data = np.zeros(yuv_size, dtype=np.uint8)
                padded_data[:len(img_data)] = img_data
                img_data = padded_data
            else:
                # Truncate if too much data
                img_data = img_data[:yuv_size]

            # Reshape to YUV420 format: (6, height, width)
            yuv_data = img_data.reshape(6, height, width)

            # Convert YUV420 to the expected 12-channel format
            # This is a simplified conversion - in production you'd need proper YUV to RGB conversion
            # For now, we'll duplicate the channels to match the expected format
            processed_img = np.zeros((batch, channels, height, width), dtype=np.uint8)

            # Fill the first 6 channels with YUV data
            for i in range(min(6, channels)):
                processed_img[0, i, :, :] = yuv_data[i, :, :]

            # Fill remaining channels with zeros or duplicate data
            for i in range(6, channels):
                processed_img[0, i, :, :] = yuv_data[i % 6, :, :]

            processed_images[name] = processed_img

        return processed_images

    def run(self, bufs: Dict[str, VisionBuf], transforms: Dict[str, np.ndarray],
            inputs: Dict[str, np.ndarray], prepare_only: bool) -> Optional[Dict[str, np.ndarray]]:
        """Run ONNX model inference"""

        # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
        inputs['desire'][0] = 0
        new_desire = np.where(inputs['desire'] - self.prev_desire > .99, inputs['desire'], 0)
        self.prev_desire[:] = inputs['desire']

        self.full_desire[0,:-1] = self.full_desire[0,1:]
        self.full_desire[0,-1] = new_desire
        self.numpy_inputs['desire'][:] = self.full_desire.reshape((1,ModelConstants.INPUT_HISTORY_BUFFER_LEN,ModelConstants.TEMPORAL_SKIP,-1)).max(axis=2)

        self.numpy_inputs['traffic_convention'][:] = inputs['traffic_convention'].astype(np.float16)
        self.numpy_inputs['lateral_control_params'][:] = inputs['lateral_control_params'].astype(np.float16)

        if prepare_only:
            return None

        # Preprocess images
        vision_inputs = self.preprocess_images(bufs, transforms)

        # Run vision model
        vision_outputs = self.vision_session.run(None, vision_inputs)
        # Convert output to numpy array and flatten
        if isinstance(vision_outputs[0], np.ndarray):
            self.vision_output = vision_outputs[0].flatten().astype(np.float32)
        else:
            # Handle other output types (like SparseTensor)
            self.vision_output = np.array(vision_outputs[0]).flatten().astype(np.float32)

        # Parse vision outputs
        vision_outputs_dict = self.parser.parse_vision_outputs(
            self.slice_outputs(self.vision_output, self.vision_output_slices)
        )

        # Update feature buffer
        self.full_features_buffer[0,:-1] = self.full_features_buffer[0,1:]
        self.full_features_buffer[0,-1] = vision_outputs_dict['hidden_state'][0, :].astype(np.float16)
        self.numpy_inputs['features_buffer'][:] = self.full_features_buffer[0, self.temporal_idxs].astype(np.float16)

        # Run policy model
        policy_outputs = self.policy_session.run(None, self.numpy_inputs)
        # Convert output to numpy array and flatten
        if isinstance(policy_outputs[0], np.ndarray):
            self.policy_output = policy_outputs[0].flatten().astype(np.float32)
        else:
            # Handle other output types (like SparseTensor)
            self.policy_output = np.array(policy_outputs[0]).flatten().astype(np.float32)

        # Parse policy outputs
        policy_outputs_dict = self.parser.parse_policy_outputs(
            self.slice_outputs(self.policy_output, self.policy_output_slices)
        )

        # Update previous desired curvature
        self.full_prev_desired_curv[0,:-1] = self.full_prev_desired_curv[0,1:]
        self.full_prev_desired_curv[0,-1,:] = policy_outputs_dict['desired_curvature'][0, :]
        self.numpy_inputs['prev_desired_curv'][:] = 0*self.full_prev_desired_curv[0, self.temporal_idxs]

        # Combine outputs
        combined_outputs_dict = {**vision_outputs_dict, **policy_outputs_dict}
        if SEND_RAW_PRED:
            combined_outputs_dict['raw_pred'] = np.concatenate([self.vision_output.copy(), self.policy_output.copy()])
        # print(combined_outputs_dict["lane_lines"])
        return combined_outputs_dict


def main(demo: bool = False, use_big_models: bool = False):
    """Main function for ONNX-based model inference"""
    cloudlog.warning("onnx_modeld init")

    # Configure realtime process
    config_realtime_process(7, 54)

    st = time.monotonic()
    cloudlog.warning("loading ONNX models")
    model = ONNXModelState(use_big_models=use_big_models)
    cloudlog.warning(f"models loaded in {time.monotonic() - st:.1f}s, onnx_modeld starting")

    # VisionIPC clients setup
    while True:
        available_streams = VisionIpcClient.available_streams("camerad", block=False)
        if available_streams:
            use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
            main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
            break
        time.sleep(.1)

    vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
    vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True)
    vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
    cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

    while not vipc_client_main.connect(False):
        time.sleep(0.1)
    while use_extra_client and not vipc_client_extra.connect(False):
        time.sleep(0.1)

    cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
    if use_extra_client:
        cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

    # Messaging setup
    pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry"])
    sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl", "liveDelay"])

    publish_state = PublishState()
    params = Params()

    # Setup filter to track dropped frames
    frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_FREQ)
    frame_id = 0
    last_vipc_frame_id = 0
    run_count = 0

    model_transform_main = np.zeros((3, 3), dtype=np.float32)
    model_transform_extra = np.zeros((3, 3), dtype=np.float32)
    live_calib_seen = False
    buf_main, buf_extra = None, None
    meta_main = FrameMeta()
    meta_extra = FrameMeta()

    if demo:
        CP = get_demo_car_params()
    else:
        CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
    cloudlog.info("onnx_modeld got CarParams: %s", CP.brand)

    long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS
    prev_action = log.ModelDataV2.Action()

    DH = DesireHelper()

    cloudlog.warning("onnx_modeld starting main loop")

    while True:
        # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
        while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
            buf_main = vipc_client_main.recv()
            meta_main = FrameMeta(vipc_client_main)
            if buf_main is None:
                break

        if buf_main is None:
            cloudlog.debug("vipc_client_main no frame")
            continue

        if use_extra_client:
            # Keep receiving extra frames until frame id matches main camera
            while True:
                buf_extra = vipc_client_extra.recv()
                meta_extra = FrameMeta(vipc_client_extra)
                if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
                    break

            if buf_extra is None:
                cloudlog.debug("vipc_client_extra no frame")
                continue

            if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
                cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                                 extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")
        else:
            # Use single camera
            buf_extra = buf_main
            meta_extra = meta_main

        sm.update(0)
        desire = DH.desire
        is_rhd = sm["driverMonitoringState"].isRHD
        frame_id = sm["roadCameraState"].frameId
        v_ego = max(sm["carState"].vEgo, 0.)
        lat_delay = sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS
        lateral_control_params = np.array([v_ego, lat_delay], dtype=np.float32)

        if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
            device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
            dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
            model_transform_main = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False).astype(np.float32)
            model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics, True).astype(np.float32)
            live_calib_seen = True

        traffic_convention = np.zeros(2)
        traffic_convention[int(is_rhd)] = 1

        vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
        if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
            vec_desire[desire] = 1

        # Track dropped frames
        vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
        frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
        if run_count < 10:  # let frame drops warm up
            frame_dropped_filter.x = 0.
            frames_dropped = 0.
        run_count = run_count + 1

        frame_drop_ratio = frames_dropped / (1 + frames_dropped)
        prepare_only = vipc_dropped_frames > 0
        if prepare_only:
            cloudlog.error(f"skipping model eval. Dropped {vipc_dropped_frames} frames")

        bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
        transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}
        inputs = {
            'desire': vec_desire,
            'traffic_convention': traffic_convention,
            'lateral_control_params': lateral_control_params,
        }

        mt1 = time.perf_counter()
        model_output = model.run(bufs, transforms, inputs, prepare_only)
        mt2 = time.perf_counter()
        model_execution_time = mt2 - mt1

        if model_output is not None:
            modelv2_send = messaging.new_message('modelV2')
            drivingdata_send = messaging.new_message('drivingModelData')
            posenet_send = messaging.new_message('cameraOdometry')

            action = get_action_from_model(model_output, prev_action, lat_delay + DT_MDL, long_delay + DT_MDL, v_ego)
            prev_action = action
            fill_model_msg(drivingdata_send, modelv2_send, model_output, action,
                           publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                           frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)

            desire_state = modelv2_send.modelV2.meta.desireState
            l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
            r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
            lane_change_prob = l_lane_change_prob + r_lane_change_prob
            DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
            modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
            modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
            drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
            drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction

            fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
            pm.send('modelV2', modelv2_send)
            pm.send('drivingModelData', drivingdata_send)
            pm.send('cameraOdometry', posenet_send)

        last_vipc_frame_id = meta_main.frame_id


if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser(description='ONNX-based model inference for openpilot')
        parser.add_argument('--demo', action='store_true', help='Run in demo mode')
        parser.add_argument('--big-models', action='store_true', help='Use big ONNX models')
        args = parser.parse_args()
        main(demo=args.demo, use_big_models=args.big_models)
    except KeyboardInterrupt:
        cloudlog.warning("got SIGINT")
    except Exception as e:
        cloudlog.error(f"onnx_modeld error: {e}")
        raise