#!/usr/bin/env python3
"""
Dual Camera Setup - Road + Wide
==============================

A setup that includes both road (/dev/video2) and wide (/dev/video6) cameras
with robust error handling for camera-specific issues.
"""
import threading
import os
import platform
import time
import cv2 as cv
from collections import namedtuple

from msgq.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging

from openpilot.tools.webcam.camera import Camera
from openpilot.common.realtime import Ratekeeper

# ---------------------------------------------------------------------------
# Camera IDs
# ---------------------------------------------------------------------------
ROAD_CAM_ID = "2"   # /dev/video2 - This one works!
WIDE_CAM_ID = "6"   # /dev/video6 - This one has issues
# ---------------------------------------------------------------------------

CameraType = namedtuple("CameraType", ["msg_name", "stream_type", "cam_id"])

CAMERAS = [
    CameraType("roadCameraState", VisionStreamType.VISION_STREAM_ROAD, ROAD_CAM_ID),
    CameraType("wideRoadCameraState", VisionStreamType.VISION_STREAM_WIDE_ROAD, WIDE_CAM_ID),
]

class ImprovedCamera(Camera):
    """Improved camera class with better error handling for problematic cameras"""

    def __init__(self, cam_type_state, stream_type, camera_id):
        self.cam_type_state = cam_type_state
        self.stream_type = stream_type
        self.cur_frame_id = 0
        self.camera_id = camera_id
        self.cap = None
        self.W = 0
        self.H = 0

        try:
            camera_id = int(camera_id)
        except ValueError:
            pass

        print(f"Opening {cam_type_state} at {camera_id}")

        # Try different settings for problematic cameras
        self._try_initialize_camera(camera_id)

    def _try_initialize_camera(self, camera_id):
        """Try different camera initialization approaches"""

        # Method 1: Standard initialization
        try:
            self.cap = cv.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {camera_id}")

            # Try standard settings first
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920.0)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080.0)
            self.cap.set(cv.CAP_PROP_FPS, 30.0)

            # Test if it can read
            ret, test_frame = self.cap.read()
            if ret:
                self.W = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
                self.H = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self.cap.get(cv.CAP_PROP_FPS)
                print(f"✅ {self.cam_type_state} initialized: {int(self.W)}x{int(self.H)} @ {actual_fps:.1f} FPS")
                print(f"✅ {self.cam_type_state} test frame successful: {test_frame.shape}")
                return
        except Exception as e:
            print(f"⚠️  Standard init failed for {self.cam_type_state}: {e}")

        # Method 2: Try with different resolution for problematic cameras
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv.VideoCapture(camera_id)

            # Try lower resolution that might be more compatible
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280.0)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720.0)
            self.cap.set(cv.CAP_PROP_FPS, 30.0)

            ret, test_frame = self.cap.read()
            if ret:
                self.W = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
                self.H = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self.cap.get(cv.CAP_PROP_FPS)
                print(f"✅ {self.cam_type_state} initialized (fallback): {int(self.W)}x{int(self.H)} @ {actual_fps:.1f} FPS")
                print(f"✅ {self.cam_type_state} test frame successful: {test_frame.shape}")
                return
        except Exception as e:
            print(f"⚠️  Fallback init failed for {self.cam_type_state}: {e}")

        # Method 3: Try even more conservative settings
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv.VideoCapture(camera_id)

            # Very conservative settings
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640.0)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480.0)
            self.cap.set(cv.CAP_PROP_FPS, 15.0)

            ret, test_frame = self.cap.read()
            if ret:
                self.W = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
                self.H = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self.cap.get(cv.CAP_PROP_FPS)
                print(f"✅ {self.cam_type_state} initialized (conservative): {int(self.W)}x{int(self.H)} @ {actual_fps:.1f} FPS")
                print(f"✅ {self.cam_type_state} test frame successful: {test_frame.shape}")
                return
        except Exception as e:
            print(f"⚠️  Conservative init failed for {self.cam_type_state}: {e}")

        # If all methods fail
        raise RuntimeError(f"Cannot initialize camera {camera_id} for {self.cam_type_state} with any settings")

    def read_frames(self):
        """Improved frame reading with better error handling"""
        failed_reads = 0
        max_failed_reads = 20  # More tolerance for problematic cameras
        consecutive_failures = 0

        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    failed_reads += 1
                    consecutive_failures += 1
                    print(f"⚠️  Failed to read frame from {self.cam_type_state} ({failed_reads}/{max_failed_reads})")

                    if consecutive_failures >= 5:
                        # Try to reinitialize camera
                        print(f"🔄 Attempting to reinitialize {self.cam_type_state}")
                        try:
                            self.cap.release()
                            time.sleep(0.5)
                            self._try_initialize_camera(self.camera_id)
                            consecutive_failures = 0
                            continue
                        except Exception as e:
                            print(f"❌ Reinitialize failed for {self.cam_type_state}: {e}")

                    if failed_reads >= max_failed_reads:
                        print(f"❌ Camera {self.cam_type_state} failed after {max_failed_reads} failures")
                        break

                    time.sleep(0.1)
                    continue

                # Reset failure counters on successful read
                failed_reads = 0
                consecutive_failures = 0

                # Rotate the frame 180 degrees (flip both axes)
                frame = cv.flip(frame, -1)

                try:
                    yuv = Camera.bgr2nv12(frame)
                    yield yuv.data.tobytes()
                except Exception as e:
                    print(f"⚠️  YUV conversion error for {self.cam_type_state}: {e}")
                    continue

            except Exception as e:
                print(f"⚠️  Read loop error for {self.cam_type_state}: {e}")
                time.sleep(0.1)
                continue

        print(f"🔚 Camera {self.cam_type_state} read loop ended")
        if self.cap:
            self.cap.release()

class DualCamerad:
    def __init__(self):
        print("🚀 Initializing Dual Camera Setup (Road + Wide)")
        print(f"📹 Road Camera: /dev/video{ROAD_CAM_ID}")
        print(f"📹 Wide Camera: /dev/video{WIDE_CAM_ID}")

        # Publish camera states
        self.pm = messaging.PubMaster([c.msg_name for c in CAMERAS])
        print(f"✅ PubMaster initialized for: {[c.msg_name for c in CAMERAS]}")

        # Vision IPC server
        self.vipc_server = VisionIpcServer("camerad")
        print("✅ VisionIPC server created")

        # Initialize cameras with improved error handling
        self.cameras = []
        for i, c in enumerate(CAMERAS):
            try:
                print(f"\n🔧 Initializing camera {i+1}/{len(CAMERAS)}: {c.msg_name}")
                cam_device = f"/dev/video{c.cam_id}" if platform.system() != "Darwin" else c.cam_id
                print(f"   Device: {cam_device}")

                # Use improved camera class
                cam = ImprovedCamera(c.msg_name, c.stream_type, cam_device)
                print(f"   Resolution: {cam.W}x{cam.H}")

                # Create VisionIPC buffers
                print(f"   Creating VisionIPC buffers...")
                self.vipc_server.create_buffers(c.stream_type, 20, cam.W, cam.H)
                print(f"   ✅ Buffers created for stream type {c.stream_type.value}")

                self.cameras.append(cam)
                print(f"   ✅ Camera {c.msg_name} ready!")

            except Exception as e:
                print(f"   ❌ Failed to initialize camera {c.msg_name}: {e}")
                print(f"      Continuing with other cameras...")

        if len(self.cameras) == 0:
            raise RuntimeError("❌ No cameras could be initialized!")

        print(f"\n✅ Successfully initialized {len(self.cameras)}/{len(CAMERAS)} cameras")

        # Start VisionIPC listener
        print("🔗 Starting VisionIPC listener...")
        self.vipc_server.start_listener()
        print("✅ VisionIPC listener started")

    def _send_yuv(self, yuv, frame_id, pub_type, yuv_type):
        """Send YUV frame to VisionIPC and publish metadata"""
        try:
            import time
            timestamp_ns = int(time.time_ns())
            self.vipc_server.send(yuv_type, yuv, frame_id, timestamp_ns, timestamp_ns)

            dat = messaging.new_message(pub_type, valid=True)
            msg = {
                "frameId": frame_id,
                "timestampSof": timestamp_ns,
                "transform": [1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0]
            }
            setattr(dat, pub_type, msg)
            self.pm.send(pub_type, dat)
        except Exception as e:
            print(f"❌ Error sending frame for {pub_type}: {e}")

    def camera_runner(self, cam):
        """Camera capture loop with improved error handling"""
        print(f"🎬 Starting camera runner for {cam.cam_type_state}")
        rk = Ratekeeper(30, None)  # 30 fps

        try:
            for yuv in cam.read_frames():
                self._send_yuv(yuv, cam.cur_frame_id, cam.cam_type_state, cam.stream_type)
                cam.cur_frame_id += 1

                # Status update every 100 frames
                if cam.cur_frame_id % 100 == 0:
                    print(f"📊 {cam.cam_type_state}: Frame {cam.cur_frame_id}")

                rk.keep_time()

        except Exception as e:
            print(f"❌ Camera runner error for {cam.cam_type_state}: {e}")

    def run(self):
        """Start camera daemon"""
        print(f"\n🏃 Starting {len(self.cameras)} camera threads...")
        threads = []

        for cam in self.cameras:
            t = threading.Thread(target=self.camera_runner, args=(cam,), daemon=True)
            t.start()
            threads.append(t)
            print(f"✅ Started thread for {cam.cam_type_state}")

        print("🎯 Dual camera daemon running. Press Ctrl+C to stop.")

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down camera daemon...")

def main():
    DualCamerad().run()

if __name__ == "__main__":
    main()