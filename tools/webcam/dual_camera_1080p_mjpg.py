#!/usr/bin/env python3
"""
Dual Camera Setup - Both at 1920x1080 @ 30 FPS with MJPG
========================================================

Forces both road (/dev/video2) and wide (/dev/video6) cameras to 1920x1080
resolution using MJPG format for maximum frame rate (30 FPS instead of 5 FPS).
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
ROAD_CAM_ID = "2"   # /dev/video2
WIDE_CAM_ID = "6"   # /dev/video6
# ---------------------------------------------------------------------------

# Target settings - MJPG format for 30 FPS
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 30
TARGET_FORMAT = 'MJPG'  # This is crucial for 30 FPS!

CameraType = namedtuple("CameraType", ["msg_name", "stream_type", "cam_id"])

CAMERAS = [
    CameraType("roadCameraState", VisionStreamType.VISION_STREAM_ROAD, ROAD_CAM_ID),
    CameraType("wideRoadCameraState", VisionStreamType.VISION_STREAM_WIDE_ROAD, WIDE_CAM_ID),
]

class HighFpsCamera(Camera):
    """Camera class that enforces 1920x1080 @ 30 FPS using MJPG format"""

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

        print(f"🔧 Opening {cam_type_state} at {camera_id} (ENFORCING 1920x1080 @ 30 FPS MJPG)")

        # Force MJPG format for 30 FPS
        self._force_mjpg_full_resolution(camera_id)

    def _force_mjpg_full_resolution(self, camera_id):
        """Force MJPG format at 1920x1080 for 30 FPS"""

        print(f"🎯 Method 1: MJPG-first setup for {self.cam_type_state}")
        try:
            self.cap = cv.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {camera_id}")

            # CRITICAL: Set MJPG format FIRST before resolution
            mjpg_fourcc = cv.VideoWriter_fourcc('M','J','P','G')
            format_success = self.cap.set(cv.CAP_PROP_FOURCC, mjpg_fourcc)
            print(f"   📹 MJPG format set: {format_success}")

            # Set buffer size to 1 for minimal latency
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

            # Now set resolution and FPS
            width_success = self.cap.set(cv.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
            height_success = self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
            fps_success = self.cap.set(cv.CAP_PROP_FPS, TARGET_FPS)

            print(f"   📐 Set: width={width_success}, height={height_success}, fps={fps_success}")

            # Verify actual settings
            actual_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv.CAP_PROP_FPS)
            actual_fourcc = int(self.cap.get(cv.CAP_PROP_FOURCC))

            # Convert fourcc back to string
            fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
            print(f"   📊 Actual: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS, format: {fourcc_str}")

            # Test frame capture
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                print(f"   📸 Test frame shape: {test_frame.shape}")

                # Success criteria: correct resolution AND high frame rate
                if (actual_width == TARGET_WIDTH and
                    actual_height == TARGET_HEIGHT and
                    actual_fps >= 25.0):  # Accept 25+ FPS (sometimes 30 reports as 25)

                    self.W = actual_width
                    self.H = actual_height
                    print(f"✅ {self.cam_type_state} SUCCESS: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS ({fourcc_str})")
                    return
                else:
                    print(f"⚠️  {self.cam_type_state} wrong settings: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            else:
                print(f"❌ {self.cam_type_state} cannot capture test frame")

        except Exception as e:
            print(f"❌ Method 1 failed for {self.cam_type_state}: {e}")

        print(f"🎯 Method 2: V4L2 backend with MJPG for {self.cam_type_state}")
        try:
            if self.cap:
                self.cap.release()

            # Try with V4L2 backend specifically
            self.cap = cv.VideoCapture(camera_id, cv.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv.VideoCapture(camera_id)

            # Set buffer size first
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

            # Force MJPG format
            mjpg_fourcc = cv.VideoWriter_fourcc('M','J','P','G')
            self.cap.set(cv.CAP_PROP_FOURCC, mjpg_fourcc)

            # Set resolution and FPS
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
            self.cap.set(cv.CAP_PROP_FPS, TARGET_FPS)

            actual_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv.CAP_PROP_FPS)

            print(f"   📊 V4L2 result: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

            ret, test_frame = self.cap.read()
            if (ret and
                actual_width == TARGET_WIDTH and
                actual_height == TARGET_HEIGHT and
                actual_fps >= 25.0):

                self.W = actual_width
                self.H = actual_height
                print(f"✅ {self.cam_type_state} SUCCESS with V4L2: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
                return

        except Exception as e:
            print(f"❌ Method 2 failed for {self.cam_type_state}: {e}")

        print(f"🎯 Method 3: Direct v4l2-ctl format setting for {self.cam_type_state}")
        try:
            # Use v4l2-ctl to force MJPG format directly
            device_path = f"/dev/video{camera_id}"

            # Set MJPG format using v4l2-ctl
            import subprocess
            result = subprocess.run([
                'v4l2-ctl',
                f'--device={device_path}',
                '--set-fmt-video=width=1920,height=1080,pixelformat=MJPG'
            ], capture_output=True, text=True)

            print(f"   🔧 v4l2-ctl result: {result.returncode}")
            if result.stdout:
                print(f"   📝 stdout: {result.stdout.strip()}")
            if result.stderr:
                print(f"   ⚠️  stderr: {result.stderr.strip()}")

            # Now try to open with OpenCV
            if self.cap:
                self.cap.release()
            self.cap = cv.VideoCapture(camera_id)

            # Just set FPS, format should already be set
            self.cap.set(cv.CAP_PROP_FPS, TARGET_FPS)
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

            actual_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv.CAP_PROP_FPS)

            print(f"   📊 After v4l2-ctl: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

            ret, test_frame = self.cap.read()
            if (ret and
                actual_width == TARGET_WIDTH and
                actual_height == TARGET_HEIGHT and
                actual_fps >= 25.0):

                self.W = actual_width
                self.H = actual_height
                print(f"✅ {self.cam_type_state} SUCCESS with v4l2-ctl: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
                return

        except Exception as e:
            print(f"❌ Method 3 failed for {self.cam_type_state}: {e}")

        # If we get here, we couldn't achieve 30 FPS
        raise RuntimeError(f"❌ Camera {camera_id} ({self.cam_type_state}) cannot achieve 30 FPS at 1920x1080!")

    def read_frames(self):
        """Read frames with FPS monitoring"""
        failed_reads = 0
        max_failed_reads = 10
        frame_count = 0
        start_time = time.time()

        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    failed_reads += 1
                    print(f"⚠️  Failed to read frame from {self.cam_type_state} ({failed_reads}/{max_failed_reads})")

                    if failed_reads >= max_failed_reads:
                        print(f"❌ Camera {self.cam_type_state} failed after {max_failed_reads} failures")
                        break

                    time.sleep(0.1)
                    continue

                # Validate frame resolution
                if frame.shape[1] != TARGET_WIDTH or frame.shape[0] != TARGET_HEIGHT:
                    print(f"⚠️  {self.cam_type_state} wrong frame size: {frame.shape[1]}x{frame.shape[0]} (expected {TARGET_WIDTH}x{TARGET_HEIGHT})")

                failed_reads = 0  # Reset on success
                frame_count += 1

                # Monitor actual FPS every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed
                    print(f"📊 {self.cam_type_state} actual FPS: {actual_fps:.1f}")

                    # Reset counters
                    frame_count = 0
                    start_time = time.time()

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

class DualCamerad30fps:
    def __init__(self):
        print("🚀 Initializing Dual Camera Setup - 1920x1080 @ 30 FPS MJPG")
        print(f"📹 Road Camera: /dev/video{ROAD_CAM_ID}")
        print(f"📹 Wide Camera: /dev/video{WIDE_CAM_ID}")
        print(f"🎯 Target: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {TARGET_FPS} FPS ({TARGET_FORMAT})")

        # Publish camera states
        self.pm = messaging.PubMaster([c.msg_name for c in CAMERAS])
        print(f"✅ PubMaster initialized for: {[c.msg_name for c in CAMERAS]}")

        # Vision IPC server
        self.vipc_server = VisionIpcServer("camerad")
        print("✅ VisionIPC server created")

        # Initialize cameras with ENFORCED 30 FPS
        self.cameras = []
        for i, c in enumerate(CAMERAS):
            try:
                print(f"\n🔧 Initializing camera {i+1}/{len(CAMERAS)}: {c.msg_name}")
                cam_device = f"/dev/video{c.cam_id}" if platform.system() != "Darwin" else c.cam_id
                print(f"   Device: {cam_device}")

                # Use high FPS camera class
                cam = HighFpsCamera(c.msg_name, c.stream_type, cam_device)

                # Verify we got the right resolution
                if cam.W != TARGET_WIDTH or cam.H != TARGET_HEIGHT:
                    raise RuntimeError(f"Camera {c.msg_name} resolution mismatch: got {cam.W}x{cam.H}, need {TARGET_WIDTH}x{TARGET_HEIGHT}")

                print(f"   ✅ Verified: {cam.W}x{cam.H} ready for 30 FPS")

                # Create VisionIPC buffers
                print(f"   Creating VisionIPC buffers...")
                self.vipc_server.create_buffers(c.stream_type, 20, cam.W, cam.H)
                print(f"   ✅ Buffers created for stream type {c.stream_type.value}")

                self.cameras.append(cam)
                print(f"   ✅ Camera {c.msg_name} ready for 30 FPS streaming!")

            except Exception as e:
                print(f"   ❌ FAILED to initialize camera {c.msg_name} at 30 FPS: {e}")
                raise  # Don't continue with slow cameras

        print(f"\n✅ Successfully initialized {len(self.cameras)}/{len(CAMERAS)} cameras for 30 FPS")

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
        """Camera capture loop optimized for 30 FPS"""
        print(f"🎬 Starting camera runner for {cam.cam_type_state} (targeting 30 FPS)")
        rk = Ratekeeper(TARGET_FPS, None)

        try:
            for yuv in cam.read_frames():
                self._send_yuv(yuv, cam.cur_frame_id, cam.cam_type_state, cam.stream_type)
                cam.cur_frame_id += 1

                # Status update every 300 frames (10 seconds at 30 FPS)
                if cam.cur_frame_id % 300 == 0:
                    print(f"📊 {cam.cam_type_state}: Frame {cam.cur_frame_id} (30 FPS target)")

                rk.keep_time()

        except Exception as e:
            print(f"❌ Camera runner error for {cam.cam_type_state}: {e}")

    def run(self):
        """Start camera daemon"""
        print(f"\n🏃 Starting {len(self.cameras)} camera threads for 30 FPS streaming...")
        threads = []

        for cam in self.cameras:
            t = threading.Thread(target=self.camera_runner, args=(cam,), daemon=True)
            t.start()
            threads.append(t)
            print(f"✅ Started 30 FPS thread for {cam.cam_type_state}")

        print("🎯 Dual 30 FPS camera daemon running. Press Ctrl+C to stop.")

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down camera daemon...")

def main():
    DualCamerad30fps().run()

if __name__ == "__main__":
    main()