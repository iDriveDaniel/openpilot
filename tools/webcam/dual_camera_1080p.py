#!/usr/bin/env python3
"""
Dual Camera Setup - Both at 1920x1080
====================================

Forces both road (/dev/video2) and wide (/dev/video6) cameras to 1920x1080 resolution.
Includes aggressive troubleshooting for problematic cameras.
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

# Target resolution - both cameras MUST be this resolution
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 30

CameraType = namedtuple("CameraType", ["msg_name", "stream_type", "cam_id"])

CAMERAS = [
    CameraType("roadCameraState", VisionStreamType.VISION_STREAM_ROAD, ROAD_CAM_ID),
    CameraType("wideRoadCameraState", VisionStreamType.VISION_STREAM_WIDE_ROAD, WIDE_CAM_ID),
]

class FullResCamera(Camera):
    """Camera class that enforces 1920x1080 resolution"""

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

        print(f"🔧 Opening {cam_type_state} at {camera_id} (ENFORCING 1920x1080)")

        # Force 1920x1080 initialization
        self._force_full_resolution(camera_id)

    def _force_full_resolution(self, camera_id):
        """Aggressively force 1920x1080 resolution"""

        print(f"🎯 Method 1: Direct 1920x1080 setup for {self.cam_type_state}")
        try:
            self.cap = cv.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {camera_id}")

            # Force exact target resolution
            success_width = self.cap.set(cv.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
            success_height = self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
            success_fps = self.cap.set(cv.CAP_PROP_FPS, TARGET_FPS)

            print(f"   📐 Set width: {success_width}, height: {success_height}, fps: {success_fps}")

            # Get actual values
            actual_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv.CAP_PROP_FPS)

            print(f"   📊 Actual: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

            # Test frame capture
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                print(f"   📸 Test frame shape: {test_frame.shape}")

                # Check if we got the target resolution
                if actual_width == TARGET_WIDTH and actual_height == TARGET_HEIGHT:
                    self.W = actual_width
                    self.H = actual_height
                    print(f"✅ {self.cam_type_state} SUCCESS at {actual_width}x{actual_height}")
                    return
                else:
                    print(f"⚠️  {self.cam_type_state} wrong resolution: got {actual_width}x{actual_height}, need {TARGET_WIDTH}x{TARGET_HEIGHT}")
            else:
                print(f"❌ {self.cam_type_state} cannot capture test frame")

        except Exception as e:
            print(f"❌ Method 1 failed for {self.cam_type_state}: {e}")

        print(f"🎯 Method 2: Force with backend for {self.cam_type_state}")
        try:
            if self.cap:
                self.cap.release()

            # Try with specific backend
            self.cap = cv.VideoCapture(camera_id, cv.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv.VideoCapture(camera_id)  # Fallback to default

            # Set buffer size to prevent lag
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

            # Set format first
            self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M','J','P','G'))

            # Force target resolution
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
            self.cap.set(cv.CAP_PROP_FPS, TARGET_FPS)

            actual_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

            print(f"   📊 Method 2 result: {actual_width}x{actual_height}")

            ret, test_frame = self.cap.read()
            if ret and actual_width == TARGET_WIDTH and actual_height == TARGET_HEIGHT:
                self.W = actual_width
                self.H = actual_height
                print(f"✅ {self.cam_type_state} SUCCESS with Method 2")
                return

        except Exception as e:
            print(f"❌ Method 2 failed for {self.cam_type_state}: {e}")

        print(f"🎯 Method 3: Check supported formats for {self.cam_type_state}")
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv.VideoCapture(camera_id)

            # Try different formats that might support 1920x1080
            formats_to_try = ['MJPG', 'YUYV', 'H264', 'MP4V']

            for fmt in formats_to_try:
                print(f"   🧪 Trying format {fmt}")
                fourcc = cv.VideoWriter_fourcc(*list(fmt))
                self.cap.set(cv.CAP_PROP_FOURCC, fourcc)
                self.cap.set(cv.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
                self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
                self.cap.set(cv.CAP_PROP_FPS, TARGET_FPS)

                actual_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

                if actual_width == TARGET_WIDTH and actual_height == TARGET_HEIGHT:
                    ret, test_frame = self.cap.read()
                    if ret:
                        self.W = actual_width
                        self.H = actual_height
                        print(f"✅ {self.cam_type_state} SUCCESS with format {fmt}")
                        return

        except Exception as e:
            print(f"❌ Method 3 failed for {self.cam_type_state}: {e}")

        # Last resort: check what the camera actually supports
        print(f"🔍 Method 4: Diagnostic for {self.cam_type_state}")
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv.VideoCapture(camera_id)

            # Get all possible settings
            width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
            fps = self.cap.get(cv.CAP_PROP_FPS)

            print(f"   📋 Default settings: {width}x{height} @ {fps} FPS")

            # Try to force again and see what happens
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

            final_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            final_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

            print(f"   📏 After forcing: {final_width}x{final_height}")

        except Exception as e:
            print(f"❌ Diagnostic failed: {e}")

        # If we get here, the camera cannot do 1920x1080
        raise RuntimeError(f"❌ Camera {camera_id} ({self.cam_type_state}) cannot achieve 1920x1080 resolution!")

    def read_frames(self):
        """Read frames with resolution validation"""
        failed_reads = 0
        max_failed_reads = 10

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
                    # You might want to resize here, but it's better to fix at source
                    # frame = cv.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

                failed_reads = 0  # Reset on success

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

class DualCamerad1080p:
    def __init__(self):
        print("🚀 Initializing Dual Camera Setup - ENFORCING 1920x1080")
        print(f"📹 Road Camera: /dev/video{ROAD_CAM_ID}")
        print(f"📹 Wide Camera: /dev/video{WIDE_CAM_ID}")
        print(f"🎯 Target Resolution: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {TARGET_FPS} FPS")

        # Publish camera states
        self.pm = messaging.PubMaster([c.msg_name for c in CAMERAS])
        print(f"✅ PubMaster initialized for: {[c.msg_name for c in CAMERAS]}")

        # Vision IPC server
        self.vipc_server = VisionIpcServer("camerad")
        print("✅ VisionIPC server created")

        # Initialize cameras with ENFORCED resolution
        self.cameras = []
        for i, c in enumerate(CAMERAS):
            try:
                print(f"\n🔧 Initializing camera {i+1}/{len(CAMERAS)}: {c.msg_name}")
                cam_device = f"/dev/video{c.cam_id}" if platform.system() != "Darwin" else c.cam_id
                print(f"   Device: {cam_device}")

                # Use enforced resolution camera class
                cam = FullResCamera(c.msg_name, c.stream_type, cam_device)

                # Verify we got the right resolution
                if cam.W != TARGET_WIDTH or cam.H != TARGET_HEIGHT:
                    raise RuntimeError(f"Camera {c.msg_name} resolution mismatch: got {cam.W}x{cam.H}, need {TARGET_WIDTH}x{TARGET_HEIGHT}")

                print(f"   ✅ Verified Resolution: {cam.W}x{cam.H}")

                # Create VisionIPC buffers
                print(f"   Creating VisionIPC buffers...")
                self.vipc_server.create_buffers(c.stream_type, 20, cam.W, cam.H)
                print(f"   ✅ Buffers created for stream type {c.stream_type.value}")

                self.cameras.append(cam)
                print(f"   ✅ Camera {c.msg_name} ready at {cam.W}x{cam.H}!")

            except Exception as e:
                print(f"   ❌ FAILED to initialize camera {c.msg_name} at 1920x1080: {e}")
                raise  # Don't continue with wrong resolution

        if len(self.cameras) != len(CAMERAS):
            raise RuntimeError("❌ Not all cameras could be initialized at 1920x1080!")

        print(f"\n✅ Successfully initialized {len(self.cameras)}/{len(CAMERAS)} cameras at 1920x1080")

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
        """Camera capture loop"""
        print(f"🎬 Starting camera runner for {cam.cam_type_state} at {cam.W}x{cam.H}")
        rk = Ratekeeper(TARGET_FPS, None)

        try:
            for yuv in cam.read_frames():
                self._send_yuv(yuv, cam.cur_frame_id, cam.cam_type_state, cam.stream_type)
                cam.cur_frame_id += 1

                # Status update every 100 frames
                if cam.cur_frame_id % 100 == 0:
                    print(f"📊 {cam.cam_type_state}: Frame {cam.cur_frame_id} @ {cam.W}x{cam.H}")

                rk.keep_time()

        except Exception as e:
            print(f"❌ Camera runner error for {cam.cam_type_state}: {e}")

    def run(self):
        """Start camera daemon"""
        print(f"\n🏃 Starting {len(self.cameras)} camera threads at 1920x1080...")
        threads = []

        for cam in self.cameras:
            t = threading.Thread(target=self.camera_runner, args=(cam,), daemon=True)
            t.start()
            threads.append(t)
            print(f"✅ Started thread for {cam.cam_type_state} @ {cam.W}x{cam.H}")

        print("🎯 Dual 1080p camera daemon running. Press Ctrl+C to stop.")

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down camera daemon...")

def main():
    DualCamerad1080p().run()

if __name__ == "__main__":
    main()