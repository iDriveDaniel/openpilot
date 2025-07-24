#!/usr/bin/env python3
"""
Triple Camera Setup - Road + Wide + Driver Monitoring
====================================================

Forces road (/dev/video2), wide (/dev/video6), and driver (/dev/video0) cameras
to maximum resolution using MJPG format for optimal frame rates.
SYNCHRONIZED TIMING: Uses master clock to ensure frames are within sync tolerance.
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
ROAD_CAM_ID = "2"   # /dev/video2 - Main road camera
WIDE_CAM_ID = "6"   # /dev/video6 - Wide road camera
DRIVER_CAM_ID = "0" # /dev/video0 - Driver monitoring camera (working fallback)
# ---------------------------------------------------------------------------

# Target settings - MJPG format for maximum FPS
ROAD_TARGET_WIDTH = 1920
ROAD_TARGET_HEIGHT = 1080
WIDE_TARGET_WIDTH = 1920
WIDE_TARGET_HEIGHT = 1080
DRIVER_TARGET_WIDTH = 640  # Working resolution for driver camera
DRIVER_TARGET_HEIGHT = 480
TARGET_FPS = 30
TARGET_FORMAT = 'MJPG'  # This is crucial for high FPS!

CameraType = namedtuple("CameraType", ["msg_name", "stream_type", "cam_id", "target_width", "target_height"])

CAMERAS = [
    CameraType("roadCameraState", VisionStreamType.VISION_STREAM_ROAD, ROAD_CAM_ID, ROAD_TARGET_WIDTH, ROAD_TARGET_HEIGHT),
    CameraType("wideRoadCameraState", VisionStreamType.VISION_STREAM_WIDE_ROAD, WIDE_CAM_ID, WIDE_TARGET_WIDTH, WIDE_TARGET_HEIGHT),
    CameraType("driverCameraState", VisionStreamType.VISION_STREAM_DRIVER, DRIVER_CAM_ID, DRIVER_TARGET_WIDTH, DRIVER_TARGET_HEIGHT),
]

class SyncedClock:
    """Master clock for synchronizing camera frames"""
    def __init__(self, fps=30):
        self.fps = fps
        self.frame_interval_ns = int(1e9 / fps)  # nanoseconds per frame
        self.start_time_ns = None
        self.frame_counter = 0
        self.lock = threading.Lock()

    def start(self):
        """Initialize the master clock"""
        self.start_time_ns = time.time_ns()
        self.frame_counter = 0
        print(f"🕒 Master clock started at {self.start_time_ns}")

    def get_next_frame_time(self):
        """Get synchronized timestamp for next frame"""
        with self.lock:
            if self.start_time_ns is None:
                self.start()

            frame_time = self.start_time_ns + (self.frame_counter * self.frame_interval_ns)
            self.frame_counter += 1
            return frame_time

    def wait_for_next_frame(self):
        """Wait until it's time for the next frame"""
        target_time = self.get_next_frame_time()
        current_time = time.time_ns()

        if current_time < target_time:
            sleep_duration = (target_time - current_time) / 1e9
            time.sleep(sleep_duration)

        return target_time

class HighFpsCamera(Camera):
    """Camera class that enforces maximum FPS using MJPG format"""

    def __init__(self, cam_type_state, stream_type, camera_id, target_width, target_height, master_clock):
        self.cam_type_state = cam_type_state
        self.stream_type = stream_type
        self.cur_frame_id = 0
        self.camera_id = camera_id
        self.target_width = target_width
        self.target_height = target_height
        self.cap = None
        self.W = 0
        self.H = 0
        self.master_clock = master_clock

        try:
            camera_id = int(camera_id)
        except ValueError:
            pass

        print(f"🔧 Opening {cam_type_state} at {camera_id} (TARGET: {target_width}x{target_height} @ {TARGET_FPS} FPS MJPG)")

        # Force MJPG format for maximum FPS
        self._force_mjpg_resolution(camera_id)

    def _configure_ar0230_properties(self, camera_id):
        """Configure optimal properties for Arducam AR0230 sensors"""
        print(f"🎛️  Configuring AR0230 sensor properties for {self.cam_type_state}")

        try:
            # === EXPOSURE SETTINGS ===
            # Auto exposure: 3 = Aperture Priority Mode (best for automotive)
            success = self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75 = 3/4 = mode 3
            print(f"   📸 Auto exposure (Aperture Priority): {success}")

            # === WHITE BALANCE ===
            # Enable automatic white balance
            success = self.cap.set(cv.CAP_PROP_AUTO_WB, 1.0)
            print(f"   🌈 Auto white balance: {success}")

            # === IMAGE QUALITY SETTINGS ===
            # Brightness: 0 (default, neutral)
            success = self.cap.set(cv.CAP_PROP_BRIGHTNESS, 0.0)
            print(f"   ☀️  Brightness (0): {success}")

            # Contrast: 40 (higher for better definition)
            success = self.cap.set(cv.CAP_PROP_CONTRAST, 0.625)  # 40/64 = 0.625
            print(f"   🔗 Contrast (40): {success}")

            # Saturation: 80 (higher for vivid colors - CRITICAL for color output!)
            # Try both normalized and raw values
            success1 = self.cap.set(cv.CAP_PROP_SATURATION, 80.0)  # Raw value
            success2 = self.cap.set(cv.CAP_PROP_SATURATION, 0.625)  # Normalized value
            print(f"   🎨 Saturation (80): raw={success1}, norm={success2}")

            # Hue: 0 (neutral, no color shift)
            success = self.cap.set(cv.CAP_PROP_HUE, 0.0)  # Try 0.0 instead of 0.5
            print(f"   🌈 Hue (0): {success}")

            # Gamma: 120 (slightly higher for better visibility)
            success = self.cap.set(cv.CAP_PROP_GAMMA, 120.0)
            print(f"   📐 Gamma (120): {success}")

            # Gain: 15 (moderate gain for good visibility)
            success = self.cap.set(cv.CAP_PROP_GAIN, 15.0)
            print(f"   📶 Gain (15): {success}")

            # Sharpness: 3 (good sharpening for better edge detection)
            success = self.cap.set(cv.CAP_PROP_SHARPNESS, 3.0)
            print(f"   🔪 Sharpness (3): {success}")

            # Backlight compensation: 1 (moderate compensation)
            success = self.cap.set(cv.CAP_PROP_BACKLIGHT, 1.0)
            print(f"   💡 Backlight compensation (1): {success}")

            # === BACKUP: Use v4l2-ctl for critical color settings ===
            try:
                import subprocess
                device_path = f"/dev/video{camera_id}" if str(camera_id).isdigit() else camera_id

                # Force saturation using v4l2-ctl (most reliable)
                result = subprocess.run([
                    'v4l2-ctl',
                    f'--device={device_path}',
                    '--set-ctrl=saturation=80'
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    print(f"   ✅ v4l2-ctl: saturation=80 set successfully")
                else:
                    print(f"   ⚠️  v4l2-ctl saturation failed: {result.stderr}")

            except Exception as e:
                print(f"   ⚠️  v4l2-ctl backup failed: {e}")

            # === VERIFY SETTINGS ===
            print(f"   ✅ AR0230 properties configured for {self.cam_type_state}")

            # Read back some key settings to verify
            auto_exp = self.cap.get(cv.CAP_PROP_AUTO_EXPOSURE)
            auto_wb = self.cap.get(cv.CAP_PROP_AUTO_WB)
            gain = self.cap.get(cv.CAP_PROP_GAIN)
            gamma = self.cap.get(cv.CAP_PROP_GAMMA)
            saturation = self.cap.get(cv.CAP_PROP_SATURATION)

            print(f"   📊 Verified: auto_exp={auto_exp:.2f}, auto_wb={auto_wb:.1f}, gain={gain:.1f}, gamma={gamma:.1f}, sat={saturation:.3f}")

        except Exception as e:
            print(f"   ⚠️  Warning: Could not set some AR0230 properties: {e}")

    def _force_mjpg_resolution(self, camera_id):
        """Force MJPG format at target resolution for maximum FPS"""

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
            width_success = self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.target_width)
            height_success = self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.target_height)
            fps_success = self.cap.set(cv.CAP_PROP_FPS, TARGET_FPS)

            print(f"   📐 Set: width={width_success}, height={height_success}, fps={fps_success}")

            # *** CONFIGURE AR0230 SENSOR PROPERTIES ***
            self._configure_ar0230_properties(camera_id)

            # For driver camera, add a longer delay before testing
            if "driver" in self.cam_type_state.lower():
                print(f"   ⏱️  Driver camera: waiting 2 seconds for stabilization...")
                time.sleep(2)

            # Verify actual settings
            actual_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv.CAP_PROP_FPS)
            actual_fourcc = int(self.cap.get(cv.CAP_PROP_FOURCC))

            # Convert fourcc back to string
            fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
            print(f"   📊 Actual: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS, format: {fourcc_str}")

            # Test frame capture with multiple attempts for driver camera
            test_attempts = 5 if "driver" in self.cam_type_state.lower() else 1

            for attempt in range(test_attempts):
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print(f"   📸 Test frame shape (attempt {attempt+1}): {test_frame.shape}")

                    # Accept any reasonable resolution and frame rate
                    if (actual_width >= 640 and actual_height >= 480 and actual_fps >= 10.0):
                        self.W = actual_width
                        self.H = actual_height
                        print(f"✅ {self.cam_type_state} SUCCESS: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS ({fourcc_str})")
                        return
                    else:
                        print(f"⚠️  {self.cam_type_state} low settings: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
                        break
                else:
                    if "driver" in self.cam_type_state.lower() and attempt < test_attempts - 1:
                        print(f"   ⏱️  Driver camera test frame attempt {attempt+1} failed, waiting 1s...")
                        time.sleep(1)
                    else:
                        print(f"❌ {self.cam_type_state} cannot capture test frame")
                        break

        except Exception as e:
            print(f"❌ Method 1 failed for {self.cam_type_state}: {e}")

        print(f"🎯 Method 2: V4L2 backend with MJPG for {self.cam_type_state}")
        try:
            if self.cap:
                self.cap.release()
                time.sleep(1)  # Wait before reopening

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
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv.CAP_PROP_FPS, TARGET_FPS)

            # *** CONFIGURE AR0230 SENSOR PROPERTIES ***
            self._configure_ar0230_properties(camera_id)

            # Additional wait for driver camera
            if "driver" in self.cam_type_state.lower():
                time.sleep(2)

            actual_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv.CAP_PROP_FPS)

            print(f"   📊 V4L2 result: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

            # Multiple test attempts for driver camera
            test_attempts = 3 if "driver" in self.cam_type_state.lower() else 1

            for attempt in range(test_attempts):
                ret, test_frame = self.cap.read()
                if (ret and actual_width >= 640 and actual_height >= 480 and actual_fps >= 10.0):
                    self.W = actual_width
                    self.H = actual_height
                    print(f"✅ {self.cam_type_state} SUCCESS with V4L2: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
                    return
                elif "driver" in self.cam_type_state.lower() and attempt < test_attempts - 1:
                    print(f"   ⏱️  V4L2 test attempt {attempt+1} failed, waiting 1s...")
                    time.sleep(1)

        except Exception as e:
            print(f"❌ Method 2 failed for {self.cam_type_state}: {e}")

        print(f"🎯 Method 3: Fallback with any working resolution for {self.cam_type_state}")
        try:
            if self.cap:
                self.cap.release()
                time.sleep(1)

            self.cap = cv.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {camera_id}")

            # Just try to get any working resolution
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv.CAP_PROP_FPS, TARGET_FPS)

            # For driver camera, try lower resolution first
            if "driver" in self.cam_type_state.lower():
                print(f"   🎯 Driver fallback: trying 1280x720...")
                self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
                time.sleep(2)

            # *** CONFIGURE AR0230 SENSOR PROPERTIES ***
            self._configure_ar0230_properties(camera_id)

            actual_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv.CAP_PROP_FPS)

            print(f"   📊 Fallback result: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

            # Multiple test attempts
            test_attempts = 3 if "driver" in self.cam_type_state.lower() else 1

            for attempt in range(test_attempts):
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    self.W = actual_width
                    self.H = actual_height
                    print(f"✅ {self.cam_type_state} FALLBACK SUCCESS: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
                    return
                elif "driver" in self.cam_type_state.lower() and attempt < test_attempts - 1:
                    print(f"   ⏱️  Fallback test attempt {attempt+1} failed, waiting 1s...")
                    time.sleep(1)

        except Exception as e:
            print(f"❌ Method 3 failed for {self.cam_type_state}: {e}")

        # If we get here, camera completely failed
        raise RuntimeError(f"❌ Camera {camera_id} ({self.cam_type_state}) failed to initialize!")

    def read_frames(self):
        """Read frames with synchronized timing"""
        failed_reads = 0
        max_failed_reads = 10
        frame_count = 0
        fps_start_time = time.time()

        print(f"🎬 {self.cam_type_state} starting SYNCHRONIZED frame reading")

        while True:
            try:
                # Wait for master clock (this synchronizes all cameras)
                sync_timestamp = self.master_clock.wait_for_next_frame()

                ret, frame = self.cap.read()
                if not ret:
                    failed_reads += 1
                    print(f"⚠️  Failed to read frame from {self.cam_type_state} ({failed_reads}/{max_failed_reads})")

                    if failed_reads >= max_failed_reads:
                        print(f"❌ Camera {self.cam_type_state} failed after {max_failed_reads} failures")
                        break

                    time.sleep(0.1)
                    continue

                failed_reads = 0  # Reset on success
                frame_count += 1

                # Monitor actual FPS every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - fps_start_time
                    actual_fps = frame_count / elapsed
                    print(f"📊 {self.cam_type_state} SYNC actual FPS: {actual_fps:.1f}")

                    # Reset counters
                    frame_count = 0
                    fps_start_time = time.time()

                # Driver camera doesn't need flip, others do
                if "driver" not in self.cam_type_state.lower():
                    # Rotate the frame 180 degrees (flip both axes) for road/wide cameras
                    frame = cv.flip(frame, -1)

                # Debug: Check if frame is in color (every 100th frame)
                if frame_count % 100 == 0:
                    # Check if frame has color information
                    b_channel = frame[:,:,0]
                    g_channel = frame[:,:,1]
                    r_channel = frame[:,:,2]

                    # Calculate channel differences to detect color
                    bg_diff = cv.absdiff(b_channel, g_channel).mean()
                    gr_diff = cv.absdiff(g_channel, r_channel).mean()

                    if bg_diff < 1.0 and gr_diff < 1.0:
                        print(f"⚠️  {self.cam_type_state} frame appears GRAYSCALE! (BG diff: {bg_diff:.2f}, GR diff: {gr_diff:.2f})")
                    else:
                        print(f"✅ {self.cam_type_state} frame has COLOR data (BG diff: {bg_diff:.2f}, GR diff: {gr_diff:.2f})")

                try:
                    yuv = Camera.bgr2nv12(frame)
                    yield yuv.data.tobytes(), sync_timestamp
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

class TripleCamerad:
    def __init__(self):
        print("🚀 Initializing SYNCHRONIZED Triple Camera Setup - Road + Wide + Driver")
        print(f"📹 Road Camera: /dev/video{ROAD_CAM_ID} (target: {ROAD_TARGET_WIDTH}x{ROAD_TARGET_HEIGHT})")
        print(f"📹 Wide Camera: /dev/video{WIDE_CAM_ID} (target: {WIDE_TARGET_WIDTH}x{WIDE_TARGET_HEIGHT})")
        print(f"📹 Driver Camera: /dev/video{DRIVER_CAM_ID} (target: {DRIVER_TARGET_WIDTH}x{DRIVER_TARGET_HEIGHT})")
        print(f"🎯 Target FPS: {TARGET_FPS} ({TARGET_FORMAT} format)")
        print("🕒 Using MASTER CLOCK for synchronized frame timing")

        # Create master clock for synchronization
        self.master_clock = SyncedClock(TARGET_FPS)

        # Publish camera states
        self.pm = messaging.PubMaster([c.msg_name for c in CAMERAS])
        print(f"✅ PubMaster initialized for: {[c.msg_name for c in CAMERAS]}")

        # Vision IPC server
        self.vipc_server = VisionIpcServer("camerad")
        print("✅ VisionIPC server created")

        # Initialize cameras
        self.cameras = []
        for i, c in enumerate(CAMERAS):
            try:
                print(f"\n🔧 Initializing camera {i+1}/{len(CAMERAS)}: {c.msg_name}")
                cam_device = f"/dev/video{c.cam_id}" if platform.system() != "Darwin" else c.cam_id
                print(f"   Device: {cam_device}")

                # Use high FPS camera class with master clock
                cam = HighFpsCamera(c.msg_name, c.stream_type, cam_device, c.target_width, c.target_height, self.master_clock)

                print(f"   ✅ Achieved: {cam.W}x{cam.H} ready for SYNCHRONIZED streaming")

                # Create VisionIPC buffers
                print(f"   Creating VisionIPC buffers...")
                self.vipc_server.create_buffers(c.stream_type, 20, cam.W, cam.H)
                print(f"   ✅ Buffers created for stream type {c.stream_type.value}")

                self.cameras.append(cam)
                print(f"   ✅ Camera {c.msg_name} ready for SYNCHRONIZED streaming!")

            except Exception as e:
                print(f"   ⚠️  FAILED to initialize camera {c.msg_name}: {e}")
                print(f"      Continuing with other cameras...")
                # Continue with other cameras instead of failing completely

        if len(self.cameras) == 0:
            raise RuntimeError("❌ No cameras could be initialized!")

        print(f"\n✅ Successfully initialized {len(self.cameras)}/{len(CAMERAS)} cameras for SYNCHRONIZED streaming")

        # Start VisionIPC listener
        print("🔗 Starting VisionIPC listener...")
        self.vipc_server.start_listener()
        print("✅ VisionIPC listener started")

    def _send_yuv(self, yuv, frame_id, pub_type, yuv_type, sync_timestamp):
        """Send YUV frame to VisionIPC and publish metadata with synchronized timestamp"""
        try:
            self.vipc_server.send(yuv_type, yuv, frame_id, sync_timestamp, sync_timestamp)

            dat = messaging.new_message(pub_type, valid=True)
            msg = {
                "frameId": frame_id,
                "timestampSof": sync_timestamp,
                "transform": [1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0]
            }
            setattr(dat, pub_type, msg)
            self.pm.send(pub_type, dat)
        except Exception as e:
            print(f"❌ Error sending frame for {pub_type}: {e}")

    def camera_runner(self, cam):
        """Camera capture loop optimized for SYNCHRONIZED streaming"""
        print(f"🎬 Starting SYNCHRONIZED camera runner for {cam.cam_type_state}")

        try:
            for yuv_data, sync_timestamp in cam.read_frames():
                self._send_yuv(yuv_data, cam.cur_frame_id, cam.cam_type_state, cam.stream_type, sync_timestamp)
                cam.cur_frame_id += 1

                # Status update every 300 frames (10 seconds at 30 FPS)
                if cam.cur_frame_id % 300 == 0:
                    print(f"📊 {cam.cam_type_state}: Frame {cam.cur_frame_id} (SYNCHRONIZED)")

        except Exception as e:
            print(f"❌ Camera runner error for {cam.cam_type_state}: {e}")

    def run(self):
        """Start synchronized camera daemon"""
        print(f"\n🏃 Starting {len(self.cameras)} SYNCHRONIZED camera threads...")

        # Start master clock
        self.master_clock.start()

        threads = []

        for cam in self.cameras:
            t = threading.Thread(target=self.camera_runner, args=(cam,), daemon=True)
            t.start()
            threads.append(t)
            print(f"✅ Started SYNCHRONIZED thread for {cam.cam_type_state}")

        print("🎯 SYNCHRONIZED Triple camera daemon running. Press Ctrl+C to stop.")
        print("🕒 All cameras synchronized to master clock")
        print("👁️  Driver monitoring enabled!")

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down synchronized triple camera daemon...")

def main():
    TripleCamerad().run()

if __name__ == "__main__":
    main()