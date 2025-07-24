import av
import cv2 as cv

class Camera:
  def __init__(self, cam_type_state, stream_type, camera_id):
    try:
      camera_id = int(camera_id)
    except ValueError: # allow strings, ex: /dev/video0
      pass
    self.cam_type_state = cam_type_state
    self.stream_type = stream_type
    self.cur_frame_id = 0

    print(f"Opening {cam_type_state} at {camera_id}")

    self.cap = cv.VideoCapture(camera_id)

    if not self.cap.isOpened():
      raise RuntimeError(f"Failed to open camera {camera_id} for {cam_type_state}")

    # Set camera properties
    self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920.0)
    self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080.0)
    self.cap.set(cv.CAP_PROP_FPS, 30.0)

    # Get actual properties
    self.W = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
    self.H = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    actual_fps = self.cap.get(cv.CAP_PROP_FPS)

    print(f"✅ {cam_type_state} initialized: {int(self.W)}x{int(self.H)} @ {actual_fps:.1f} FPS")

    # Test read one frame to make sure camera works
    ret, test_frame = self.cap.read()
    if not ret:
      raise RuntimeError(f"Camera {camera_id} opened but cannot read frames for {cam_type_state}")
    print(f"✅ {cam_type_state} test frame successful: {test_frame.shape}")

  @classmethod
  def bgr2nv12(self, bgr):
    frame = av.VideoFrame.from_ndarray(bgr, format='bgr24')
    return frame.reformat(format='nv12').to_ndarray()

  def read_frames(self):
    failed_reads = 0
    max_failed_reads = 10

    while True:
      ret, frame = self.cap.read()
      if not ret:
        failed_reads += 1
        print(f"⚠️  Failed to read frame from {self.cam_type_state} ({failed_reads}/{max_failed_reads})")

        if failed_reads >= max_failed_reads:
          print(f"❌ Camera {self.cam_type_state} failed after {max_failed_reads} consecutive read failures")
          break

        # Brief pause before retry
        import time
        time.sleep(0.1)
        continue

      # Reset failed read counter on successful read
      failed_reads = 0

      # Rotate the frame 180 degrees (flip both axes)
      frame = cv.flip(frame, -1)

      try:
        yuv = Camera.bgr2nv12(frame)
        yield yuv.data.tobytes()
      except Exception as e:
        print(f"⚠️  YUV conversion error for {self.cam_type_state}: {e}")
        continue

    print(f"🔚 Camera {self.cam_type_state} read loop ended")
    self.cap.release()
