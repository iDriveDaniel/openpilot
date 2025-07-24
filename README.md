# AR2030 Camera Video Recorder

Professional video recording from AR2030 camera sensor with multiple format support, hardware acceleration, and comprehensive configuration options.

## 📹 AR2030 Sensor Specifications

- **Resolution**: 1920x1200 (2.3MP)
- **Frame rates**: Up to 60fps at full resolution
- **Output formats**: Raw Bayer, YUV422, RGB
- **Interface**: MIPI CSI-2 4-lane
- **HDR support**: Yes
- **Global shutter**: Yes
- **Manufacturer**: ON Semiconductor

## ✨ Features

- ✅ **Multiple video codecs** (H.264, H.265, VP9, MJPEG)
- ✅ **Hardware acceleration support**
- ✅ **Real-time compression**
- ✅ **Configurable quality settings**
- ✅ **Automatic file rotation**
- ✅ **Performance monitoring**
- ✅ **Error recovery**
- ✅ **Predefined recording presets**
- ✅ **AR2030-specific camera controls**
- ✅ **Comprehensive logging**

## 🔧 Requirements

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-opencv v4l-utils ffmpeg

# Fedora/RHEL
sudo dnf install python3-opencv v4l-utils ffmpeg

# Arch Linux
sudo pacman -S python-opencv v4l-utils ffmpeg
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Recording
```bash
# Start recording with default settings (1920x1200@30fps)
python3 ar2030_video_recorder.py

# Record for 5 minutes with high quality preset
python3 ar2030_video_recorder.py --preset high_quality --duration 300

# Fast recording for real-time applications
python3 ar2030_video_recorder.py --preset fast
```

### List Available Presets
```bash
python3 ar2030_video_recorder.py --list-presets
```

Output:
```
Available recording presets:
==================================================
high_quality - High quality recording (1920x1200@30fps)
             1920x1200 @ 30fps, mp4v
standard     - Standard quality recording (1280x800@30fps)
             1280x800 @ 30fps, XVID
fast         - Fast recording for real-time (640x400@60fps)
             640x400 @ 60fps, MJPG
surveillance - Surveillance mode (1920x1200@15fps, space efficient)
             1920x1200 @ 15fps, H264
```

## 📋 Recording Presets

| Preset | Resolution | FPS | Codec | Use Case |
|--------|------------|-----|--------|----------|
| `high_quality` | 1920x1200 | 30 | mp4v | High quality recording |
| `standard` | 1280x800 | 30 | XVID | Standard quality |
| `fast` | 640x400 | 60 | MJPG | Real-time applications |
| `surveillance` | 1920x1200 | 15 | H264 | Long-term monitoring |

## 🎛️ Usage Examples

### Custom Configuration
```bash
# Custom resolution and codec
python3 ar2030_video_recorder.py \
    --width 1280 --height 720 \
    --fps 60 --codec H264 \
    --output ./my_recordings

# Surveillance setup with file rotation
python3 ar2030_video_recorder.py \
    --preset surveillance \
    --max-file-size 1024 \
    --output ./security_footage \
    --prefix security_cam

# High frame rate recording
python3 ar2030_video_recorder.py \
    --width 640 --height 480 \
    --fps 60 --codec MJPG \
    --pixel-format MJPG
```

### Advanced Options
```bash
# Disable auto file rotation
python3 ar2030_video_recorder.py --no-auto-rotate

# Specify different camera device
python3 ar2030_video_recorder.py --device /dev/video1

# Adjust camera buffer size
python3 ar2030_video_recorder.py --buffer-size 5
```

## 📊 Command Line Options

### Recording Parameters
- `--width` - Video width (default: 1920)
- `--height` - Video height (default: 1200)
- `--fps` - Frame rate (default: 30)
- `--codec` - Video codec: mp4v, XVID, MJPG, H264, H265, VP9 (default: mp4v)
- `--quality` - Video quality 1-100 (default: 85)

### Device and Output
- `--device` - Camera device path (default: /dev/video0)
- `--output` - Output directory (default: ./recordings)
- `--prefix` - Filename prefix (default: ar2030)

### Recording Control
- `--duration` - Recording duration in seconds (0 = infinite)
- `--max-file-size` - Max file size in MB before rotation (default: 2048)
- `--no-auto-rotate` - Disable automatic file rotation

### Advanced Options
- `--pixel-format` - Camera pixel format: YUYV, MJPG, RGB24 (default: YUYV)
- `--buffer-size` - Camera buffer size (default: 3)

## 📁 Output Files

### File Naming
Files are automatically named with timestamps:
```
ar2030_20240101_143022.mp4
ar2030_20240101_144523.avi
security_cam_20240101_150030.mp4
```

### Automatic Rotation
Files are automatically rotated when:
- Maximum duration is reached (default: 1 hour)
- Maximum file size is reached (default: 2GB)
- Can be disabled with `--no-auto-rotate`

## 🔍 Monitoring and Logging

### Real-time Status
The recorder displays real-time information:
```
📸 Recording: 1847 frames, 30 fps, 61.6s
💾 File size: 45.2 MB
```

### Log Files
Detailed logs are saved to `recordings/recorder.log`:
```
2024-01-01 14:30:22,123 - INFO - ✅ AR2030 camera detected on /dev/video0
2024-01-01 14:30:22,145 - INFO - 📐 Resolution: 1920x1200 (requested: 1920x1200)
2024-01-01 14:30:22,146 - INFO - 🎬 Frame rate: 30.0fps (requested: 30fps)
```

## 🎥 Supported Video Codecs

| Codec | Extension | Quality | Speed | Use Case |
|-------|-----------|---------|--------|----------|
| mp4v | .mp4 | High | Medium | General purpose |
| H264 | .mp4 | High | Fast | Streaming, web |
| H265 | .mp4 | Very High | Slow | Archival |
| XVID | .avi | Medium | Fast | Compatibility |
| MJPG | .avi | Low | Very Fast | Real-time |
| VP9 | .mp4 | High | Slow | Web, modern |

## 🔧 Camera Controls

The recorder automatically configures AR2030-specific controls:
- Auto exposure
- Auto white balance
- Auto gain
- Brightness adjustment
- Contrast enhancement
- Saturation control
- Sharpness optimization

## 🚨 Troubleshooting

### Camera Not Detected
```bash
# Check available cameras
ls /dev/video*

# Check camera information
v4l2-ctl --device /dev/video0 --info

# List supported formats
v4l2-ctl --device /dev/video0 --list-formats-ext
```

### Permission Issues
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Set device permissions
sudo chmod 666 /dev/video0
```

### Codec Issues
```bash
# Check OpenCV codec support
python3 -c "import cv2; print(cv2.getBuildInformation())"

# Install additional codecs
sudo apt install ubuntu-restricted-extras
```

## 📈 Performance Tips

1. **Use MJPG for high frame rates** (60+ fps)
2. **Use H264 for efficient storage** (surveillance)
3. **Reduce resolution for real-time** applications
4. **Increase buffer size** for stable capture
5. **Use SSD storage** for high bitrate recording

## 🔄 Exit and Cleanup

The recorder handles graceful shutdown:
- **Ctrl+C** - Stop recording and save file
- **SIGTERM** - Graceful shutdown
- **Automatic cleanup** on errors

## 📋 System Requirements

- **OS**: Linux (Ubuntu, Fedora, Arch, etc.)
- **Python**: 3.8+
- **OpenCV**: 4.8+
- **Camera**: AR2030 sensor via V4L2
- **Storage**: Fast storage recommended for high bitrates

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source. Use it responsibly!

---

**🎬 Happy Recording with AR2030! 📹**
