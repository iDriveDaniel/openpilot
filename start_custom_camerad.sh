#!/bin/bash
# Custom USB Camera Daemon Startup Script
# ========================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERAD_SCRIPT="$SCRIPT_DIR/custom_usb_camerad.py"
CONFIG_SCRIPT="$SCRIPT_DIR/camera_config.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="/tmp/custom_camerad.log"

print_banner() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo "   Custom USB Camera Daemon for OpenPilot"
    echo "============================================================"
    echo -e "${NC}"
}

check_dependencies() {
    echo -e "${YELLOW}🔍 Checking dependencies...${NC}"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 not found${NC}"
        exit 1
    fi

    # Check OpenCV
    if ! python3 -c "import cv2" 2>/dev/null; then
        echo -e "${RED}❌ OpenCV not found. Install with: pip install opencv-python${NC}"
        exit 1
    fi

    # Check numpy
    if ! python3 -c "import numpy" 2>/dev/null; then
        echo -e "${RED}❌ NumPy not found. Install with: pip install numpy${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Dependencies OK${NC}"
}

check_cameras() {
    echo -e "${YELLOW}📹 Checking cameras...${NC}"

    # Get camera devices from config
    ROAD_CAM=$(python3 -c "from camera_config import ROAD_CAMERA_DEVICE; print(ROAD_CAMERA_DEVICE)")
    WIDE_CAM=$(python3 -c "from camera_config import WIDE_CAMERA_DEVICE; print(WIDE_CAMERA_DEVICE)")

    # Check if camera devices exist
    if [ ! -e "/dev/video$ROAD_CAM" ]; then
        echo -e "${RED}❌ Road camera /dev/video$ROAD_CAM not found${NC}"
        exit 1
    fi

    if [ ! -e "/dev/video$WIDE_CAM" ]; then
        echo -e "${RED}❌ Wide camera /dev/video$WIDE_CAM not found${NC}"
        exit 1
    fi

    # Test camera access
    if ! timeout 3 python3 -c "
import cv2
cap = cv2.VideoCapture($ROAD_CAM)
ret, frame = cap.read()
cap.release()
assert ret and frame is not None
" 2>/dev/null; then
        echo -e "${RED}❌ Cannot read from road camera /dev/video$ROAD_CAM${NC}"
        exit 1
    fi

    if ! timeout 3 python3 -c "
import cv2
cap = cv2.VideoCapture($WIDE_CAM)
ret, frame = cap.read()
cap.release()
assert ret and frame is not None
" 2>/dev/null; then
        echo -e "${RED}❌ Cannot read from wide camera /dev/video$WIDE_CAM${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Cameras OK - Road: /dev/video$ROAD_CAM, Wide: /dev/video$WIDE_CAM${NC}"
}

check_openpilot() {
    echo -e "${YELLOW}🚗 Checking OpenPilot environment...${NC}"

    # Check if we're in openpilot directory
    if [ ! -f "SConstruct" ] || [ ! -d "cereal" ]; then
        echo -e "${RED}❌ Not in OpenPilot directory. Please run from openpilot root.${NC}"
        exit 1
    fi

    # Check cereal messaging
    if ! python3 -c "import cereal.messaging" 2>/dev/null; then
        echo -e "${RED}❌ Cereal messaging not available. Build OpenPilot first.${NC}"
        exit 1
    fi

    # Check VisionIPC
    if ! python3 -c "from msgq.visionipc import VisionIpcServer" 2>/dev/null; then
        echo -e "${RED}❌ VisionIPC not available. Build OpenPilot first.${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ OpenPilot environment OK${NC}"
}

stop_existing_camerad() {
    echo -e "${YELLOW}🛑 Stopping existing camerad processes...${NC}"

    # Stop native camerad if running
    if pgrep -f "camerad" > /dev/null; then
        echo "Stopping native camerad..."
        pkill -f "camerad" || true
        sleep 2
    fi

    # Stop our custom camerad if running
    if pgrep -f "custom_usb_camerad.py" > /dev/null; then
        echo "Stopping existing custom camerad..."
        pkill -f "custom_usb_camerad.py" || true
        sleep 2
    fi

    echo -e "${GREEN}✅ Existing processes stopped${NC}"
}

start_camerad() {
    echo -e "${YELLOW}🚀 Starting custom camera daemon...${NC}"

    # Validate configuration first
    if ! python3 "$CONFIG_SCRIPT"; then
        echo -e "${RED}❌ Invalid configuration${NC}"
        exit 1
    fi

    # Start the daemon
    echo "Starting daemon (logging to $LOG_FILE)..."

    if [ "$1" = "--foreground" ] || [ "$1" = "-f" ]; then
        # Run in foreground
        python3 "$CAMERAD_SCRIPT"
    else
        # Run in background
        nohup python3 "$CAMERAD_SCRIPT" > "$LOG_FILE" 2>&1 &
        DAEMON_PID=$!

        # Wait a moment and check if it started successfully
        sleep 3
        if kill -0 "$DAEMON_PID" 2>/dev/null; then
            echo -e "${GREEN}✅ Custom camera daemon started (PID: $DAEMON_PID)${NC}"
            echo "View logs with: tail -f $LOG_FILE"
        else
            echo -e "${RED}❌ Failed to start camera daemon. Check logs: $LOG_FILE${NC}"
            exit 1
        fi
    fi
}

show_status() {
    echo -e "${YELLOW}📊 Camera daemon status:${NC}"

    if pgrep -f "custom_usb_camerad.py" > /dev/null; then
        PID=$(pgrep -f "custom_usb_camerad.py")
        echo -e "${GREEN}✅ Custom camera daemon running (PID: $PID)${NC}"

        # Show resource usage
        if command -v ps &> /dev/null; then
            echo "Resource usage:"
            ps -p "$PID" -o pid,ppid,cpu,pmem,cmd --no-headers
        fi
    else
        echo -e "${RED}❌ Custom camera daemon not running${NC}"
    fi

    # Check for log file
    if [ -f "$LOG_FILE" ]; then
        echo "Last 5 log lines:"
        tail -n 5 "$LOG_FILE"
    fi
}

show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start, -s        Start camera daemon (background)"
    echo "  start -f         Start camera daemon (foreground)"
    echo "  stop             Stop camera daemon"
    echo "  restart          Restart camera daemon"
    echo "  status           Show daemon status"
    echo "  test             Test camera setup"
    echo "  logs             Show daemon logs"
    echo "  help, -h         Show this help"
    echo ""
    echo "Environment variables:"
    echo "  ROAD_CAM=N       Road camera device ID (default: 0)"
    echo "  WIDE_CAM=N       Wide camera device ID (default: 2)"
    echo "  DEBUG_CAMERAD=1  Enable debug mode"
}

# Main script logic
case "${1:-start}" in
    "start" | "-s")
        print_banner
        check_dependencies
        check_openpilot
        check_cameras
        stop_existing_camerad
        start_camerad "$2"
        ;;
    "stop")
        echo -e "${YELLOW}🛑 Stopping camera daemon...${NC}"
        if pgrep -f "custom_usb_camerad.py" > /dev/null; then
            pkill -f "custom_usb_camerad.py"
            echo -e "${GREEN}✅ Camera daemon stopped${NC}"
        else
            echo -e "${YELLOW}⚠️  Camera daemon not running${NC}"
        fi
        ;;
    "restart")
        $0 stop
        sleep 2
        $0 start
        ;;
    "status")
        show_status
        ;;
    "test")
        print_banner
        check_dependencies
        check_openpilot
        check_cameras
        echo -e "${GREEN}✅ All tests passed!${NC}"
        ;;
    "logs")
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo -e "${RED}❌ Log file not found: $LOG_FILE${NC}"
            exit 1
        fi
        ;;
    "help" | "-h")
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac