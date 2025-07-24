#!/bin/bash
"""
BNO055 Setup Script for Openpilot

This script helps set up the BNO055 sensor for use with openpilot:
1. Installs required I2C tools
2. Configures I2C permissions
3. Tests BNO055 detection
4. Provides usage instructions

Usage:
    chmod +x setup_bno055.sh
    sudo ./setup_bno055.sh
"""

set -e  # Exit on any error

echo "🚀 BNO055 Setup Script for Openpilot"
echo "===================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run as root (use sudo)"
    echo "Usage: sudo $0"
    exit 1
fi

# Get the actual user (not root when using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
ACTUAL_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)

echo "Setting up BNO055 for user: $ACTUAL_USER"

# Step 1: Install I2C tools
echo ""
echo "📦 Step 1: Installing I2C tools..."
apt-get update -qq
apt-get install -y i2c-tools python3-smbus

# Step 2: Enable I2C interface
echo ""
echo "🔧 Step 2: Enabling I2C interface..."

# Check if I2C is already enabled
if ! grep -q "^dtparam=i2c_arm=on" /boot/config.txt; then
    echo "dtparam=i2c_arm=on" >> /boot/config.txt
    echo "   I2C enabled in /boot/config.txt"
else
    echo "   I2C already enabled in /boot/config.txt"
fi

# Load I2C kernel modules
modprobe i2c-dev 2>/dev/null || echo "   i2c-dev module already loaded"

# Add to /etc/modules for persistent loading
if ! grep -q "^i2c-dev" /etc/modules; then
    echo "i2c-dev" >> /etc/modules
    echo "   Added i2c-dev to /etc/modules"
fi

# Step 3: Set up I2C permissions
echo ""
echo "🔐 Step 3: Setting up I2C permissions..."

# Add user to i2c group
usermod -a -G i2c "$ACTUAL_USER"
echo "   Added $ACTUAL_USER to i2c group"

# Set I2C device permissions
if [ -e /dev/i2c-1 ]; then
    chown root:i2c /dev/i2c-1
    chmod 664 /dev/i2c-1
    echo "   Set permissions for /dev/i2c-1"
else
    echo "   ⚠️  /dev/i2c-1 not found (may appear after reboot)"
fi

# Create udev rule for persistent I2C permissions
cat > /etc/udev/rules.d/99-i2c.rules << 'EOF'
SUBSYSTEM=="i2c-dev", GROUP="i2c", MODE="0664"
EOF
echo "   Created udev rule for I2C permissions"

# Step 4: Test I2C detection
echo ""
echo "🔍 Step 4: Testing I2C bus..."

if [ -e /dev/i2c-1 ]; then
    echo "Available I2C buses:"
    i2cdetect -l

    echo ""
    echo "Scanning I2C bus 1 for devices..."
    i2cdetect -y 1

    echo ""
    if i2cdetect -y 1 | grep -q " 28 "; then
        echo "✅ BNO055 detected at address 0x28!"
    else
        echo "❌ BNO055 not detected at address 0x28"
        echo "   Check your wiring:"
        echo "   BNO055 VCC  -> 3.3V or 5V"
        echo "   BNO055 GND  -> GND"
        echo "   BNO055 SDA  -> GPIO 2 (Pin 3)"
        echo "   BNO055 SCL  -> GPIO 3 (Pin 5)"
    fi
else
    echo "⚠️  I2C bus not available yet (may require reboot)"
fi

# Step 5: Create test script
echo ""
echo "📝 Step 5: Creating test commands..."

# Create a quick test command
cat > "$ACTUAL_HOME/test_bno055.sh" << 'EOF'
#!/bin/bash
echo "🔍 Quick BNO055 Test"
echo "==================="

echo "1. Checking I2C bus..."
if [ -e /dev/i2c-1 ]; then
    echo "✅ I2C bus 1 available"
else
    echo "❌ I2C bus 1 not available"
    exit 1
fi

echo ""
echo "2. Scanning for BNO055..."
if i2cdetect -y 1 | grep -q " 28 "; then
    echo "✅ BNO055 detected at address 0x28"
else
    echo "❌ BNO055 not detected at address 0x28"
    echo "Check your wiring and power supply"
    exit 1
fi

echo ""
echo "3. Testing BNO055 communication..."
cd /home/beast/Daniel/openpilot/system/sensord
python3 test_bno055_hardware.py
EOF

chown "$ACTUAL_USER:$ACTUAL_USER" "$ACTUAL_HOME/test_bno055.sh"
chmod +x "$ACTUAL_HOME/test_bno055.sh"

# Step 6: Instructions
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "Next steps:"
echo "1. 🔄 REBOOT your system to ensure I2C is fully enabled:"
echo "   sudo reboot"
echo ""
echo "2. 🔌 After reboot, connect your BNO055:"
echo "   BNO055 VCC  -> 3.3V or 5V"
echo "   BNO055 GND  -> GND"
echo "   BNO055 SDA  -> GPIO 2 (Pin 3)"
echo "   BNO055 SCL  -> GPIO 3 (Pin 5)"
echo ""
echo "3. 🧪 Test the BNO055:"
echo "   $ACTUAL_HOME/test_bno055.sh"
echo ""
echo "4. 🚀 Run the sensor daemon:"
echo "   cd /home/beast/Daniel/openpilot/system/sensord"
echo "   python3 custom_sensord.py"
echo ""
echo "5. 📊 Monitor sensor data:"
echo "   python3 example_usage.py"
echo ""

if [ -e /dev/i2c-1 ] && i2cdetect -y 1 | grep -q " 28 "; then
    echo "✅ BNO055 is ready to use!"
else
    echo "⚠️  BNO055 setup incomplete - check hardware connection after reboot"
fi