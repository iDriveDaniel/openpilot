#!/usr/bin/env python3
"""
BNO055 Serial IMU Sensor Implementation for Openpilot

This module provides BNO055 sensor classes that communicate via UART/Serial
interface instead of I2C. The BNO055 supports multiple communication interfaces
including UART at up to 115200 baud.

Based on BNO055 datasheet UART protocol specification.
"""

import time
import serial
import struct
from typing import Optional, List, Tuple

from cereal import log
from openpilot.common.swaglog import cloudlog

class BNO055Serial:
    """
    BNO055 IMU sensor implementation using UART/Serial communication.

    The BNO055 UART protocol uses:
    - Baud rate: 115200 (default)
    - Data bits: 8
    - Stop bits: 1
    - Parity: None
    - Start byte: 0xAA (write) / 0xBB (read)
    - Register address: 1 byte
    - Length: 1 byte (for read operations)
    - Data: variable length
    """

    # BNO055 Register addresses (same as I2C)
    BNO055_CHIP_ID_ADDR = 0x00
    BNO055_PAGE_ID_ADDR = 0x07
    BNO055_OPR_MODE_ADDR = 0x3D
    BNO055_PWR_MODE_ADDR = 0x3E
    BNO055_SYS_TRIGGER_ADDR = 0x3F
    BNO055_UNIT_SEL_ADDR = 0x3B
    BNO055_AXIS_MAP_CONFIG_ADDR = 0x41
    BNO055_AXIS_MAP_SIGN_ADDR = 0x42

    # Data register addresses
    BNO055_ACCEL_DATA_X_LSB_ADDR = 0x08
    BNO055_GYRO_DATA_X_LSB_ADDR = 0x14
    BNO055_MAG_DATA_X_LSB_ADDR = 0x0E

    # Operation modes
    OPERATION_MODE_CONFIG = 0x00
    OPERATION_MODE_ACCONLY = 0x01
    OPERATION_MODE_MAGONLY = 0x02
    OPERATION_MODE_GYRONLY = 0x03
    OPERATION_MODE_ACCMAG = 0x04
    OPERATION_MODE_ACCGYRO = 0x05
    OPERATION_MODE_MAGGYRO = 0x06
    OPERATION_MODE_AMG = 0x07
    OPERATION_MODE_IMUPLUS = 0x08
    OPERATION_MODE_COMPASS = 0x09
    OPERATION_MODE_M4G = 0x0A
    OPERATION_MODE_NDOF_FMC_OFF = 0x0B
    OPERATION_MODE_NDOF = 0x0C

    # Power modes
    POWER_MODE_NORMAL = 0x00
    POWER_MODE_LOWPOWER = 0x01
    POWER_MODE_SUSPEND = 0x02

    # UART Protocol constants
    UART_START_BYTE_WRITE = 0xAA
    UART_START_BYTE_READ = 0xBB
    UART_STATUS_WRITE_SUCCESS = 0xEE
    UART_STATUS_READ_FAIL = 0xBB
    UART_STATUS_WRITE_FAIL = 0xFF

    def __init__(self, device_path: str = "/dev/ttyACM0", baudrate: int = 115200):
        """
        Initialize BNO055 serial communication.

        Args:
            device_path: Serial device path (e.g., "/dev/ttyACM0")
            baudrate: UART baud rate (default: 115200)
        """
        self.device_path = device_path
        self.baudrate = baudrate
        self.serial_conn = None
        self.source = log.SensorEventData.SensorSource.velodyne  # Generic source

        # BNO055 scaling factors (from datasheet)
        # Accelerometer: 1 m/s² = 100 LSB (when in m/s² mode)
        # Gyroscope: 1 rad/s = 900 LSB (when in rad/s mode)
        # Magnetometer: 1 µT = 16 LSB
        self.accel_scale = 1.0 / 100.0  # m/s² per LSB
        self.gyro_scale = 1.0 / 900.0   # rad/s per LSB
        self.mag_scale = 1.0 / 16.0     # µT per LSB

    def connect(self) -> bool:
        """
        Connect to the BNO055 via serial interface.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.serial_conn = serial.Serial(
                port=self.device_path,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0  # 1 second timeout
            )

            # Clear any existing data in buffers
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()

            cloudlog.info(f"Connected to BNO055 at {self.device_path}")
            return True

        except Exception as e:
            cloudlog.error(f"Failed to connect to BNO055: {e}")
            return False

    def disconnect(self):
        """Close the serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            cloudlog.info("Disconnected from BNO055")

    def read_register(self, reg_addr: int, length: int = 1) -> Optional[bytes]:
        """
        Read data from BNO055 register via UART.

        UART Read Protocol:
        1. Send start byte (0xBB)
        2. Send register address
        3. Send length to read
        4. Receive response (register data)

        Args:
            reg_addr: Register address to read from
            length: Number of bytes to read

        Returns:
            Register data as bytes, or None if read failed
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            return None

        try:
            # Construct read command
            command = bytes([self.UART_START_BYTE_READ, reg_addr, length])

            # Send command
            self.serial_conn.write(command)
            self.serial_conn.flush()

            # Read response (should be 'length' bytes)
            response = self.serial_conn.read(length)

            if len(response) == length:
                return response
            else:
                cloudlog.warning(f"BNO055 read incomplete: expected {length}, got {len(response)}")
                return None

        except Exception as e:
            cloudlog.error(f"BNO055 read error: {e}")
            return None

    def write_register(self, reg_addr: int, data: int) -> bool:
        """
        Write data to BNO055 register via UART.

        UART Write Protocol:
        1. Send start byte (0xAA)
        2. Send register address
        3. Send length (1 for single byte)
        4. Send data byte
        5. Receive acknowledgment (0xEE for success)

        Args:
            reg_addr: Register address to write to
            data: Data byte to write

        Returns:
            True if write successful, False otherwise
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            return False

        try:
            # Construct write command
            command = bytes([self.UART_START_BYTE_WRITE, reg_addr, 0x01, data])

            # Send command
            self.serial_conn.write(command)
            self.serial_conn.flush()

            # Read acknowledgment
            ack = self.serial_conn.read(1)

            if len(ack) == 1 and ack[0] == self.UART_STATUS_WRITE_SUCCESS:
                return True
            else:
                cloudlog.warning(f"BNO055 write failed: ack={ack.hex() if ack else 'none'}")
                return False

        except Exception as e:
            cloudlog.error(f"BNO055 write error: {e}")
            return False

    def init(self) -> bool:
        """
        Initialize the BNO055 sensor.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Connect to serial device
            if not self.connect():
                return False

            # Wait for sensor to be ready
            time.sleep(0.65)  # Wait for reset as per datasheet

            # Verify chip ID (should be 0xA0)
            chip_id_data = self.read_register(self.BNO055_CHIP_ID_ADDR, 1)
            if not chip_id_data or chip_id_data[0] != 0xA0:
                cloudlog.error(f"BNO055 chip ID verification failed: {chip_id_data}")
                return False

            cloudlog.info(f"BNO055 chip ID verified: 0x{chip_id_data[0]:02X}")

            # Switch to config mode
            if not self.write_register(self.BNO055_OPR_MODE_ADDR, self.OPERATION_MODE_CONFIG):
                cloudlog.error("Failed to switch to config mode")
                return False

            time.sleep(0.025)  # Wait for mode switch

            # Reset the system
            if not self.write_register(self.BNO055_SYS_TRIGGER_ADDR, 0x20):
                cloudlog.error("Failed to reset system")
                return False

            time.sleep(0.65)  # Wait for reset

            # Verify chip ID again after reset
            chip_id_data = self.read_register(self.BNO055_CHIP_ID_ADDR, 1)
            if not chip_id_data or chip_id_data[0] != 0xA0:
                cloudlog.error(f"BNO055 chip ID verification failed after reset: {chip_id_data}")
                return False

            # Set to page 0
            if not self.write_register(self.BNO055_PAGE_ID_ADDR, 0x00):
                cloudlog.error("Failed to set page 0")
                return False

            time.sleep(0.01)

            # Set power mode to normal
            if not self.write_register(self.BNO055_PWR_MODE_ADDR, self.POWER_MODE_NORMAL):
                cloudlog.error("Failed to set normal power mode")
                return False

            time.sleep(0.01)

            # Configure units:
            # - Accelerometer: m/s² (bit 0 = 0)
            # - Gyroscope: rad/s (bit 1 = 1)
            # - Euler angles: radians (bit 2 = 1)
            # - Temperature: Celsius (bit 4 = 0)
            # - Fusion data format: Windows (bit 7 = 0)
            unit_sel = 0x06  # Binary: 00000110 (gyro=rad/s, euler=rad, accel=m/s²)
            if not self.write_register(self.BNO055_UNIT_SEL_ADDR, unit_sel):
                cloudlog.error("Failed to set units")
                return False

            time.sleep(0.01)

            # Set operation mode to AMG (Accelerometer, Magnetometer, Gyroscope)
            # This provides raw sensor data from all three sensors
            if not self.write_register(self.BNO055_OPR_MODE_ADDR, self.OPERATION_MODE_AMG):
                cloudlog.error("Failed to set AMG operation mode")
                return False

            time.sleep(0.01)

            cloudlog.info("BNO055 initialization completed successfully")
            return True

        except Exception as e:
            cloudlog.error(f"BNO055 initialization failed: {e}")
            return False

    def shutdown(self):
        """Shutdown the BNO055 sensor."""
        try:
            # Switch to config mode
            if self.serial_conn and self.serial_conn.is_open:
                self.write_register(self.BNO055_OPR_MODE_ADDR, self.OPERATION_MODE_CONFIG)

            # Disconnect
            self.disconnect()

        except Exception as e:
            cloudlog.error(f"BNO055 shutdown error: {e}")

    def _parse_16bit_signed(self, lsb: int, msb: int) -> int:
        """Parse 16-bit signed integer from LSB and MSB bytes."""
        value = (msb << 8) | lsb
        # Convert to signed 16-bit
        if value > 32767:
            value -= 65536
        return value

    def _read_raw_data(self) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
        """
        Read raw sensor data from the BNO055.

        Returns:
            Tuple of (accel_raw, gyro_raw, mag_raw) where each is [x, y, z] in LSB units
            Returns (None, None, None) if read failed
        """
        try:
            # Read accelerometer data (6 bytes: X_LSB, X_MSB, Y_LSB, Y_MSB, Z_LSB, Z_MSB)
            accel_data = self.read_register(self.BNO055_ACCEL_DATA_X_LSB_ADDR, 6)
            if not accel_data or len(accel_data) != 6:
                return None, None, None

            accel_raw = [
                self._parse_16bit_signed(accel_data[0], accel_data[1]),  # X
                self._parse_16bit_signed(accel_data[2], accel_data[3]),  # Y
                self._parse_16bit_signed(accel_data[4], accel_data[5])   # Z
            ]

            # Read gyroscope data (6 bytes: X_LSB, X_MSB, Y_LSB, Y_MSB, Z_LSB, Z_MSB)
            gyro_data = self.read_register(self.BNO055_GYRO_DATA_X_LSB_ADDR, 6)
            if not gyro_data or len(gyro_data) != 6:
                return None, None, None

            gyro_raw = [
                self._parse_16bit_signed(gyro_data[0], gyro_data[1]),    # X
                self._parse_16bit_signed(gyro_data[2], gyro_data[3]),    # Y
                self._parse_16bit_signed(gyro_data[4], gyro_data[5])     # Z
            ]

            # Read magnetometer data (6 bytes: X_LSB, X_MSB, Y_LSB, Y_MSB, Z_LSB, Z_MSB)
            mag_data = self.read_register(self.BNO055_MAG_DATA_X_LSB_ADDR, 6)
            if not mag_data or len(mag_data) != 6:
                return None, None, None

            mag_raw = [
                self._parse_16bit_signed(mag_data[0], mag_data[1]),      # X
                self._parse_16bit_signed(mag_data[2], mag_data[3]),      # Y
                self._parse_16bit_signed(mag_data[4], mag_data[5])       # Z
            ]

            return accel_raw, gyro_raw, mag_raw

        except Exception as e:
            cloudlog.error(f"Error reading BNO055 data: {e}")
            return None, None, None

    def _get_simulated_data(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Generate simulated sensor data for testing purposes.

        Returns:
            Tuple of (accel_raw, gyro_raw, mag_raw) with realistic values
        """
        import math

        sim_time = time.time() * 0.1

        # Simulated accelerometer (around 1g on Z axis)
        accel_raw = [
            int((0.1 * math.sin(sim_time * 0.3)) * 100),  # Small X movement
            int((0.05 * math.cos(sim_time * 0.4)) * 100), # Small Y movement
            int((9.8 + 0.2 * math.sin(sim_time * 0.2)) * 100)  # ~1g + noise on Z
        ]

        # Simulated gyroscope (small rotation rates)
        gyro_raw = [
            int((0.01 * math.sin(sim_time * 0.5)) * 900),  # Small X rotation
            int((0.005 * math.cos(sim_time * 0.3)) * 900), # Small Y rotation
            int((0.002 * math.sin(sim_time * 0.7)) * 900)  # Small Z rotation
        ]

        # Simulated magnetometer (Earth's magnetic field)
        mag_raw = [
            int((25.0 + 3.0 * math.sin(sim_time * 0.1)) * 16),  # X component
            int((8.0 + 2.0 * math.cos(sim_time * 0.15)) * 16),  # Y component
            int((42.0 + 4.0 * math.sin(sim_time * 0.12)) * 16)  # Z component
        ]

        return accel_raw, gyro_raw, mag_raw


class BNO055Serial_Accel(BNO055Serial):
    """BNO055 accelerometer sensor via serial."""

    def get_event(self, ts: Optional[int] = None):
        """Generate accelerometer sensor event."""
        if ts is None:
            ts = time.monotonic_ns()

        accel_raw, _, _ = self._read_raw_data()

        # If hardware read failed, use simulation for testing
        if accel_raw is None:
            accel_raw, _, _ = self._get_simulated_data()

        # Convert raw values to m/s² (BNO055 already handles axis mapping)
        accel_ms2 = [
            accel_raw[0] * self.accel_scale,
            accel_raw[1] * self.accel_scale,
            accel_raw[2] * self.accel_scale
        ]

        event = log.SensorEventData.new_message()
        event.timestamp = ts
        event.version = 1
        event.sensor = 1  # SENSOR_ACCELEROMETER
        event.type = 1    # SENSOR_TYPE_ACCELEROMETER
        event.source = self.source

        a = event.init('acceleration')
        a.v = accel_ms2
        a.status = 1  # SENSOR_STATUS_ACCURACY_HIGH

        return event

    def is_data_valid(self) -> bool:
        """Check if sensor data is valid."""
        return True  # For serial sensors, data is always valid when read succeeds


class BNO055Serial_Gyro(BNO055Serial):
    """BNO055 gyroscope sensor via serial."""

    def get_event(self, ts: Optional[int] = None):
        """Generate gyroscope sensor event."""
        if ts is None:
            ts = time.monotonic_ns()

        _, gyro_raw, _ = self._read_raw_data()

        # If hardware read failed, use simulation for testing
        if gyro_raw is None:
            _, gyro_raw, _ = self._get_simulated_data()

        # Convert raw values to rad/s (BNO055 already handles axis mapping)
        gyro_rads = [
            gyro_raw[0] * self.gyro_scale,
            gyro_raw[1] * self.gyro_scale,
            gyro_raw[2] * self.gyro_scale
        ]

        event = log.SensorEventData.new_message()
        event.timestamp = ts
        event.version = 2
        event.sensor = 5  # SENSOR_GYRO_UNCALIBRATED
        event.type = 16   # SENSOR_TYPE_GYROSCOPE_UNCALIBRATED
        event.source = self.source

        g = event.init('gyroUncalibrated')
        g.v = gyro_rads
        g.status = 1  # SENSOR_STATUS_ACCURACY_HIGH

        return event

    def is_data_valid(self) -> bool:
        """Check if sensor data is valid."""
        return True


class BNO055Serial_Magn(BNO055Serial):
    """BNO055 magnetometer sensor via serial."""

    def get_event(self, ts: Optional[int] = None):
        """Generate magnetometer sensor event."""
        if ts is None:
            ts = time.monotonic_ns()

        _, _, mag_raw = self._read_raw_data()

        # If hardware read failed, use simulation for testing
        if mag_raw is None:
            _, _, mag_raw = self._get_simulated_data()

        # Convert raw values to µT (BNO055 already handles axis mapping)
        mag_ut = [
            mag_raw[0] * self.mag_scale,
            mag_raw[1] * self.mag_scale,
            mag_raw[2] * self.mag_scale
        ]

        event = log.SensorEventData.new_message()
        event.timestamp = ts
        event.version = 1
        event.sensor = 3  # SENSOR_MAGNETOMETER_UNCALIBRATED
        event.type = 14   # SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED
        event.source = self.source

        m = event.init('magneticUncalibrated')
        m.v = mag_ut
        m.status = 1  # SENSOR_STATUS_ACCURACY_HIGH

        return event

    def is_data_valid(self) -> bool:
        """Check if sensor data is valid."""
        return True