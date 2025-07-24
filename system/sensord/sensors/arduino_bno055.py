#!/usr/bin/env python3
"""
Arduino BNO055 Sensor Implementation for Openpilot

This module provides sensor classes that communicate with an Arduino
running BNO055 firmware that outputs data in CSV/key-value format.

The Arduino outputs data in this format:
heading:0.00,pitch:0.06,roll:0.06,gyroX:-0.00,gyroY:-0.00,gyroZ:0.00,
linX:0.00,linY:0.02,linZ:-0.32,magX:32.38,magY:-41.56,magZ:2.38,
accX:0.00,accY:0.00,accZ:9.48,gravX:0.01,gravY:-0.01,gravZ:9.80,
temp:32,cal:0,0,0,0
"""

import time
import serial
import threading
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from cereal import log
from openpilot.common.swaglog import cloudlog

@dataclass
class BNO055Data:
    """Data structure for parsed BNO055 sensor data."""
    heading: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    lin_x: float = 0.0
    lin_y: float = 0.0
    lin_z: float = 0.0
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0
    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 0.0
    grav_x: float = 0.0
    grav_y: float = 0.0
    grav_z: float = 0.0
    temp: float = 0.0
    cal_sys: int = 0
    cal_gyro: int = 0
    cal_accel: int = 0
    cal_mag: int = 0
    timestamp: float = 0.0
    valid: bool = False

class ArduinoBNO055:
    """
    Arduino BNO055 sensor communication and data parsing.

    This class handles serial communication with an Arduino that has
    a BNO055 sensor connected and outputs data in CSV format.
    """

    def __init__(self, device_path: str = "/dev/ttyACM0", baudrate: int = 115200):
        """
        Initialize Arduino BNO055 communication.

        Args:
            device_path: Serial device path (e.g., "/dev/ttyACM0")
            baudrate: UART baud rate (default: 115200)
        """
        self.device_path = device_path
        self.baudrate = baudrate
        self.serial_conn = None
        self.source = log.SensorEventData.SensorSource.velodyne  # Generic source

        # Data storage
        self.latest_data = BNO055Data()
        self.data_lock = threading.Lock()
        self.read_thread = None
        self.stop_reading = threading.Event()

        # Statistics
        self.total_reads = 0
        self.parse_errors = 0
        self.last_read_time = 0.0

    def connect(self) -> bool:
        """
        Connect to the Arduino via serial interface.

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

            # Wait for Arduino to initialize after serial connection
            time.sleep(2.0)  # Arduino resets when serial connection opens

            cloudlog.info(f"Connected to Arduino BNO055 at {self.device_path}")
            return True

        except Exception as e:
            cloudlog.error(f"Failed to connect to Arduino BNO055: {e}")
            return False

    def disconnect(self):
        """Close the serial connection."""
        self.stop_reading.set()

        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2.0)

        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            cloudlog.info("Disconnected from Arduino BNO055")

    def parse_data_line(self, line: str) -> Optional[BNO055Data]:
        """
        Parse a line of data from the Arduino.

        Expected format:
        heading:0.00,pitch:0.06,roll:0.06,gyroX:-0.00,gyroY:-0.00,gyroZ:0.00,
        linX:0.00,linY:0.02,linZ:-0.32,magX:32.38,magY:-41.56,magZ:2.38,
        accX:0.00,accY:0.00,accZ:9.48,gravX:0.01,gravY:-0.01,gravZ:9.80,
        temp:32,cal:0,0,0,0

        Args:
            line: Data line from Arduino

        Returns:
            BNO055Data object if parsing successful, None otherwise
        """
        try:
            # Skip non-data lines
            if not ':' in line or 'heading:' not in line:
                return None

            # Parse key-value pairs
            data = BNO055Data()
            data.timestamp = time.time()

            # Split by comma and parse each key:value pair
            pairs = line.strip().split(',')

            for pair in pairs:
                if ':' not in pair:
                    continue

                key, value_str = pair.split(':', 1)

                try:
                    value = float(value_str)
                except ValueError:
                    continue

                # Map keys to data fields
                if key == 'heading':
                    data.heading = value
                elif key == 'pitch':
                    data.pitch = value
                elif key == 'roll':
                    data.roll = value
                elif key == 'gyroX':
                    data.gyro_x = value
                elif key == 'gyroY':
                    data.gyro_y = value
                elif key == 'gyroZ':
                    data.gyro_z = value
                elif key == 'linX':
                    data.lin_x = value
                elif key == 'linY':
                    data.lin_y = value
                elif key == 'linZ':
                    data.lin_z = value
                elif key == 'magX':
                    data.mag_x = value
                elif key == 'magY':
                    data.mag_y = value
                elif key == 'magZ':
                    data.mag_z = value
                elif key == 'accX':
                    data.acc_x = value
                elif key == 'accY':
                    data.acc_y = value
                elif key == 'accZ':
                    data.acc_z = value
                elif key == 'gravX':
                    data.grav_x = value
                elif key == 'gravY':
                    data.grav_y = value
                elif key == 'gravZ':
                    data.grav_z = value
                elif key == 'temp':
                    data.temp = value
                elif key == 'cal':
                    # Calibration is the first value after 'cal:'
                    data.cal_sys = int(value)

            # Parse calibration values (cal:0,0,0,0 format)
            cal_start = line.find('cal:')
            if cal_start >= 0:
                cal_part = line[cal_start + 4:]  # Skip 'cal:'
                cal_values = cal_part.split(',')
                if len(cal_values) >= 4:
                    try:
                        data.cal_sys = int(cal_values[0])
                        data.cal_gyro = int(cal_values[1])
                        data.cal_accel = int(cal_values[2])
                        data.cal_mag = int(cal_values[3])
                    except (ValueError, IndexError):
                        pass  # Keep default values

            data.valid = True
            return data

        except Exception as e:
            cloudlog.warning(f"Error parsing Arduino data line: {e}")
            return None

    def _read_thread_func(self):
        """Background thread function to continuously read data from Arduino."""
        cloudlog.info("Started Arduino BNO055 data reading thread")

        while not self.stop_reading.is_set():
            try:
                if not self.serial_conn or not self.serial_conn.is_open:
                    time.sleep(0.1)
                    continue

                # Read a line of data
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()

                if line:
                    self.total_reads += 1
                    parsed_data = self.parse_data_line(line)

                    if parsed_data:
                        with self.data_lock:
                            self.latest_data = parsed_data
                            self.last_read_time = time.time()
                    else:
                        self.parse_errors += 1

            except Exception as e:
                cloudlog.warning(f"Error in Arduino BNO055 read thread: {e}")
                time.sleep(0.1)

        cloudlog.info("Arduino BNO055 data reading thread stopped")

    def init(self) -> bool:
        """
        Initialize the Arduino BNO055 sensor.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Connect to Arduino
            if not self.connect():
                return False

            # Start background reading thread
            self.stop_reading.clear()
            self.read_thread = threading.Thread(target=self._read_thread_func, daemon=True)
            self.read_thread.start()

            # Wait for first data
            start_time = time.time()
            while time.time() - start_time < 5.0:  # Wait up to 5 seconds
                with self.data_lock:
                    if self.latest_data.valid:
                        cloudlog.info("Arduino BNO055 initialization completed successfully")
                        return True
                time.sleep(0.1)

            cloudlog.warning("Arduino BNO055 initialization timed out waiting for data")
            return False

        except Exception as e:
            cloudlog.error(f"Arduino BNO055 initialization failed: {e}")
            return False

    def shutdown(self):
        """Shutdown the Arduino BNO055 sensor."""
        try:
            cloudlog.info("Shutting down Arduino BNO055...")
            self.disconnect()
        except Exception as e:
            cloudlog.error(f"Arduino BNO055 shutdown error: {e}")

    def get_latest_data(self) -> BNO055Data:
        """
        Get the latest sensor data.

        Returns:
            Latest BNO055Data object
        """
        with self.data_lock:
            return self.latest_data

    def is_data_valid(self) -> bool:
        """Check if sensor data is valid and recent."""
        with self.data_lock:
            # Data is valid if we have recent readings (within last 1 second)
            return (self.latest_data.valid and
                   (time.time() - self.last_read_time) < 1.0)

    def get_statistics(self) -> Dict:
        """Get reading statistics."""
        return {
            'total_reads': self.total_reads,
            'parse_errors': self.parse_errors,
            'success_rate': (self.total_reads - self.parse_errors) / max(1, self.total_reads),
            'last_read_age': time.time() - self.last_read_time if self.last_read_time > 0 else float('inf')
        }


class ArduinoBNO055_Accel(ArduinoBNO055):
    """Arduino BNO055 accelerometer sensor."""

    def get_event(self, ts: Optional[int] = None):
        """Generate accelerometer sensor event."""
        if ts is None:
            ts = time.monotonic_ns()

        data = self.get_latest_data()

        # Use linear acceleration (gravity removed) for better vehicle dynamics
        # Fall back to raw acceleration if linear not available
        if abs(data.lin_x) + abs(data.lin_y) + abs(data.lin_z) > 0.01:
            accel_ms2 = [data.lin_x, data.lin_y, data.lin_z]
        else:
            accel_ms2 = [data.acc_x, data.acc_y, data.acc_z]

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


class ArduinoBNO055_Gyro(ArduinoBNO055):
    """Arduino BNO055 gyroscope sensor."""

    def get_event(self, ts: Optional[int] = None):
        """Generate gyroscope sensor event."""
        if ts is None:
            ts = time.monotonic_ns()

        data = self.get_latest_data()

        # Gyroscope data (assuming rad/s, convert if needed)
        gyro_rads = [data.gyro_x, data.gyro_y, data.gyro_z]

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


class ArduinoBNO055_Magn(ArduinoBNO055):
    """Arduino BNO055 magnetometer sensor."""

    def get_event(self, ts: Optional[int] = None):
        """Generate magnetometer sensor event."""
        if ts is None:
            ts = time.monotonic_ns()

        data = self.get_latest_data()

        # Magnetometer data in µT
        mag_ut = [data.mag_x, data.mag_y, data.mag_z]

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