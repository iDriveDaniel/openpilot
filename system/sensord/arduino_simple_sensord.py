#!/usr/bin/env python3
"""
Simple Arduino BNO055 Sensor Daemon for Openpilot

A simplified version that avoids threading issues by reading Arduino data
synchronously in each polling loop.

Usage:
    python3 arduino_simple_sensord.py [--device /dev/ttyACM0] [--baudrate 115200]
"""

import os
import time
import signal
import sys
import argparse
import serial
import threading
from dataclasses import dataclass
from typing import Optional

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.common.realtime import config_realtime_process, Ratekeeper
from openpilot.common.swaglog import cloudlog

@dataclass
class SimpleBNO055Data:
    """Simple data structure for BNO055 sensor data."""
    # Linear acceleration (m/s²)
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0

    # Gyroscope (rad/s)
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0

    # Magnetometer (µT)
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0

    # Temperature (°C)
    temp: float = 25.0

    # Data validity
    valid: bool = False
    timestamp: int = 0

class SimpleArduinoBNO055:
    """Simple Arduino BNO055 sensor interface without threading."""

    def __init__(self, device_path: str, baudrate: int = 115200):
        self.device_path = device_path
        self.baudrate = baudrate
        self.serial_conn: Optional[serial.Serial] = None
        self.latest_data = SimpleBNO055Data()
        self.parse_errors = 0
        self.total_reads = 0

    def connect(self) -> bool:
        """Connect to Arduino."""
        try:
            self.serial_conn = serial.Serial(
                self.device_path,
                self.baudrate,
                timeout=0.1,  # Short timeout for non-blocking reads
                write_timeout=1.0
            )

            # Clear buffers
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()

            # Wait for Arduino initialization
            time.sleep(2.0)

            cloudlog.info(f"Connected to Arduino BNO055 at {self.device_path}")
            return True

        except Exception as e:
            cloudlog.error(f"Failed to connect to Arduino BNO055: {e}")
            return False

    def disconnect(self):
        """Disconnect from Arduino."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            cloudlog.info("Disconnected from Arduino BNO055")

    def read_data(self) -> bool:
        """Read and parse one line of data from Arduino. Returns True if new data was received."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return False

        try:
            # Try to read a line
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()

                if line:
                    self.total_reads += 1
                    if self.parse_data_line(line):
                        return True
                    else:
                        self.parse_errors += 1

            return False

        except Exception as e:
            cloudlog.warning(f"Error reading Arduino data: {e}")
            return False

    def parse_data_line(self, line: str) -> bool:
        """Parse a line of Arduino data. Returns True if successfully parsed."""
        try:
            # Expected format: heading:0.00,pitch:0.06,roll:0.06,gyroX:-0.00,...
            if 'gyroX' not in line or 'accX' not in line:
                return False

            data = SimpleBNO055Data()
            data.timestamp = time.monotonic_ns()

            # Parse key-value pairs
            pairs = line.split(',')
            for pair in pairs:
                if ':' in pair:
                    key, value_str = pair.split(':', 1)
                    try:
                        value = float(value_str)

                        # Map Arduino data to our format
                        if key == 'linX':  # Linear acceleration
                            data.accel_x = value
                        elif key == 'linY':
                            data.accel_y = value
                        elif key == 'linZ':
                            data.accel_z = value
                        elif key == 'gyroX':  # Gyroscope (convert deg/s to rad/s)
                            data.gyro_x = value * 0.017453293  # deg to rad
                        elif key == 'gyroY':
                            data.gyro_y = value * 0.017453293
                        elif key == 'gyroZ':
                            data.gyro_z = value * 0.017453293
                        elif key == 'magX':  # Magnetometer
                            data.mag_x = value
                        elif key == 'magY':
                            data.mag_y = value
                        elif key == 'magZ':
                            data.mag_z = value
                        elif key == 'temp':  # Temperature
                            data.temp = value

                    except ValueError:
                        continue

            data.valid = True
            self.latest_data = data
            return True

        except Exception as e:
            cloudlog.warning(f"Error parsing Arduino data: {e}")
            return False

def create_accelerometer_event(data: SimpleBNO055Data) -> object:
    """Create accelerometer sensor event."""
    from cereal import log

    event = log.SensorEventData.new_message()
    event.sensor = 1
    event.type = 1
    event.timestamp = data.timestamp
    event.source = log.SensorEventData.SensorSource.bno055

    event.init('acceleration')
    event.acceleration.v = [data.accel_x, data.accel_y, data.accel_z]
    event.acceleration.status = 3  # High accuracy

    return event

def create_gyroscope_event(data: SimpleBNO055Data) -> object:
    """Create gyroscope sensor event."""
    from cereal import log

    event = log.SensorEventData.new_message()
    event.sensor = 5
    event.type = 16
    event.timestamp = data.timestamp
    event.source = log.SensorEventData.SensorSource.bno055

    event.init('gyroUncalibrated')
    event.gyroUncalibrated.v = [data.gyro_x, data.gyro_y, data.gyro_z]
    event.gyroUncalibrated.status = 3  # High accuracy

    return event

def create_magnetometer_event(data: SimpleBNO055Data) -> object:
    """Create magnetometer sensor event."""
    from cereal import log

    event = log.SensorEventData.new_message()
    event.sensor = 3
    event.type = 14
    event.timestamp = data.timestamp
    event.source = log.SensorEventData.SensorSource.bno055

    event.init('magneticUncalibrated')
    event.magneticUncalibrated.v = [data.mag_x, data.mag_y, data.mag_z, 0.0, 0.0, 0.0]  # [field + bias]
    event.magneticUncalibrated.status = 3  # High accuracy

    return event

def polling_loop(arduino: SimpleArduinoBNO055, service: str, event_creator, exit_event: threading.Event):
    """Polling loop for a specific sensor service."""
    pm = messaging.PubMaster([service])
    rk = Ratekeeper(SERVICE_LIST[service].frequency, print_delay_threshold=None)

    cloudlog.info(f"Starting {service} polling loop at {SERVICE_LIST[service].frequency} Hz")

    while not exit_event.is_set():
        try:
            # Read fresh data from Arduino
            arduino.read_data()

            if arduino.latest_data.valid:
                # Create sensor event
                evt = event_creator(arduino.latest_data)

                # Create and send message
                msg = messaging.new_message(service, valid=True)
                setattr(msg, service, evt)
                pm.send(service, msg)

        except Exception as e:
            cloudlog.exception(f"Error in {service} polling loop: {e}")

        rk.keep_time()

    cloudlog.info(f"Stopped {service} polling loop")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    cloudlog.info(f"Received signal {signum}, shutting down...")
    global exit_event
    exit_event.set()

def main():
    """Main function."""
    global exit_event

    # Parse arguments
    parser = argparse.ArgumentParser(description='Simple Arduino BNO055 Sensor Daemon')
    parser.add_argument('--device', type=str, default='/dev/ttyACM0', help='Arduino device path')
    parser.add_argument('--baudrate', type=int, default=115200, help='Serial baud rate')
    args = parser.parse_args()

    # Configure real-time process
    config_realtime_process([1], 1)

    # Set up signal handling
    exit_event = threading.Event()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    cloudlog.info(f"Starting simple Arduino BNO055 sensor daemon")
    cloudlog.info(f"Device: {args.device}, Baud rate: {args.baudrate}")

    # Check if device exists
    if not os.path.exists(args.device):
        cloudlog.error(f"Arduino device {args.device} not found!")
        return

    # Connect to Arduino
    arduino = SimpleArduinoBNO055(args.device, args.baudrate)
    if not arduino.connect():
        cloudlog.error("Failed to connect to Arduino BNO055!")
        return

    cloudlog.info("Arduino BNO055 connected successfully!")

    try:
        # Wait for initial data
        cloudlog.info("Waiting for sensor data...")
        start_time = time.time()
        while time.time() - start_time < 10.0:
            if arduino.read_data() and arduino.latest_data.valid:
                cloudlog.info("Sensor data available, starting polling loops")
                break
            time.sleep(0.1)
        else:
            cloudlog.error("Timeout waiting for sensor data")
            return

        # Create sensor event creators
        creators = [
            ("accelerometer", create_accelerometer_event),
            ("gyroscope", create_gyroscope_event),
            ("magnetometer", create_magnetometer_event),
        ]

        # Start polling threads
        threads = []
        for service, creator in creators:
            thread = threading.Thread(
                target=polling_loop,
                args=(arduino, service, creator, exit_event),
                daemon=True,
                name=f"{service}_thread"
            )
            threads.append(thread)
            thread.start()
            cloudlog.info(f"Started {service} polling thread")

        cloudlog.info("Simple Arduino BNO055 sensor daemon started successfully!")

        # Main loop
        while not exit_event.is_set():
            time.sleep(1.0)

            # Check if any threads died
            for thread in threads:
                if not thread.is_alive():
                    cloudlog.error("A sensor thread has died, shutting down")
                    exit_event.set()
                    break

    except Exception as e:
        cloudlog.exception(f"Error in main loop: {e}")
        exit_event.set()

    finally:
        # Cleanup
        cloudlog.info("Shutting down simple Arduino BNO055 sensor daemon...")
        exit_event.set()

        # Wait for threads
        for thread in threads:
            thread.join(timeout=2.0)

        # Disconnect Arduino
        arduino.disconnect()

        cloudlog.info("Simple Arduino BNO055 sensor daemon shutdown complete")

if __name__ == "__main__":
    main()