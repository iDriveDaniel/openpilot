#!/usr/bin/env python3
"""
Arduino BNO055 Sensor Daemon for Openpilot

This is a drop-in replacement for the regular sensord.py that uses
Arduino BNO055 sensors instead of LSM6DS3/MMC5603NJ sensors.

It follows the exact same pattern as the original sensord.py but
uses our Arduino BNO055 implementation.
"""

import os
import time
import threading

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.common.realtime import config_realtime_process, Ratekeeper
from openpilot.common.swaglog import cloudlog

from openpilot.system.sensord.sensors.arduino_bno055 import (
    ArduinoBNO055_Accel, ArduinoBNO055_Gyro, ArduinoBNO055_Magn
)

def polling_loop(sensor, service: str, event: threading.Event) -> None:
    """
    Polling loop for Arduino BNO055 sensors.
    This follows the exact same pattern as the original sensord.py
    """
    pm = messaging.PubMaster([service])
    rk = Ratekeeper(SERVICE_LIST[service].frequency, print_delay_threshold=None)

    cloudlog.info(f"Starting {service} polling loop at {SERVICE_LIST[service].frequency} Hz")

    while not event.is_set():
        try:
            evt = sensor.get_event()
            if not sensor.is_data_valid():
                continue
            msg = messaging.new_message(service, valid=True)
            setattr(msg, service, evt)
            pm.send(service, msg)
        except Exception:
            cloudlog.exception(f"Error in {service} polling loop")
        rk.keep_time()

def main() -> None:
    """Main function - follows the exact same pattern as original sensord.py"""

    # Configure real-time process (same as original)
    config_realtime_process([1], 1)

    # Arduino device configuration
    device_path = "/dev/ttyACM0"  # Default Arduino device
    baudrate = 115200

    # Check if device exists
    if not os.path.exists(device_path):
        cloudlog.error(f"Arduino device {device_path} not found!")
        # Try alternative devices
        for alt_device in ["/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1"]:
            if os.path.exists(alt_device):
                device_path = alt_device
                cloudlog.info(f"Using alternative device: {device_path}")
                break
        else:
            cloudlog.error("No Arduino device found!")
            return

    cloudlog.info(f"Using Arduino BNO055 at {device_path} (baudrate: {baudrate})")

    # Create a shared sensor connection (all sensors use the same Arduino)
    base_sensor = ArduinoBNO055_Accel(device_path, baudrate)

    # Initialize the Arduino connection
    cloudlog.info("Initializing Arduino BNO055 connection...")
    if not base_sensor.init():
        cloudlog.error("Failed to initialize Arduino BNO055!")
        return

    cloudlog.info("Arduino BNO055 initialized successfully!")

    # Create sensor instances that share the same connection
    accel_sensor = base_sensor
    gyro_sensor = ArduinoBNO055_Gyro(device_path, baudrate)
    mag_sensor = ArduinoBNO055_Magn(device_path, baudrate)

    # Share the connection and data with other sensors (CRITICAL: Don't call init() on these!)
    gyro_sensor.serial_conn = base_sensor.serial_conn
    gyro_sensor.latest_data = base_sensor.latest_data
    gyro_sensor.data_lock = base_sensor.data_lock
    gyro_sensor.last_read_time = base_sensor.last_read_time
    gyro_sensor.stop_reading = base_sensor.stop_reading
    gyro_sensor.read_thread = None  # Ensure no separate thread

    mag_sensor.serial_conn = base_sensor.serial_conn
    mag_sensor.latest_data = base_sensor.latest_data
    mag_sensor.data_lock = base_sensor.data_lock
    mag_sensor.last_read_time = base_sensor.last_read_time
    mag_sensor.stop_reading = base_sensor.stop_reading
    mag_sensor.read_thread = None  # Ensure no separate thread

    # Sensor configuration (same format as original sensord.py)
    # Note: All our sensors use polling (interrupt=False) since they share one serial connection
    sensors_cfg = [
        (accel_sensor, "accelerometer", False),
        (gyro_sensor, "gyroscope", False),
        (mag_sensor, "magnetometer", False),
    ]

    # Wait for initial data
    cloudlog.info("Waiting for sensor data...")
    start_time = time.time()
    while time.time() - start_time < 10.0:
        if base_sensor.is_data_valid():
            cloudlog.info("Sensor data available, starting polling loops")
            break
        time.sleep(0.1)
    else:
        cloudlog.error("Timeout waiting for sensor data")
        base_sensor.shutdown()
        return

    # Create and start polling threads (same pattern as original)
    event = threading.Event()
    threads = []

    for sensor, service, interrupt in sensors_cfg:
        if not interrupt:  # All our sensors use polling
            thread = threading.Thread(target=polling_loop, args=(sensor, service, event), daemon=True)
            threads.append(thread)
            thread.start()
            cloudlog.info(f"Started {service} polling thread")

    cloudlog.info("Arduino BNO055 sensor daemon started successfully!")

    try:
        # Main loop - just wait for threads
        while True:
            # Check if any threads have died
            for thread in threads:
                if not thread.is_alive():
                    cloudlog.error("A sensor thread has died, shutting down")
                    event.set()
                    break

            if event.is_set():
                break

            time.sleep(1.0)

    except KeyboardInterrupt:
        cloudlog.info("Shutting down due to keyboard interrupt")
        event.set()

    # Cleanup
    cloudlog.info("Shutting down Arduino BNO055 sensor daemon...")
    event.set()

    # Wait for threads to finish
    for thread in threads:
        thread.join(timeout=2.0)

    # Shutdown the Arduino connection
    base_sensor.shutdown()
    cloudlog.info("Arduino BNO055 sensor daemon shutdown complete")

if __name__ == "__main__":
    main()