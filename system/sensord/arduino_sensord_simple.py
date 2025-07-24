#!/usr/bin/env python3
"""
Arduino BNO055 Sensor Daemon - Simplified Version

Uses a single sensor instance to generate all three sensor types.
"""

import os
import time
import threading

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.common.realtime import config_realtime_process, Ratekeeper
from openpilot.common.swaglog import cloudlog

from openpilot.system.sensord.sensors.arduino_bno055 import ArduinoBNO055_Accel

def debug_print(msg):
    """Print to both console and cloudlog"""
    print(f"[DEBUG] {msg}")
    cloudlog.info(msg)

def create_gyro_event(sensor, ts=None):
    """Create gyroscope event from sensor data"""
    if ts is None:
        ts = time.monotonic_ns()

    data = sensor.get_latest_data()

    from cereal import log
    event = log.SensorEventData.new_message()
    event.timestamp = ts
    event.version = 2
    event.sensor = 5  # SENSOR_GYRO_UNCALIBRATED
    event.type = 16   # SENSOR_TYPE_GYROSCOPE_UNCALIBRATED
    event.source = sensor.source

    g = event.init('gyroUncalibrated')
    g.v = [data.gyro_x, data.gyro_y, data.gyro_z]
    g.status = 1  # SENSOR_STATUS_ACCURACY_HIGH

    return event

def create_mag_event(sensor, ts=None):
    """Create magnetometer event from sensor data"""
    if ts is None:
        ts = time.monotonic_ns()

    data = sensor.get_latest_data()

    from cereal import log
    event = log.SensorEventData.new_message()
    event.timestamp = ts
    event.version = 1
    event.sensor = 3  # SENSOR_MAGNETOMETER_UNCALIBRATED
    event.type = 14   # SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED
    event.source = sensor.source

    m = event.init('magneticUncalibrated')
    m.v = [data.mag_x, data.mag_y, data.mag_z]
    m.status = 1  # SENSOR_STATUS_ACCURACY_HIGH

    return event

def polling_loop(sensor, service: str, event_creator, event: threading.Event) -> None:
    """Polling loop for Arduino BNO055 sensors using single sensor instance"""
    pm = messaging.PubMaster([service])
    rk = Ratekeeper(SERVICE_LIST[service].frequency, print_delay_threshold=None)

    debug_print(f"Starting {service} polling loop at {SERVICE_LIST[service].frequency} Hz")

    message_count = 0
    error_count = 0

    while not event.is_set():
        try:
            # Check if data is valid
            if not sensor.is_data_valid():
                rk.keep_time()
                continue

            # Create event using the appropriate creator function
            if event_creator:
                evt = event_creator(sensor)
            else:
                evt = sensor.get_event()  # For accelerometer

            if evt is None:
                rk.keep_time()
                continue

            # Create and send message
            msg = messaging.new_message(service, valid=True)
            setattr(msg, service, evt)
            # print(service, msg)
            pm.send(service, msg)

            message_count += 1
            if message_count % 100 == 0:  # Print every 100 messages
                debug_print(f"{service}: Sent {message_count} messages")

                # Show actual data being sent
                if service == 'accelerometer':
                    accel = evt.acceleration.v
                    debug_print(f"   Accel data: [{accel[0]:.3f}, {accel[1]:.3f}, {accel[2]:.3f}] m/s²")
                    # Show orientation data from the sensor
                    data = sensor.get_latest_data()
                    debug_print(f"   Orientation: heading={data.heading:.1f}°, pitch={data.pitch:.1f}°, roll={data.roll:.1f}°")
                elif service == 'gyroscope':
                    gyro = evt.gyroUncalibrated.v
                    gyro_deg = [v * 180 / 3.14159 for v in gyro]
                    debug_print(f"   Gyro data: [{gyro_deg[0]:.2f}, {gyro_deg[1]:.2f}, {gyro_deg[2]:.2f}] deg/s")
                elif service == 'magnetometer':
                    mag_v = evt.magneticUncalibrated.v
                    mag = [mag_v[0], mag_v[1], mag_v[2]]
                    debug_print(f"   Mag data: [{mag[0]:.1f}, {mag[1]:.1f}, {mag[2]:.1f}] µT")

        except Exception as e:
            error_count += 1
            debug_print(f"Error in {service} polling loop (#{error_count}): {e}")
            cloudlog.exception(f"Error in {service} polling loop")
        rk.keep_time()

    debug_print(f"{service} polling loop ended. Sent {message_count} messages, {error_count} errors")

def main() -> None:
    """Main function using single sensor instance"""

    debug_print("=== Arduino BNO055 Sensor Daemon (Simplified) Starting ===")

    # Configure real-time process
    config_realtime_process([1], 1)
    debug_print("Real-time process configured")

    # Arduino device configuration
    device_path = "/dev/ttyACM0"
    baudrate = 115200

    # Check if device exists
    if not os.path.exists(device_path):
        debug_print(f"Arduino device {device_path} not found!")
        for alt_device in ["/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1"]:
            if os.path.exists(alt_device):
                device_path = alt_device
                debug_print(f"Using alternative device: {device_path}")
                break
        else:
            debug_print("No Arduino device found!")
            return

    debug_print(f"Using Arduino BNO055 at {device_path} (baudrate: {baudrate})")

    # Create single sensor instance
    sensor = ArduinoBNO055_Accel(device_path, baudrate)

    # Initialize the Arduino connection
    debug_print("Initializing Arduino BNO055 connection...")
    if not sensor.init():
        debug_print("Failed to initialize Arduino BNO055!")
        return

    debug_print("Arduino BNO055 initialized successfully!")

    # Wait for initial data
    debug_print("Waiting for sensor data...")
    start_time = time.time()
    while time.time() - start_time < 10.0:
        if sensor.is_data_valid():
            debug_print("Sensor data available, starting polling loops")
            break
        time.sleep(0.1)
    else:
        debug_print("Timeout waiting for sensor data")
        sensor.shutdown()
        return

    # Sensor configuration - all use the same sensor instance but different event creators
    sensors_cfg = [
        (sensor, "accelerometer", None),  # Use sensor's built-in get_event
        (sensor, "gyroscope", create_gyro_event),  # Use custom event creator
        (sensor, "magnetometer", create_mag_event),  # Use custom event creator
    ]

    # Create and start polling threads
    event = threading.Event()
    threads = []

    for sensor_inst, service, event_creator in sensors_cfg:
        thread = threading.Thread(
            target=polling_loop,
            args=(sensor_inst, service, event_creator, event),
            daemon=True
        )
        threads.append(thread)
        thread.start()
        debug_print(f"Started {service} polling thread")

    debug_print("Arduino BNO055 sensor daemon started successfully!")

    try:
        # Main loop
        while True:
            # Check if any threads have died
            for thread in threads:
                if not thread.is_alive():
                    debug_print("A sensor thread has died, shutting down")
                    event.set()
                    break

            if event.is_set():
                break

            time.sleep(1.0)

    except KeyboardInterrupt:
        debug_print("Shutting down due to keyboard interrupt")
        event.set()

    # Cleanup
    debug_print("Shutting down Arduino BNO055 sensor daemon...")
    event.set()

    # Wait for threads to finish
    for thread in threads:
        thread.join(timeout=2.0)

    # Shutdown the Arduino connection
    sensor.shutdown()
    debug_print("Arduino BNO055 sensor daemon shutdown complete")

if __name__ == "__main__":
    main()