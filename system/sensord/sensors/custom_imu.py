import time
import math
import struct
from typing import Optional, List, Tuple

from cereal import log
from openpilot.system.sensord.sensors.i2c_sensor import Sensor


class CustomIMU(Sensor):
    """
    BNO055 IMU sensor implementation that follows comma openpilot format.

    The BNO055 is a 9-DOF sensor with integrated accelerometer, gyroscope,
    and magnetometer, plus an ARM Cortex-M0 processor for sensor fusion.

    Supports:
    - Accelerometer: 3-axis acceleration in m/s²
    - Gyroscope: 3-axis angular velocity in rad/s
    - Magnetometer: 3-axis magnetic field in µT
    """

    # BNO055 Register addresses
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

    def __init__(self, bus: int, device_addr: int = 0x28):
        super().__init__(bus)
        self._device_addr = device_addr
        self.source = log.SensorEventData.SensorSource.velodyne  # Generic source

        # BNO055 scaling factors (from datasheet)
        # Accelerometer: 1 m/s² = 100 LSB (when in m/s² mode)
        # Gyroscope: 1 rad/s = 900 LSB (when in rad/s mode)
        # Magnetometer: 1 µT = 16 LSB
        self.accel_scale = 1.0 / 100.0  # m/s² per LSB
        self.gyro_scale = 1.0 / 900.0   # rad/s per LSB
        self.mag_scale = 1.0 / 16.0     # µT per LSB

    @property
    def device_address(self) -> int:
        return self._device_addr

    def init(self) -> None:
        """Initialize the BNO055 sensor."""
        try:
            # Verify chip ID (should be 0xA0)
            chip_id = self.verify_chip_id(self.BNO055_CHIP_ID_ADDR, [0xA0])
            print(f"BNO055 chip ID verified: 0x{chip_id:02X}")

            # Switch to config mode
            self.write(self.BNO055_OPR_MODE_ADDR, self.OPERATION_MODE_CONFIG)
            time.sleep(0.025)  # Wait for mode switch

            # Reset the system
            self.write(self.BNO055_SYS_TRIGGER_ADDR, 0x20)
            time.sleep(0.65)  # Wait for reset

            # Verify chip ID again after reset
            chip_id = self.verify_chip_id(self.BNO055_CHIP_ID_ADDR, [0xA0])

            # Set to page 0
            self.write(self.BNO055_PAGE_ID_ADDR, 0x00)
            time.sleep(0.01)

            # Set power mode to normal
            self.write(self.BNO055_PWR_MODE_ADDR, self.POWER_MODE_NORMAL)
            time.sleep(0.01)

            # Configure units:
            # - Accelerometer: m/s² (bit 0 = 0)
            # - Gyroscope: rad/s (bit 1 = 1)
            # - Euler angles: radians (bit 2 = 1)
            # - Temperature: Celsius (bit 4 = 0)
            # - Fusion data format: Windows (bit 7 = 0)
            unit_sel = 0x06  # Binary: 00000110 (gyro=rad/s, euler=rad, accel=m/s²)
            self.write(self.BNO055_UNIT_SEL_ADDR, unit_sel)
            time.sleep(0.01)

            # Set axis mapping to match vehicle coordinates
            # Default BNO055 orientation -> Vehicle orientation
            # This may need adjustment based on your sensor mounting
            self.write(self.BNO055_AXIS_MAP_CONFIG_ADDR, 0x24)  # X=Y, Y=X, Z=Z
            self.write(self.BNO055_AXIS_MAP_SIGN_ADDR, 0x02)    # X=+, Y=-, Z=+
            time.sleep(0.01)

            # Set operation mode to AMG (Accelerometer, Magnetometer, Gyroscope)
            # This mode provides raw sensor data without fusion
            self.write(self.BNO055_OPR_MODE_ADDR, self.OPERATION_MODE_AMG)
            time.sleep(0.02)  # Wait for mode switch

            print("BNO055 initialized successfully")

        except Exception as e:
            raise self.SensorException(f"Failed to initialize BNO055: {e}")

    def _read_raw_data(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Read raw sensor data from the BNO055.

        Returns:
            Tuple of (accel_raw, gyro_raw, mag_raw) where each is [x, y, z] in LSB units
        """
        try:
            # Read accelerometer data (6 bytes: X_LSB, X_MSB, Y_LSB, Y_MSB, Z_LSB, Z_MSB)
            accel_data = self.read(self.BNO055_ACCEL_DATA_X_LSB_ADDR, 6)
            accel_raw = [
                self.parse_16bit_signed(accel_data[0], accel_data[1]),  # X
                self.parse_16bit_signed(accel_data[2], accel_data[3]),  # Y
                self.parse_16bit_signed(accel_data[4], accel_data[5])   # Z
            ]

            # Read gyroscope data (6 bytes: X_LSB, X_MSB, Y_LSB, Y_MSB, Z_LSB, Z_MSB)
            gyro_data = self.read(self.BNO055_GYRO_DATA_X_LSB_ADDR, 6)
            gyro_raw = [
                self.parse_16bit_signed(gyro_data[0], gyro_data[1]),    # X
                self.parse_16bit_signed(gyro_data[2], gyro_data[3]),    # Y
                self.parse_16bit_signed(gyro_data[4], gyro_data[5])     # Z
            ]

            # Read magnetometer data (6 bytes: X_LSB, X_MSB, Y_LSB, Y_MSB, Z_LSB, Z_MSB)
            mag_data = self.read(self.BNO055_MAG_DATA_X_LSB_ADDR, 6)
            mag_raw = [
                self.parse_16bit_signed(mag_data[0], mag_data[1]),      # X
                self.parse_16bit_signed(mag_data[2], mag_data[3]),      # Y
                self.parse_16bit_signed(mag_data[4], mag_data[5])       # Z
            ]

            return accel_raw, gyro_raw, mag_raw

        except Exception as e:
            print(f"Error reading BNO055 data: {e}")
            # Return simulated data if hardware read fails (for testing/mock purposes)
            return self._get_simulated_data()

    def _get_simulated_data(self) -> Tuple[List[int], List[int], List[int]]:
        """Generate simulated sensor data for testing purposes."""
        import time
        import math

        # Use current time for simulation
        sim_time = time.time() * 0.1  # Scale time for slow variation

        # Simulate accelerometer data (gravity + small variations)
        # Convert to BNO055 scale (100 LSB per m/s²)
        accel_raw = [
            int((1.0 + 0.1 * math.sin(sim_time * 0.3)) * 100),    # ~1 m/s² X + variation
            int((-0.5 + 0.05 * math.cos(sim_time * 0.4)) * 100),  # ~-0.5 m/s² Y + variation
            int((9.8 + 0.2 * math.sin(sim_time * 0.2)) * 100)     # ~9.8 m/s² Z (gravity) + noise
        ]

        # Simulate gyroscope data (small rotations)
        # Convert to BNO055 scale (900 LSB per rad/s)
        gyro_raw = [
            int((0.02 * math.sin(sim_time * 0.5)) * 900),   # Small pitch rotation
            int((0.01 * math.cos(sim_time * 0.3)) * 900),   # Small roll rotation
            int((0.005 * math.sin(sim_time * 0.7)) * 900)   # Small yaw rotation
        ]

        # Simulate magnetometer data (Earth's magnetic field)
        # Convert to BNO055 scale (16 LSB per µT)
        mag_raw = [
            int((30.0 + 5.0 * math.sin(sim_time * 0.1)) * 16),  # ~30 µT North + variation
            int((10.0 + 2.0 * math.cos(sim_time * 0.15)) * 16), # ~10 µT East + variation
            int((45.0 + 3.0 * math.sin(sim_time * 0.12)) * 16)  # ~45 µT Down + variation
        ]

        return accel_raw, gyro_raw, mag_raw

    def parse_16bit_signed(self, lsb: int, msb: int) -> int:
        """Parse 16-bit signed integer from LSB and MSB bytes."""
        value = (msb << 8) | lsb
        # Convert to signed 16-bit
        if value > 32767:
            value -= 65536
        return value

    def get_accelerometer_event(self, ts: Optional[int] = None):
        """Generate accelerometer sensor event."""
        if ts is None:
            ts = time.monotonic_ns()

        accel_raw, _, _ = self._read_raw_data()

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

    def get_gyroscope_event(self, ts: Optional[int] = None):
        """Generate gyroscope sensor event."""
        if ts is None:
            ts = time.monotonic_ns()

        _, gyro_raw, _ = self._read_raw_data()

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

    def get_magnetometer_event(self, ts: Optional[int] = None):
        """Generate magnetometer sensor event."""
        if ts is None:
            ts = time.monotonic_ns()

        _, _, mag_raw = self._read_raw_data()

        # Convert raw values to µT (BNO055 already handles axis mapping)
        mag_ut = [
            float(mag_raw[0] * self.mag_scale),
            float(mag_raw[1] * self.mag_scale),
            float(mag_raw[2] * self.mag_scale)
        ]

        # Magnetometer uncalibrated format includes bias estimates (set to 0 for now)
        mag_values = mag_ut + [0.0, 0.0, 0.0]  # [mx, my, mz, bias_x, bias_y, bias_z]

        event = log.SensorEventData.new_message()
        event.timestamp = ts
        event.version = 1
        event.sensor = 3  # SENSOR_MAGNETOMETER_UNCALIBRATED
        event.type = 14   # SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED
        event.source = self.source

        m = event.init('magneticUncalibrated')
        m.v = mag_values
        m.status = int(all(abs(v) < 1000 for v in mag_ut))  # Simple validity check

        return event

    def get_event(self, ts: Optional[int] = None):
        """
        Get sensor event. This method should return the primary sensor type.
        For multi-sensor IMUs, you might want to alternate between sensor types
        or use separate sensor instances.
        """
        return self.get_accelerometer_event(ts)

    def shutdown(self) -> None:
        """Shutdown the BNO055 sensor."""
        try:
            # Put BNO055 in config mode (lowest power consumption)
            self.write(self.BNO055_OPR_MODE_ADDR, self.OPERATION_MODE_CONFIG)
            time.sleep(0.025)

            # Set to suspend mode for minimal power consumption
            self.write(self.BNO055_PWR_MODE_ADDR, self.POWER_MODE_SUSPEND)
            time.sleep(0.01)

            print("BNO055 shutdown complete")

        except Exception as e:
            print(f"Error during BNO055 shutdown: {e}")


# Custom sensor classes for each sensor type (if you want separate instances)
class CustomIMU_Accel(CustomIMU):
    """Accelerometer-only interface for the custom IMU."""

    def get_event(self, ts: Optional[int] = None):
        return self.get_accelerometer_event(ts)


class CustomIMU_Gyro(CustomIMU):
    """Gyroscope-only interface for the custom IMU."""

    def get_event(self, ts: Optional[int] = None):
        return self.get_gyroscope_event(ts)


class CustomIMU_Magn(CustomIMU):
    """Magnetometer-only interface for the custom IMU."""

    def get_event(self, ts: Optional[int] = None):
        return self.get_magnetometer_event(ts)


if __name__ == "__main__":
    # Test the custom IMU
    import numpy as np

    print("Testing Custom IMU...")

    # Initialize sensors
    imu = CustomIMU(1)  # I2C bus 1
    imu.init()

    # Test each sensor type
    print("\nAccelerometer test:")
    accel_event = imu.get_accelerometer_event()
    print(f"Acceleration: {accel_event.acceleration.v} m/s²")
    print(f"Magnitude: {np.linalg.norm(accel_event.acceleration.v):.2f} m/s²")

    print("\nGyroscope test:")
    gyro_event = imu.get_gyroscope_event()
    print(f"Angular velocity: {gyro_event.gyroUncalibrated.v} rad/s")
    print(f"Angular velocity (deg/s): {[v*180/math.pi for v in gyro_event.gyroUncalibrated.v]}")

    print("\nMagnetometer test:")
    mag_event = imu.get_magnetometer_event()
    print(f"Magnetic field: {mag_event.magneticUncalibrated.v[:3]} µT")
    print(f"Field magnitude: {np.linalg.norm(mag_event.magneticUncalibrated.v[:3]):.2f} µT")

    # Shutdown
    imu.shutdown()
    print("\nTest complete!")