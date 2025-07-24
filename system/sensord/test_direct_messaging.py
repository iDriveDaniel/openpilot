#!/usr/bin/env python3
"""
Direct messaging test - minimal test case
"""

import time
import cereal.messaging as messaging
from cereal import log

def test_direct():
    print("🧪 Testing direct messaging...")

    # Step 1: Create subscriber first
    print("1. Creating subscriber...")
    poller = messaging.Poller()
    sock = messaging.sub_sock('accelerometer', poller=poller, timeout=100)
    print("   Subscriber created")

    # Step 2: Wait a moment for connection
    time.sleep(0.5)

    # Step 3: Create publisher
    print("2. Creating publisher...")
    pm = messaging.PubMaster(['accelerometer'])
    print("   Publisher created")

    # Step 4: Wait for ZMQ to establish connections
    time.sleep(0.5)

    # Step 5: Publish message
    print("3. Publishing message...")
    msg = messaging.new_message('accelerometer', valid=True)

    # Create sensor data
    event = log.SensorEventData.new_message()
    event.timestamp = time.monotonic_ns()
    event.version = 1
    event.sensor = 1
    event.type = 1
    event.source = 0

    a = event.init('acceleration')
    a.v = [1.0, 2.0, 9.8]
    a.status = 1

    msg.accelerometer = event
    pm.send('accelerometer', msg)
    print(f"   Message sent: {msg.accelerometer.acceleration.v}")

    # Step 6: Try to receive
    print("4. Attempting to receive...")
    received = False
    for i in range(10):  # Try for 10 seconds
        for socket in poller.poll(1000):  # 1 second timeout
            if socket == sock:
                recv_msg = messaging.recv_one(socket)
                if recv_msg is not None:
                    print(f"   ✅ Message received: {recv_msg.accelerometer.acceleration.v}")
                    received = True
                    break
        if received:
            break
        print(f"   Attempt {i+1}/10...")

    if not received:
        print("   ❌ No message received")

    return received

if __name__ == "__main__":
    success = test_direct()
    print(f"\n🎯 Test result: {'PASS' if success else 'FAIL'}")