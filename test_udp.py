import socket
import time
import math

TEENSY_IP = "192.168.1.15"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Starting to send 2-channel motor test packets to Teensy({TEENSY_IP}:{UDP_PORT}).")
print("Press Ctrl + C to stop.")

start_time = time.time()

try:
    while True:
        elapsed = time.time() - start_time
        
        # Generate signals so motor 1 and motor 11 move differently
        angle_motor_1 = 0.5 * math.sin(elapsed * 2.0)
        angle_motor_11 = 0.3 * math.cos(elapsed * 1.5)

        # Build format string: "P,value1,value11"
        message = f"P,{angle_motor_1:.3f},{angle_motor_11:.3f}"

        sock.sendto(message.encode(), (TEENSY_IP, UDP_PORT))

        print(f"Sending -> {message}   ", end="\r")
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n[Info] User stopped the transmission.")