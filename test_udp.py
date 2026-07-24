import socket
import time
import math

TEENSY_IP = "192.168.1.15"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Teensy({TEENSY_IP}:{UDP_PORT})로 2채널 모터 테스트 패킷 전송을 시작합니다.")
print("종료하려면 Ctrl + C를 누르세요.")

start_time = time.time()

try:
    while True:
        elapsed = time.time() - start_time
        
        # 1번 모터와 11번 모터가 각각 다르게 움직이도록 신호 생성
        angle_motor_1 = 0.5 * math.sin(elapsed * 2.0)
        angle_motor_11 = 0.3 * math.cos(elapsed * 1.5)

        # 포맷 문자열 생성: "P,1번값,11번값"
        message = f"P,{angle_motor_1:.3f},{angle_motor_11:.3f}"
        
        sock.sendto(message.encode(), (TEENSY_IP, UDP_PORT))
        
        print(f"전송 중 -> {message}   ", end="\r")
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n[안내] 사용자가 전송을 중단했습니다.")