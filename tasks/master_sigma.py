"""
Master Sigma Script using ZeroMQ to publish its pose, gripper state, and trigger status.
Date: 2025-12-30
"""
from socket import socket
from socket import socket
import sys
import zmq
import time
import ctypes
import numpy as np
from forcedimension_core import dhd, drd
from forcedimension_core import dhd, drd
from forcedimension_core.dhd.os_independent import kbHit, kbGet

CONNECT = True

def sigma_init():
    if (ID := drd.open()) == -1:
        print(f"Error: {dhd.errorGetLastStr()}")
        sys.exit(1)

    if (name := dhd.getSystemName()) is not None:
        print(isinstance(name, str))  # prints "True"
        print(name)

    if drd.isInitialized():
        print("device already initialized\n")
    else:
        print("initializing device...")
        if drd.autoInit() < 0:
            print(f"error: failed to initialize device ({dhd.errorGetLastStr()})")
            time.sleep(2.0)
            sys.exit(-1)

    if drd.start() < 0:
        print(f"error: failed to start device ({dhd.errorGetLastStr()})")
        time.sleep(2.0)
        sys.exit(-1)

    drd.moveToPos([0.0, 0.0, 0.0], block=True, ID=ID)
    drd.moveToRot([0.0, 0.0, 0.0], block=True, ID=ID)
    drd.moveToGrip(0.0, block=True, ID=ID)

    if drd.stop(True, ID=ID) < 0:
        print(f"error: failed to stop robotic regulation ({dhd.errorGetLastStr()})")
        time.sleep(2.0)
        sys.exit(-1)

    return ID

def main(FPS:int=100):
    # Try to open the first available device
    ID = sigma_init()

    # 初始化 ZeroMQ
    pub_context = zmq.Context()
    pub_socket = pub_context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5555")
    time.sleep(0.1)  # 让订阅者有机会连接

    # Optional: Receiver for feedback (if needed)
    sub_context = zmq.Context()
    sub_socket = sub_context.socket(zmq.SUB)  # or REP
    sub_socket.setsockopt(zmq.CONFLATE, 1)  # 只保留最新一帧数据
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")   # 订阅所有消息
    sub_socket.connect("tcp://localhost:5556")  # different port

    print("主端已启动，正在发送数据...")

    dhd.enableForce(enable=True, ID=ID)
    dhd.expert.enableExpertMode()
    
    # 初始缓冲区q
    pos = np.zeros(3)
    delta_pos = np.zeros(3)
    pos_zero = None
    orn = np.zeros((3, 3))
    gripper_angle = ctypes.c_double(0.0)
    pos_connect = False
    terminate = False

    try:
        print("按 'q' 键退出; 按 't' 键连接位置零点。")
        while True:
            if kbHit():
                # 2. 只读取一次按键字符并存入变量值
                key = kbGet()
                
                # 3. 针对变量进行逻辑判断
                if key == 't':
                    if not pos_connect:
                        pos_connect = True
                        if (dhd.getPositionAndOrientationFrame(p_out=pos, matrix_out=orn) == -1):
                            break
                        pos_zero = pos.copy()
                        print("Position connection enabled.")
                
                elif key == 'q':
                    terminate = True
                    pos_connect = False
                    print("退出信号已发送...")

            start_time = time.perf_counter()

            if dhd.setForceAndTorque([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], ID=ID) == -1:
                break
            
            # 读取夹爪角度
            if (dhd.getGripperAngleRad(out=ctypes.byref(gripper_angle)) == -1):
                break

            # 读取位置
            if (dhd.getPositionAndOrientationFrame(p_out=pos, matrix_out=orn) == -1):
                break

            if pos_zero is not None:
                delta_pos = pos - pos_zero

            # 存储数据
            message = {
                "pos": delta_pos,
                "orientation": orn,
                "gripper": gripper_angle.value,
                "triggered": pos_connect,
                "timestamp": time.time()
            }
            
            # 高效发送：使用 pyobj 会自动序列化
            pub_socket.send_pyobj(message)
            pub_socket.send_pyobj(message)
            
            # 控制频率
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, (1.0 / FPS) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            if terminate:
                print("退出程序...")
                break


    except Exception as ex:
        print(f"\n发生错误: {ex}")

    finally:
        dhd.close(ID)
        drd.close(ID)
        drd.close(ID)
        print("\n设备已关闭。")
        pub_socket.close()
        pub_context.term()
        sub_socket.close()
        sub_context.term()
        print("ZeroMQ 已关闭。")

        pub_socket.close()
        pub_context.term()
        sub_socket.close()
        sub_context.term()
        print("ZeroMQ 已关闭。")



if __name__ == "__main__":
    main()