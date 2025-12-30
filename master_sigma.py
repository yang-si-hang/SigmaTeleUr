"""
Master Sigma Script using ZeroMQ to publish its pose and gripper state.
Date: 2025-12-30
"""
import sys
import zmq
import time
import ctypes
import numpy as np
from forcedimension_core import dhd
from forcedimension_core.dhd.os_independent import kbHit, kbGet

def main():
    # 1. 初始化 ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")  # 绑定本地 5555 端口

    # Try to open the first available device
    if (ID := dhd.open()) == -1:
        print(f"Error: {dhd.errorGetLastStr()}")
        sys.exit(1)

    if (name := dhd.getSystemName()) is not None:
        print(isinstance(name, str))  # prints "True"
        print(name)

    print("主端已启动，正在发送数据...")

    dhd.enableForce(enable=True, ID=ID)
    dhd.expert.enableExpertMode()
    
    # 初始缓冲区q
    pos = np.zeros(3)
    orn = np.zeros((3, 3))
    gripper_angle = ctypes.c_double(0.0)

    try:
        print("按 'q' 键退出。")
        while not (kbHit() and kbGet() == 'q'):
            if dhd.setForce([0.0, 0.0, 0.0], ID=ID) == -1:
                break
            
            # 读取位置
            if (dhd.getPositionAndOrientationFrame(p_out=pos, matrix_out=orn) == -1):
                break

            # 读取夹爪角度
            if (dhd.getGripperAngleRad(out=ctypes.byref(gripper_angle)) == -1):
                break

            # 2. 存储数据
            message = {
                "pos": pos,
                "orientation": orn,
                "gripper": gripper_angle.value,
                "timestamp": time.time()
            }
            
            # 高效发送：使用 pyobj 会自动序列化
            socket.send_pyobj(message)
            
            # 控制频率，Sigma 通常建议 1kHz
            time.sleep(0.01)

    except Exception as ex:
        print(f"\n发生错误: {ex}")

    finally:
        # 关闭设备
        dhd.close(ID)
        print("\n设备已关闭。")


if __name__ == "__main__":
    main()