"""
Slave UR Script using ZeroMQ to follow Master Sigma's pose and gripper state.
Date: 2025-12-30
"""
import time
import zmq
import numpy as np
from scipy.spatial.transform import Rotation as R
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

import utilize.robotiq_gripper as robotiq_gripper
from utilize.const import ROBOT_IP, PORT

GRIPPER_ENC_MAX, GRIPPER_ENC_MIN = 0.53, 0.0
# 旋转矩阵：从 Slave base frame 到 Master base frame 的变换
rot_slave0_master0 = R.from_matrix(np.array([[0, 1, 0],
                                             [0, 0, -1],
                                             [-1, 0, 0]]))

def gripper_enc_to_pos(enc:float) -> int:
    """Convert gripper encoder value to position (0-255)."""
    if enc < GRIPPER_ENC_MIN:
        enc = GRIPPER_ENC_MIN
    elif enc > GRIPPER_ENC_MAX:
        enc = GRIPPER_ENC_MAX
    pos = int((GRIPPER_ENC_MAX - enc) / (GRIPPER_ENC_MAX - GRIPPER_ENC_MIN) * 255)
    return pos

def main(FPS:int=100):
    gripper = robotiq_gripper.RobotiqGripper()
    gripper.connect(ROBOT_IP, PORT)
    gripper.activate()

    rtde_r = RTDEReceiveInterface(ROBOT_IP, frequency=200)
    rtde_c = RTDEControlInterface(ROBOT_IP, frequency=200)

    init_pose = rtde_r.getActualTCPPose()
    print(f"初始位姿: {init_pose}")
    init_pose[3:6] = R.from_matrix(np.array([[-1, 0, 0],
                                             [0, 0, -1],
                                             [0, -1, 0]])).as_rotvec()
    rtde_c.moveL(init_pose, 0.2, 0.2, False)
    zero_pos = rtde_r.getActualTCPPose()[0:3]
    zero_orn_mat = R.from_rotvec(rtde_r.getActualTCPPose()[3:6]).as_matrix()

    # 初始化 UR 发布者
    pub_context = zmq.Context()
    pub_socket = pub_context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5556")
    time.sleep(0.1)  # 让订阅者有机会连接

    sub_context = zmq.Context()
    sub_socket = sub_context.socket(zmq.SUB)
    sub_socket.connect("tcp://localhost:5555")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")   # 订阅所有消息
    sub_socket.setsockopt(zmq.CONFLATE, 1)  # 只保留最新一帧数据

    print("从端已启动，正在跟随主端...")

    try:
        while True:
            start_time = time.perf_counter()

            # 非阻塞或阻塞接收最新位姿
            data = sub_socket.recv_pyobj()
            pos = data.get("pos", np.zeros(3))
            orn = data.get("orientation", np.zeros((3, 3)))
            gripper_enc = data.get("gripper", 0.0)
            connected = data.get("connected", False)
            timestamp = data.get("timestamp", 0.0)

            latency = time.time() - timestamp
            print(f"数据延迟: {latency:.4f} 秒")

            if connected:
                target_pos = zero_pos + np.array([-pos[1], pos[0], pos[2]])*0.5
                r = R.from_matrix(orn)
                target_orn_mat = zero_orn_mat @ rot_slave0_master0.as_matrix() @ r.as_matrix() @ rot_slave0_master0.inv().as_matrix()
                target_pose =  np.concatenate((target_pos, R.from_matrix(target_orn_mat).as_rotvec()))
                print(f"目标位姿: {target_pose.round(4)}")
                # rtde_c.moveL(target_pose, 0.2, 0.4, True)
            else:
                current_pose = rtde_r.getActualTCPPose()
                target_pose = current_pose.copy()
                target_pose[3:6] = R.from_matrix(zero_orn_mat).as_rotvec()
                print("主端未连接，保持当前位置")
            rtde_c.servoL(target_pose, 0.2, 0.2, 0.01, 0.1, 100)

            current_pose = rtde_r.getActualTCPPose()
            print(f"实际位姿: {[round(x, 4) for x in current_pose]}")

            griper_pos = gripper_enc_to_pos(gripper_enc)
            gripper.move(griper_pos, 255, 255)
            print(f"设置 Gripper 位置到: {griper_pos}")

            ur_data = {
                "actual_tcp_pose": rtde_r.getActualTCPPose(),
                "actual_gripper_pos": gripper.get_current_position(),
                "timestamp": time.time()
            }
            pub_socket.send_pyobj(ur_data)

            # 控制频率
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, (1.0 / FPS) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)


    except KeyboardInterrupt:
        print("\n跟随已停止")

    finally:
        print("断开连接...")
        gripper.disconnect()
        # rtde_c.servoStop()
        # rtde_c.stopScript()
        rtde_r.disconnect()
        rtde_c.disconnect()
        pub_socket.close()
        pub_context.term()
        sub_socket.close()
        sub_context.term()

if __name__ == "__main__":
    main()