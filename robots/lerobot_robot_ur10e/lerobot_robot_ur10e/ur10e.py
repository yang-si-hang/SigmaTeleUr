"""
Lerobot UR10e 机器人接口实现

Date: 2026-01-13
"""
import numpy as np
import rtde_control
import rtde_receive
from lerobot.scripts import lerobot_replay
from lerobot.robots.robot import Robot
from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError

import utilize.robotiq_gripper as robotiq_gripper
from .config_ur10e import Ur10eRobotConfig


ACTION_NAMES = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz', 'GripperPos']

class Ur10eRobot(Robot):
    config_class = Ur10eRobotConfig
    name = "ur10e"

    def __init__(self, config: Ur10eRobotConfig):
        super().__init__(config)
        self.config = config
        self.rtde_c = None
        self.rtde_r = None
        self.gripper = None
        # self.cameras = make_cameras_from_configs(config.cameras)
        self.cameras = {}

        self._initial_pose = None

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

    @property
    def observation_features(self) -> dict:
        # 匹配数据集中的 observation.state (7维) 和图像
        features = {
            "observation.state": {
                "shape": (7,),
                "dtype": "float32",
                "names": ACTION_NAMES
            }
        }
        # 添加摄像头特征 (如 observation.images.main)
        for cam_name, cam in self.cameras.items():
            features[f"observation.images.{cam_name}"] = {
                "shape": (3, cam.height, cam.width),
                "dtype": "video", # LeRobot 内部处理通常映射为 uint8/float32
            }
        return features

    @property
    def action_features(self) -> dict:
        # 匹配数据集中的 action (7维)
        return {
            "action": {
                "shape": (7,),
                "dtype": "float32",
                "names": ACTION_NAMES
            }
        }
    
    # --- 状态检查 ---
    @property
    def is_connected(self) -> bool:
        return self.rtde_c is not None and self.rtde_r is not None

    def connect(self):
        print(f"Connecting to UR10e at {self.config.robot_ip}...")
        self.rtde_c = rtde_control.RTDEControlInterface(self.config.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.config.robot_ip)
        
        print(f"Connecting to Robotiq gripper on {self.config.gripper_port}...")
        self.gripper = robotiq_gripper.RobotiqGripper()
        self.gripper.connect(self.config.robot_ip, self.config.gripper_port)
        self.gripper.activate()

        for cam in self.cameras.values():
            cam.connect()
            
        print("Robot, Gripper and Cameras Connected.")

    def disconnect(self):
        if self.rtde_c:
            self.rtde_c.servoStop()
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
        if self.rtde_r:
            self.rtde_r.disconnect()
        if self.gripper:
            self.gripper.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
        print("UR10e and Gripper Disconnected.")

    def get_observation(self) -> dict:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        state_vector = self._get_state()
        obs_dict = {
            "observation.state": state_vector
        }

        for cam_key, cam in self.cameras.items():
            obs_dict[f"observation.images.{cam_key}"] = cam.async_read()

        return obs_dict

    def send_action(self, action_dict: dict) -> dict:
        """
        action_dict: 包含 'action' 键的字典，其值为 7 维数组
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            # 1. 按照 ACTION_NAMES 的顺序提取前6个作为位姿
            target_pose = [float(action_dict[name]) for name in ACTION_NAMES[:6]]
            
            # 2. 提取夹爪目标值
            gripper_target = float(action_dict[ACTION_NAMES[6]])

            # 3. 发送机器人运动指令
            # servoL 适合平滑的流式控制
            self.rtde_c.servoL(
                target_pose, 
                self.config.velocity, 
                self.config.acceleration, 
                0.005, 0.1, 100
            )

            # 4. 发送夹爪指令
            self.gripper.move(int(gripper_target), 255, 150)

        except KeyError as e:
            print(f"Error: 接收到的字典缺少预期的键名 {e}")
            print(f"当前收到的内容是: {action_dict.keys()}")
            raise

        return action_dict

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # self._initial_pose = self._jacobi.get_ee_pose()
        # xyz = self._initial_pose[:3, 3]
        # rpy = t3d.euler.mat2euler(self._initial_pose[:3, :3])
        # print(f"Initial pose: xyz={xyz}, rpy={rpy}")

    def _get_state(self) -> np.ndarray:
        # 1. 获取位姿 (6维)
        tcp_pose = self.rtde_r.getActualTCPPose()
        # 2. 获取夹爪位置 (归一化到数据集预期的范围，假设数据集里 0-1 或 0-255)
        gripper_pos = self.gripper.get_current_position() 
        
        # 组合成 7 维状态
        state_vector = np.concatenate([tcp_pose, [gripper_pos]]).astype(np.float32)
        return state_vector
    
    def _get_img(self):
        for cam_name, cam in self.cameras.items():
            # 注意：LeRobot 默认图像格式通常是 (C, H, W)
            img = cam.read()
            # 如果 cam.read() 返回的是 (H, W, C)，需要转置:
            img = np.transpose(img, (2, 0, 1))
            

def main():
    lerobot_replay.main()

if __name__ == "__main__":
    main()