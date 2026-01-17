from dataclasses import dataclass, field
from lerobot.robots.utils import RobotConfig
from lerobot.cameras import CameraConfig
from const import ROBOT_IP, PORT

@RobotConfig.register_subclass("ur10e")
@dataclass
class Ur10eRobotConfig(RobotConfig):
    # 机器人 IP 地址
    robot_ip: str = ROBOT_IP
    gripper_port: int = PORT
    # 运动参数
    velocity: float = 0.5
    acceleration: float = 0.3
    # 这里的 type 字符串要和 CLI 匹配
    type: str = "ur10e"
    absolute: bool = True
    cameras: dict[str, CameraConfig] = field(default_factory=dict)