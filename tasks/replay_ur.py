"""
python replay_ur.py \
    --robot.type=ur10e \
    --dataset.repo_id=your_username/your_ur10e_dataset \
    --dataset.episode=0 \
    --play_sounds=False

Date: 2026-01-13
"""
from lerobot.scripts import lerobot_replay
from lerobot.robots import ROBOT_CONFIG_REGISTRY, ROBOT_CLASS_REGISTRY
from robots.lerobot_robot_ur10e.lerobot_robot_ur10e.ur10e import UR10eRobot, UR10eRobotConfig

# --- 关键：手动注册自定义机器人 ---
ROBOT_CONFIG_REGISTRY["ur10e"] = UR10eRobotConfig
ROBOT_CLASS_REGISTRY["ur10e"] = UR10eRobot

def main():
    # 这会调用 lerobot 原生的解析器，但现在它认识 "ur10e" 了
    lerobot_replay.main()

if __name__ == "__main__":
    main()