"""
Robot Trajectory and data frequency visualization.
Robot data export to CSV.
Date: 2026-01-02
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from const import DEMO_PATH, VIS_PATH

def visualize_trajectory(h5_file_path):
    # 1. 读取数据
    with h5py.File(h5_file_path, 'r') as f:
        # 读取末端位姿 (N, 6) -> 前三列是 XYZ
        poses = f['actual_tcp_pose'][:]
        # 读取时间戳用于颜色映射（可选）
        timestamps = f['timestamp'][:]
        # 读取夹爪数据（可选，可以用颜色表示夹爪开合）
        gripper_pos = f['actual_gripper_pos'][:]

    data_all = np.hstack((timestamps.reshape(-1, 1), poses, gripper_pos.reshape(-1, 1)))
    np.savetxt(f"{VIS_PATH}/robot_trajectory.csv", data_all, delimiter=",", header="Timestamp,X,Y,Z,Rx,Ry,Rz,GripperPos", comments="")

    # 提取 XYZ 坐标 (单位通常是米)
    x = poses[:, 0]
    y = poses[:, 1]
    z = poses[:, 2]

    # 2. 创建 3D 图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3. 绘制轨迹
    # 使用 scatter 可以根据时间或夹爪状态改变颜色
    # cmap='viridis' 会随着轨迹时间变化颜色，方便观察运动方向
    sc = ax.scatter(x, y, z, c=np.arange(len(x)), cmap='viridis', s=2, label='TCP Path')
    
    # 绘制起始点和终点
    ax.plot(x[0:1], y[0:1], z[0:1], 'go', markersize=10, label='Start') # 起点：绿色
    ax.plot(x[-1:], y[-1:], z[-1:], 'ro', markersize=10, label='End')   # 终点：红色

    # 4. 设置标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Robot End-Effector Trajectory\n{h5_file_path.split("/")[-1]}')
    
    # 5. 保持比例一致 (防止轨迹变形)
    # 计算各个轴的范围，手动设置以保持等比例显示
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.colorbar(sc, label='Time Steps')
    plt.legend()
    plt.show()

    if len(timestamps) > 1:
        print("正在生成轨迹频率分析图表...")
        
        # 1. 计算时间差 (dt)
        dt = np.diff(timestamps)
        
        # 2. 处理可能的时间戳重复 (dt=0) 或异常情况，防止除零错误
        # 将小于 1微秒 的间隔视为 1微秒
        dt = np.maximum(dt, 1e-6)
        
        # 3. 计算频率 (Hz)
        freqs = 1.0 / dt
        
        # 创建新的画布
        fig_freq = plt.figure(figsize=(12, 6))
        
        # 绘制频率变化曲线
        plt.plot(freqs, color='#ff7f0e', linewidth=1, label='Instant Frequency')
        
        # 绘制平均频率虚线
        mean_freq = np.mean(freqs)
        plt.axhline(mean_freq, color='blue', linestyle='--', label=f'Mean Freq: {mean_freq:.1f} Hz')
        
        # 设置坐标轴
        plt.title(f'Robot Control Frequency Stability\n(Target: Usually 125Hz/500Hz)', fontsize=14)
        plt.xlabel('Step Index', fontsize=12)
        plt.ylabel('Frequency (Hz)', fontsize=12)
        
        # 自动调整Y轴范围，过滤掉极端的尖峰以便观察主体
        # 使用百分位数来确定显示范围，排除偶尔的系统卡顿造成的巨大波动
        y_lower = np.percentile(freqs, 1) * 0.9
        y_upper = np.percentile(freqs, 99) * 1.1
        plt.ylim(y_lower, y_upper)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图片
        save_name = VIS_PATH / 'robot_data_frequency.svg'
        plt.savefig(save_name, dpi=150)
        print(f"频率分析图已保存为: {save_name}")

if __name__ == "__main__":
    import glob
    h5_files = glob.glob(str(DEMO_PATH / "robot_trajectory_*.h5"))
    if h5_files:
        latest_file = max(h5_files) # 默认读取最新的文件
        visualize_trajectory(latest_file)
    else:
        print("未找到 .h5 文件，请先运行采样脚本。")