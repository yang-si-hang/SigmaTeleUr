"""
使用Orbbec Gemini相机采集彩色图像数据，并保存为HDF5格式文件
帧率设置30Hz，分辨率640x480
通过ZMQ触发录制开始和停止（来自master_sigma.py）
"""
import cv2
import numpy as np
import h5py
import zmq
import time
from pathlib import Path
from tqdm import tqdm
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode, OBFormat

from const import SAMPLE_PATH

HEIGHT, WIDTH = 480, 640


def post_compress_h5(input_file, output_file=None):
    input_path = Path(input_file)
    if output_file is None:
        output_path = input_path.parent / f"compressed_{input_path.name}"
    else:
        output_path = Path(output_file)

    print(f"开始处理: {input_path.name}")
    
    with h5py.File(input_path, 'r') as f_in:
        # 检查数据是否存在
        if 'color' not in f_in:
            print("错误：在文件中未找到 'color' 数据集！")
            return

        color_in = f_in['color']
        sys_ts_in = f_in['system_time']
        
        num_frames = color_in.shape[0]
        h, w, c = color_in.shape[1:]

        with h5py.File(output_path, 'w') as f_out:
            # 1. 复制元数据（内参等）
            if 'metadata' in f_in:
                f_in.copy('metadata', f_out)
            
            # 2. 创建压缩后的数据集
            # chunks=(1, h, w, c) 确保每一帧都能独立快速解压
            color_out = f_out.create_dataset(
                "color", 
                shape=(num_frames, h, w, c),
                dtype='uint8',
                chunks=(1, h, w, c),
                compression="gzip",
                compression_opts=4  # 无损压缩等级 4 (1-9，越大越慢)
            )
            
            # 3. 复制时间戳
            f_out.create_dataset("system_time", data=sys_ts_in[:])
            
            # 4. 逐帧搬运（防止内存溢出）
            print(f"正在压缩 {num_frames} 帧图像...")
            for i in tqdm(range(num_frames)):
                # 读取一帧，存入一帧
                frame_data = color_in[i]
                if frame_data.size > 0:
                    color_out[i] = frame_data
                else:
                    print(f"警告：第 {i} 帧数据损坏或为空")

    print(f"压缩完成！新文件已保存至: {output_path}")
    print(f"原始大小: {input_path.stat().st_size / 1024**2:.2f} MB")
    print(f"压缩后大小: {output_path.stat().st_size / 1024**2:.2f} MB")

def main():
    pipeline = Pipeline()
    config = Config()
    
    # 启用对齐
    config.set_align_mode(OBAlignMode.SW_MODE)

        # 尝试寻找最接近的分辨率且 FPS 为 30 的配置
        # 常见的为 640x480 @ 30fps
    try:

        profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = profiles.get_video_stream_profile(640, 480, OBFormat.MJPG, 30)
        except:
            # 如果没有 640x480，获取默认并尝试打印所有支持的
            color_profile = profiles.get_default_video_stream_profile()
            print(f"Warning: 指定配置不可用，使用默认: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()}fps")

        # color_profile = profiles.get_default_video_stream_profile()
        # print("支持的 Color 分辨率和帧率:")
        # for p in profiles.video_stream_profiles:
        #     print(f"- {p.get_width()}x{p.get_height()} @ {p.get_fps()}fps, {p.get_format()}")

        config.enable_stream(color_profile)
        pipeline.start(config)
        print(f"实际运行配置: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()}fps, {color_profile.get_format()}")
        
        # 获取相机内参（用于后续对齐和3D重建）
        param = pipeline.get_camera_param()
        intrinsics = param.rgb_intrinsic
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    sub_context = zmq.Context()
    sub_socket = sub_context.socket(zmq.SUB)
    sub_socket.setsockopt(zmq.CONFLATE, 1)  # 只保留最新一帧数据
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")   # 订阅所有消息
    sub_socket.connect("tcp://localhost:5555")

    # 创建 HDF5 文件
    h5_filename = str(SAMPLE_PATH / f"capture_{time.strftime('%Y%m%d_%H%M%S')}.h5")
    with h5py.File(h5_filename, 'w') as h5f:
        # 1. 存储相机元数据（内参）
        meta = h5f.create_group("metadata")
        meta.attrs['fx'] = intrinsics.fx
        meta.attrs['fy'] = intrinsics.fy
        meta.attrs['cx'] = intrinsics.cx
        meta.attrs['cy'] = intrinsics.cy

        # 2. 创建可扩展的数据集 (maxshape=None 允许持续添加帧)
        # 彩色图通常是 (H, W, 3), 深度图是 (H, W)
        color_ds = h5f.create_dataset("color", (0, HEIGHT, WIDTH, 3), 
                                      maxshape=(None, HEIGHT, WIDTH, 3), dtype='uint8',
                                      chunks=(1, HEIGHT, WIDTH, 3), compression="lzf")
       
        # 3. 时间戳数据集
        sys_ts_ds = h5f.create_dataset("system_time", (0,), maxshape=(None,), dtype='float64')
        dev_ts_ds = h5f.create_dataset("device_time", (0,), maxshape=(None,), dtype='uint64')

        print(f"开始录制至 {h5_filename}... 按 's' 采样帧，按 'p' 停止退出")
        
        prev_time = 0
        fps = 0
        fps_update_interval = 5
        frame_counter_for_fps = 0
        is_recording = False
        frame_count = 0
        cv2.namedWindow("Orbbec HDF5 Recorder", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Orbbec HDF5 Recorder", 800, 600)

        try:
            while True:
                try:
                    data = sub_socket.recv_pyobj(flags=zmq.NOBLOCK)
                    triggered = data.get("triggered", False)
                    # print(f"Triggered: {triggered}")
                    if triggered:
                        if not is_recording:
                            is_recording = True
                            print(">>> 开始录制...")

                    elif is_recording:
                        print(f">>> 停止录制。总计保存: {frame_count} 帧")
                        break
                except zmq.Again:
                    pass  # 无新数据，继续

                frames = pipeline.wait_for_frames(40)
                if not frames: continue

                color_frame = frames.get_color_frame()
                if not color_frame: continue

                current_time = time.time()
                time_diff = current_time - prev_time
                if time_diff > 0:
                    # 计算瞬时 FPS
                    actual_fps = 1.0 / time_diff
                    # 简单的平滑处理：每隔几帧更新一次显示数值
                    if frame_counter_for_fps % fps_update_interval == 0:
                        fps = actual_fps
                prev_time = current_time
                frame_counter_for_fps += 1

                raw_data = color_frame.get_data()
                if raw_data is None: continue
                
                # 转换为 numpy 数组并解码
                # data_array = np.frombuffer(raw_data, dtype=np.uint8)
                # color_image = cv2.imdecode(data_array, cv2.IMREAD_COLOR)

                # 检查解码是否成功
                # if color_image is None or color_image.size == 0:
                #     continue

                # 获取系统时间（精确到微秒）
                sys_ts = time.time()
                # 获取相机硬件时间戳（毫秒）
                dev_ts = color_frame.get_timestamp()

                # 数据转换
                color_image = cv2.imdecode(np.asanyarray(color_frame.get_data()), cv2.IMREAD_COLOR)

                # --- 录制逻辑 ---
                if is_recording:
                    # 自动增长数据集
                    new_size = frame_count + 1
                    color_ds.resize(new_size, axis=0)
                    sys_ts_ds.resize(new_size, axis=0)
                    dev_ts_ds.resize(new_size, axis=0)

                    # 写入数据
                    color_ds[frame_count] = color_image
                    sys_ts_ds[frame_count] = sys_ts
                    dev_ts_ds[frame_count] = dev_ts
                    frame_count += 1
                    
                    # 在画面上显示录制状态点（红点）
                    cv2.circle(color_image, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(color_image, f"REC: {frame_count} frames", (50, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(color_image, f"FPS: {fps:.1f}", (250, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 显示预览
                cv2.imshow("Orbbec HDF5 Recorder", color_image)
                key = cv2.waitKey(10)

                if key & 0xFF == ord('s'):
                    if not is_recording:
                        is_recording = True
                        print(">>> 开始录制...")
                
                elif key & 0xFF == ord('p'):
                    print(f">>> 停止录制。总计保存: {frame_count} 帧")
                    break

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()