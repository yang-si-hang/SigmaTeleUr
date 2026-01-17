"""
测试 Lerobot Camera Orbbec 相机的基本功能: 连接、读取彩色图像和深度图像、显示图像等。

Date: 2026-01-16
"""
import time
import cv2
import numpy as np
import argparse
import logging

from lerobot.cameras.configs import ColorMode
from lerobot_camera_orbbec import OrbbecCamera, OrbbecCameraConfig

# 设置日志显示
logging.basicConfig(level=logging.INFO)

def main(args):
    # 1. 配置相机
    # 如果不知道序列号，先传个空字符或名字，类内部应该能处理(如果有实现find逻辑)
    # 或者先运行 lerobot-find-cameras orbbec
    config = OrbbecCameraConfig(
        serial_number_or_name=args.serial,
        fps=30,
        width=640,
        height=480,
        use_depth=args.use_depth,
        color_mode=ColorMode.RGB, # 我们在类里强制转为了 RGB
    )

    print(f"Initializing camera with config: {config}")
    camera = OrbbecCamera(config)

    try:
        # 2. 连接相机
        print("Connecting...")
        camera.connect()
        print(f"Connected! Camera SN: {camera.serial_number}")

        # 3. 循环读取
        print("Starting stream. Press 'q' to exit.")
        
        frame_count = 0
        start_time = time.time()

        while True:
            # 一次性获取同步的 color 和 depth
            data = camera.async_read(timeout_ms=1000)
            color_image = data["color"]
            depth_image = data.get("depth") # 如果没开 use_depth，这里是 None

            # 可视化 BGR 转换
            show_img = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            if depth_image is not None:
                # 归一化深度用于显示
                depth_view = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_view, cv2.COLORMAP_JET)
                
                # 对齐尺寸后拼接
                if show_img.shape[:2] != depth_colormap.shape[:2]:
                    depth_colormap = cv2.resize(depth_colormap, (show_img.shape[1], show_img.shape[0]))
                display_img = np.hstack((show_img, depth_colormap))
            else:
                display_img = show_img

            cv2.imshow("Gemini 335 Test", display_img)

            # 计算 FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"FPS: {fps:.2f}")

            # 退出检测
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 4. 断开连接 (非常重要，否则下次运行可能连不上)
        print("Disconnecting...")
        camera.disconnect()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # CP9JA530008V  Orbbec Gemini 335 的默认序列号
    parser.add_argument("--serial", type=str, default="CP9JA530008V", help="Serial number or Camera Name")
    parser.add_argument("--use_depth", action="store_true", help="Enable depth stream")
    args = parser.parse_args()
    
    # 示例用法:
    # python test_gemini.py --use_depth
    # python test_gemini.py --serial "AY123456" --use_depth
    main(args)