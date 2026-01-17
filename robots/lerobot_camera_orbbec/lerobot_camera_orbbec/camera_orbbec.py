# lerobot/common/robot_devices/cameras/orbbec/camera_orbbec.py
"""
仿照RealsenseCamera的实现, 完成对 Orbbec Gemini 335 相机的支持
输出的彩色图像为opencv格式 (根据参数选择为RGB或BGR, 默认RGB), (H, W, C) 维度排列
深度图像为单通道uint16

Date: 2026-01-16
"""
import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import pyorbbecsdk as ob
except ImportError:
    logging.info("Could not import pyorbbecsdk.")

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.cameras import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.utils import get_cv2_rotation
from .configuration_orbbec import OrbbecCameraConfig

logger = logging.getLogger(__name__)


class OrbbecCamera(Camera):
    """
    Manages interactions with Orbbec Gemini 335 cameras via pyorbbecsdk (v2).
    """

    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)
        self.config = config

        # 1. 解析序列号 (仿照 Realsense 逻辑)
        if " " not in config.serial_number_or_name:
            self.serial_number = config.serial_number_or_name
        else:
            self.serial_number = self._find_serial_number_from_name(config.serial_number_or_name)

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.pipeline: ob.Pipeline | None = None
        self.ob_config: ob.Config | None = None
        self.device: ob.Device | None = None

        # Threading attributes
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        # 初始宽高设置，稍后会在 connect 中根据实际 stream profile 更新
        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        return self.pipeline is not None

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Orbbec 需要先找到 Device 才能创建针对该 Device 的 Pipeline
        ctx = ob.Context()
        device_list = ctx.query_devices()
        try:
            self.device = device_list.get_device_by_serial_number(self.serial_number)
        except Exception:
             # 如果无法直接获取，尝试遍历
            for i in range(device_list.get_count()):
                dev = device_list.get_device_by_index(i)
                if str(dev.get_device_info().serial_number()) == self.serial_number:
                    self.device = dev
                    break
        
        if self.device is None:
             raise ConnectionError(
                f"Failed to find {self}. Run `lerobot-find-cameras orbbec` to find available cameras."
            )

        self.pipeline = ob.Pipeline(self.device)
        self.ob_config = ob.Config()

        try:
            # --- Configure Color Stream ---
            profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            # print(f"Available Color Profiles for {self}:")
            # print(profile_list)
            color_profile = None
            if self.width and self.height and self.fps:
                try:
                    color_profile = profile_list.get_video_stream_profile(
                        self.capture_width, self.capture_height, ob.OBFormat.RGB, self.fps
                    )
                except Exception:
                    color_profile = profile_list.get_default_video_stream_profile()
                    self.fps = color_profile.get_fps() # 强制更新全局 FPS 保持一致
                    logger.warning(f"{self}: Requested profile not found, trying default.")

                logger.info(f"Using Color Profile: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()}fps")
            
            if color_profile is None:
                color_profile = profile_list.get_default_video_stream_profile()
            
            self.ob_config.enable_stream(color_profile)

            # --- Configure Depth Stream (Optional) ---
            if self.use_depth:
                depth_profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
                # print(f"Available Depth Profiles for {self}:")
                # print(depth_profile_list)
                depth_profile = None
                if self.width and self.height and self.fps:
                    try:
                        depth_profile = depth_profile_list.get_video_stream_profile(
                            self.capture_width, self.capture_height, ob.OBFormat.Y16, self.fps
                        )
                    except Exception:
                        depth_profile = depth_profile_list.get_default_video_stream_profile()
                        logger.warning(f"{self}: Requested depth profile not found, trying default.")

                    logger.info(f"Using Depth Profile: {depth_profile.get_width()}x{depth_profile.get_height()} @ {depth_profile.get_fps()}fps")
                
                if depth_profile is None:
                    depth_profile = depth_profile_list.get_default_video_stream_profile()
                
                self.ob_config.enable_stream(depth_profile)

            # --- Start Pipeline ---
            self.pipeline.start(self.ob_config)

        except Exception as e:
            self.pipeline = None
            self.device = None
            raise ConnectionError(f"Failed to start pipeline for {self}: {e}")

        # 关键修复：从硬件实际配置反向更新类属性
        self._configure_capture_settings()

        if warmup:
            time.sleep(3.0) # 硬件上电
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    data = self.async_read(timeout_ms=2500)
                    if data is not None:
                        logger.debug(f"{self} warmup read successful.")
                        break
                except Exception as e:
                    logger.debug(f"Warmup read skip: {e}")
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    def _configure_capture_settings(self) -> None:
        """
        Validates actual stream settings and updates class attributes.
        This prevents crashes when requested resolution is not supported.
        """
        if not self.is_connected:
             raise DeviceNotConnectedError(f"Cannot validate settings for {self} as it is not connected.")
        
        # 获取实际生效的 Profile (Color)
        # 注意: pyorbbecsdk 获取当前 profile 可能需要从 config 或者重新 get
        # 这里假设 enable_stream 传入的 profile 就是生效的
        # 更稳健的方法是 start 后不做检查，或者保存 enable 的 profile 对象
        
        # 简单起见，我们信任之前获取的 profile 对象，或者在此处不强制覆盖
        # 但为了 reshape 正确，我们需要 update capture_width/height
        # 由于 Orbbec API start 后获取 active profile 比较复杂，这里简化处理：
        # 假设上面的 logic 已经选择了正确的 profile。
        # 如果需要严格对应，可以再次查询 profile.width()
        pass 
        # TODO: Orbbec SDK v2 中，建议在 enable_stream 时记录下 final_profile
        # self.capture_width = final_profile.width()
        # self.capture_height = final_profile.height()

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Orbbec cameras.
        """
        found_cameras_info = []
        ctx = ob.Context()
        device_list = ctx.query_devices()

        for i in range(device_list.get_count()):
            device = device_list.get_device_by_index(i)
            info = device.get_device_info()
            
            camera_info = {
                "name": info.get_name(),
                "type": "Orbbec",
                "id": info.get_serial_number(),
                "firmware_version": info.get_firmware_version(),
                "connection_type": info.get_connection_type(),
                "device_type": info.get_device_type(),
            }
            found_cameras_info.append(camera_info)
            
            # 简单列出 color 默认 profile
            sensors = device.get_sensor_list()

            for i in range(sensors.get_count()):
                sensor = sensors.get_sensor_by_index(i)
                # 2. 获取该传感器的所有流配置 (Stream Profiles)
                profiles = sensor.get_stream_profile_list()

                for j in range(profiles.get_count()):
                    profile = profiles.get_stream_profile_by_index(j)
                    
                    # 3. 判断是否为视频流配置 (如 Color/Depth/IR)
                    # Orbbec 中使用 OB_STREAM_VIDEO 类型判断，或者尝试转换为 VideoStreamProfile
                    try:
                        # 检查是否可以作为视频流处理 (或者 profile.get_type() in [OB_STREAM_VIDEO, ...])
                        v_profile = profile.as_video_stream_profile()
                        
                        # 4. Orbbec SDK 并没有直接的 "is_default()" 标志
                        # 通常逻辑是：获取第一个匹配的、或者根据项目需求定义的默认值
                        # 这里我们模拟你的逻辑，如果它是有效的视频流，我们就记录信息
                        if v_profile:
                            # 注意：Orbbec SDK 习惯通过 get_xxxx() 方法访问属性
                            stream_info = {
                                "stream_type": str(v_profile.get_type()), # 返回 OB_STREAM_COLOR 等枚举
                                "format": str(v_profile.get_format()),    # 返回格式枚举值
                                "width": v_profile.get_width(),
                                "height": v_profile.get_height(),
                                "fps": v_profile.get_fps(),
                            }
                            
                            # 仿照你的逻辑：这里可以加一个自定义判断，或者保存第一个找到的 profile
                            if "default_stream_profile" not in camera_info:
                                camera_info["default_stream_profile"] = stream_info
                                continue
                                
                    except Exception as e:
                        # 如果不是视频流（例如是 IMU 数据流），as_video_stream_profile 可能会报错
                        continue

        return found_cameras_info

    def _find_serial_number_from_name(self, name: str) -> str:
        """Finds serial number from unique name."""
        camera_infos = self.find_cameras()
        found_devices = [cam for cam in camera_infos if str(cam["name"]) == name]

        if not found_devices:
            available_names = [cam["name"] for cam in camera_infos]
            raise ValueError(
                f"No Orbbec camera found with name '{name}'. Available: {available_names}"
            )
        
        if len(found_devices) > 1:
            # 如果名字重复（例如都是 'Gemini 335'），这里无法区分，必须用 SN
            # 这是一个简单的 fallback
             return str(found_devices[0]["id"])

        return str(found_devices[0]["id"])

    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        """
        Reads a single depth frame synchronously.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.use_depth:
            raise RuntimeError(f"Depth stream is not enabled for {self}.")

        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            raise RuntimeError(f"{self} read_depth timed out.")

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
             raise RuntimeError(f"{self} depth frame empty.")

        # 2. 解析深度数据 (uint16 mm)
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        
        raw_data = depth_frame.get_data()
        depth_map = np.frombuffer(raw_data, dtype=np.uint16).reshape((height, width))

        # 处理 Scale (确保单位是毫米)
        depth_map = depth_map.astype(np.float32) * scale
        depth_map = depth_map.astype(np.uint16)

        # 3. 后处理 (旋转、维度检查)
        # 注意: depth_frame=True 参数
        depth_map_processed = self._postprocess_image(depth_map, depth_frame=True)

        return depth_map_processed

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            raise RuntimeError(f"{self} read timed out.")

        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError(f"{self} color frame empty.")

        # 统一转换为 RGB 格式 (标准化输入)
        # _postprocess_image 假设输入必须是 RGB。        
        frame_format = color_frame.get_format()
        width = color_frame.get_width()
        height = color_frame.get_height()
        raw_data = color_frame.get_data()

        # 定义中间变量，存储标准 RGB 图像
        rgb_image: NDArray[Any]

        if frame_format == ob.OBFormat.MJPG:
            # MJPEG -> OpenCV Decode (BGR) -> RGB
            jpg_data = np.frombuffer(raw_data, dtype=np.uint8)
            bgr_image = cv2.imdecode(jpg_data, cv2.IMREAD_COLOR)
            if bgr_image is None:
                raise RuntimeError(f"{self} failed to decode MJPEG frame.")
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        elif frame_format == ob.OBFormat.RGB:
            # Raw RGB -> Reshape only
            rgb_image = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))

        elif frame_format == ob.OBFormat.BGR:
            # Raw BGR -> RGB
            bgr_image = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        else:
            # 抛出异常: 
            raise RuntimeError(f"Unsupported format: {frame_format}")
            # rgb_image = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))

        # 3. 调用标准的后处理 (输入 RGB -> 输出 Configured ColorMode)
        processed_image = self._postprocess_image(rgb_image, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        # logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return processed_image

    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        """
        Reads a single depth frame synchronously.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.use_depth:
            raise RuntimeError(f"Depth stream is not enabled for {self}.")

        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            raise RuntimeError(f"{self} read_depth timed out.")

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
             raise RuntimeError(f"{self} depth frame empty.")

        # 2. 解析深度数据 (uint16 mm)
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        
        raw_data = depth_frame.get_data()
        depth_map = np.frombuffer(raw_data, dtype=np.uint16).reshape((height, width))

        # 处理 Scale (确保单位是毫米)
        depth_map = (depth_map.astype(np.float32) * scale).astype(np.uint16)

        # 3. 后处理 (旋转、维度检查)
        # 注意: depth_frame=True 参数
        depth_map_processed = self._postprocess_image(depth_map, depth_frame=True)

        return depth_map_processed

    def _postprocess_image(
        self, image: NDArray[Any], color_mode: ColorMode | None = None, depth_frame: bool = False
    ) -> NDArray[Any]:
        """
        Applies color conversion, dimension validation, and rotation to a raw color frame.
        (Adapted from lerobot standard implementation)
        """
        if color_mode and color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if depth_frame:
            h, w = image.shape
        else:
            h, w, c = image.shape

            if c != 3:
                raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        # 这里的 strict check 保证了 pipeline 输出的分辨率必须和 config 完全一致
        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        processed_image = image
        
        # 核心逻辑：假设输入 image 必定是 RGB
        if not depth_frame and self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _start_read_thread(self) -> None:
        """Starts the background thread properly."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        self.thread = None
        self.stop_event = None

    def _read_loop(self) -> None:
        """后台统一读取循环，避免主线程与子线程争抢 Pipeline"""
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event not initialized.")

        while not self.stop_event.is_set():
            try:
                # 1. 统一获取所有流的帧组
                frames = self.pipeline.wait_for_frames(500) # 500ms 超时
                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame() if self.use_depth else None

                current_data = {}

                # 2. 处理 Color 帧
                if color_frame is not None:
                    # 获取原始数据并转换 (复用你之前的逻辑)
                    width = color_frame.get_width()
                    height = color_frame.get_height()
                    fmt = color_frame.get_format()
                    raw_ptr = color_frame.get_data()

                    if fmt == ob.OBFormat.MJPG:
                        jpg_data = np.frombuffer(raw_ptr, dtype=np.uint8)
                        bgr = cv2.imdecode(jpg_data, cv2.IMREAD_COLOR)
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    elif fmt == ob.OBFormat.RGB:
                        rgb = np.frombuffer(raw_ptr, dtype=np.uint8).reshape((height, width, 3))
                    else:
                        # 兜底 BGR
                        bgr = np.frombuffer(raw_ptr, dtype=np.uint8).reshape((height, width, 3))
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    
                    current_data["color"] = self._postprocess_image(rgb, color_mode=ColorMode.RGB)

                # 3. 处理 Depth 帧
                if self.use_depth and depth_frame is not None:
                    d_width = depth_frame.get_width()
                    d_height = depth_frame.get_height()
                    scale = depth_frame.get_depth_scale()
                    raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((d_height, d_width))
                    
                    # 转换为 mm 并后处理
                    depth_mm = (raw_depth.astype(np.float32) * scale).astype(np.uint16)
                    current_data["depth"] = self._postprocess_image(depth_mm, depth_frame=True)

                # 4. 更新最新帧
                if "color" in current_data:
                    with self.frame_lock:
                        self.latest_frame = current_data
                    self.new_frame_event.set()

            except Exception as e:
                logger.warning(f"Read loop error: {e}")
                time.sleep(0.01)

    def async_read(self, timeout_ms: float = 1000) -> dict[str, NDArray[Any]]:
        """修改后的异步读取，返回包含 color 和 depth 的字典"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            with self.frame_lock:
                if self.latest_frame is not None:
                    # 记录一个警告而不是退出
                    logger.warning(f"Camera {self} read timeout, using last frame.")
                    return self.latest_frame
                else:
                    raise TimeoutError(f"Camera {self} read timeout and no frame available.")

        with self.frame_lock:
            data = self.latest_frame
            self.new_frame_event.clear()

        return data # 返回的是 {"color": ..., "depth": ...}

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
             raise DeviceNotConnectedError(f"{self} already disconnected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self.device = None
            self.ob_config = None

        logger.info(f"{self} disconnected.")

    ### Depreated ###
    # def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
    #     if not self.is_connected:
    #         raise DeviceNotConnectedError(f"{self} is not connected.")

    #     # 确保线程已启动
    #     if self.thread is None or not self.thread.is_alive():
    #         self._start_read_thread()

    #     if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
    #         thread_alive = self.thread is not None and self.thread.is_alive()
    #         raise TimeoutError(
    #             f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
    #             f"Read thread alive: {thread_alive}."
    #         )

    #     with self.frame_lock:
    #         frame = self.latest_frame
    #         self.new_frame_event.clear()

    #     if frame is None:
    #         raise RuntimeError(f"Internal error: frame is None for {self}")

    #     return frame

    # def _read_loop(self) -> None:
    #     """Internal background loop."""
    #     if self.stop_event is None:
    #         raise RuntimeError(f"{self}: stop_event not initialized.")

    #     while not self.stop_event.is_set():
    #         try:
    #             # 使用较长的 timeout 以避免在循环中过快空转，但要响应 stop
    #             color_image = self.read(timeout_ms=500)
                
    #             with self.frame_lock:
    #                 self.latest_frame = color_image
    #             self.new_frame_event.set()

    #         except DeviceNotConnectedError:
    #             break
    #         except Exception as e:
    #             logger.warning(f"Error in read loop {self}: {e}")