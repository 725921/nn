"""
AirSim 无人机控制脚本，用于在 AbandonedPark 环境中执行简单的探索任务。
主要功能：连接模拟器、解锁无人机、起飞、沿路径点飞行并拍摄图像、降落。
"""

import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional

import airsim
import cv2
import numpy as np

# 配置日志输出格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AbandonedPark")


class AbandonedParkSimulator:
    """封装与 AbandonedPark 模拟器中无人机的交互操作。"""

    # 默认相机名称（前置摄像头）
    DEFAULT_CAMERA_NAME = "0"
    # 默认图像保存目录
    IMAGE_SAVE_DIR = Path("captures")

    def __init__(self, altitude: float = 10.0, waypoints: Optional[List[Tuple[float, float, float]]] = None):
        """
        初始化模拟器连接并设置任务参数。

        :param altitude: 默认飞行高度（米，负值表示上升）
        :param waypoints: 探索路径点列表，每个点为 (x, y, z) 坐标
        """
        self.altitude = altitude
        self.waypoints = waypoints or [
            (20, 0, -10),   # 向前20米
            (20, 15, -10),  # 向右15米
            (0, 15, -12),   # 向后20米，下降2米
            (0, 0, -10),    # 向左15米，回到起点
        ]

        self.client = airsim.MultirotorClient()
        self._connect()

        # 确保图像保存目录存在
        self.IMAGE_SAVE_DIR.mkdir(exist_ok=True)

    def _connect(self) -> None:
        """建立与 AirSim 模拟器的连接并确认连接状态。"""
        logger.info("正在连接到 AbandonedPark 模拟器...")
        self.client.confirmConnection()
        if self.client.ping():
            logger.info("连接成功，模拟器响应正常。")
        else:
            logger.warning("连接成功但 ping 无响应，可能存在问题。")

    def ensure_drone_mode(self) -> bool:
        """
        启用 API 控制并解锁无人机。

        :return: 是否成功进入无人机控制模式
        """
        logger.info("切换到无人机模式...")
        try:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            logger.info("无人机已解锁，API 控制已启用。")
            return True
        except Exception as e:
            logger.error(f"切换无人机模式失败: {e}")
            logger.error("请确保模拟器中已选择无人机模式（多旋翼）。")
            return False

    def takeoff_and_hover(self) -> None:
        """起飞并悬停到指定高度。"""
        logger.info(f"起飞并悬停至 {self.altitude} 米高度...")
        self.client.takeoffAsync().join()
        time.sleep(2)  # 等待稳定
        self.client.moveToZAsync(-self.altitude, 3).join()
        logger.info("悬停完成。")

    def capture_park_image(self) -> Optional[np.ndarray]:
        """
        从前置摄像头捕获一张场景图像并保存到文件。

        :return: RGB 图像数组，若失败则返回 None
        """
        logger.info("捕获图像...")
        responses = self.client.simGetImages([
            airsim.ImageRequest(
                self.DEFAULT_CAMERA_NAME,
                airsim.ImageType.Scene,
                False, False   # 不压缩，返回原始数据
            )
        ])

        if not responses or len(responses) == 0:
            logger.warning("未收到图像响应。")
            return None

        response = responses[0]
        if response.image_data_uint8 is None:
            logger.warning("图像数据为空。")
            return None

        # 转换为 numpy 数组并重塑为 RGB 图像
        img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(response.height, response.width, 3)

        # 生成带时间戳的文件名并保存
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = self.IMAGE_SAVE_DIR / f"park_capture_{timestamp}.jpg"
        cv2.imwrite(str(file_path), img_rgb)
        logger.info(f"图像已保存至: {file_path}")

        return img_rgb

    def explore_park(self) -> None:
        """沿预设路径点飞行，并在每个点捕获图像。"""
        logger.info("开始探索废弃公园...")
        for i, (x, y, z) in enumerate(self.waypoints, start=1):
            logger.info(f"路径点 {i}: 飞往 ({x}, {y}, {z})")
            self.client.moveToPositionAsync(x, y, z, 3).join()
            self.capture_park_image()
            time.sleep(1)  # 短暂停顿确保图像捕获完成
        logger.info("探索完成。")

    def cleanup(self) -> None:
        """降落无人机并释放控制。"""
        logger.info("正在降落...")
        try:
            self.client.landAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            logger.info("无人机已降落，控制已释放。")
        except Exception as e:
            logger.error(f"降落过程中出现异常: {e}")

    def __enter__(self):
        """支持上下文管理器，自动进入无人机模式。"""
        self.ensure_drone_mode()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动清理。"""
        self.cleanup()


def main():
    """主执行函数，演示无人机任务流程。"""
    logger.info("=== AbandonedPark 无人机测试 ===")

    # 等待用户确认模拟器已启动
    input("请确保 AbandonedPark.exe 已运行，然后按回车继续...")

    # 创建模拟器实例（可在此调整高度或路径点）
    simulator = AbandonedParkSimulator(altitude=10)

    try:
        # 使用上下文管理器确保即使发生异常也能执行 cleanup
        with simulator:
            simulator.takeoff_and_hover()
            simulator.capture_park_image()
            simulator.explore_park()
    except KeyboardInterrupt:
        logger.warning("用户中断任务。")
    except Exception as e:
        logger.exception(f"任务执行过程中出现未处理的异常: {e}")


if __name__ == "__main__":
    main()