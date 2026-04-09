"""
AbandonedPark 无人机模拟控制模块
使用 AirSim API 连接并控制无人机在废弃公园场景中执行任务
"""

import time
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import airsim
import numpy as np
import cv2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ParkExplorerConfig:
    """探索任务的配置参数"""
    takeoff_altitude: float = 10.0          # 起飞高度（米）
    movement_speed: float = 3.0              # 移动速度（m/s）
    image_save_dir: str = "captures"         # 图像保存目录
    camera_name: str = "0"                    # 相机名称
    waypoints: List[Tuple[float, float, float]] = None  # 路径点列表 (x, y, z)

    def __post_init__(self):
        if self.waypoints is None:
            # 默认探索路径：围绕公园
            self.waypoints = [
                (20, 0, -10),    # 向前20米，高度-10
                (20, 15, -10),   # 向右15米
                (0, 15, -12),    # 向后20米，下降2米
                (0, 0, -10),     # 向左15米，回到起点
            ]


class AbandonedParkSimulator:
    """
    废弃公园无人机模拟器控制类
    提供连接、起飞、移动、拍照、探索等功能
    """

    def __init__(self, config: Optional[ParkExplorerConfig] = None):
        """
        初始化模拟器连接

        Args:
            config: 任务配置参数，若为None则使用默认配置
        """
        self.config = config or ParkExplorerConfig()
        self.client = airsim.MultirotorClient()
        self._connected = False
        self._api_control_enabled = False

        # 确保图像保存目录存在
        Path(self.config.image_save_dir).mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        """上下文管理器入口：自动连接"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口：自动清理"""
        self.cleanup()

    def connect(self, retries: int = 3, delay: float = 1.0) -> bool:
        """
        连接到AirSim模拟器

        Args:
            retries: 重试次数
            delay: 重试间隔（秒）

        Returns:
            是否连接成功

        Raises:
            ConnectionError: 连接失败
        """
        logger.info("正在连接到 AbandonedPark 模拟器...")
        for attempt in range(1, retries + 1):
            try:
                self.client.confirmConnection()
                if self.client.ping():
                    self._connected = True
                    logger.info(f"连接成功，ping: {self.client.ping()}")
                    return True
            except Exception as e:
                logger.warning(f"连接尝试 {attempt}/{retries} 失败: {e}")
                if attempt < retries:
                    time.sleep(delay)

        raise ConnectionError("无法连接到 AirSim 模拟器，请确保模拟器已运行")

    def enable_drone_mode(self) -> bool:
        """
        启用无人机模式：获取API控制并解锁

        Returns:
            是否成功启用
        """
        if not self._connected:
            logger.error("未连接到模拟器，请先调用 connect()")
            return False

        try:
            logger.info("启用API控制...")
            self.client.enableApiControl(True)
            self._api_control_enabled = True

            logger.info("解锁无人机...")
            self.client.armDisarm(True)

            logger.info("无人机模式已启用")
            return True
        except Exception as e:
            logger.error(f"启用无人机模式失败: {e}")
            return False

    def takeoff(self, altitude: Optional[float] = None) -> bool:
        """
        起飞并悬停到指定高度

        Args:
            altitude: 目标高度（米），若为None则使用配置中的高度

        Returns:
            是否成功
        """
        target_z = -(altitude or self.config.takeoff_altitude)
        logger.info(f"起飞并爬升至 {target_z} 米高度...")

        try:
            # 起飞
            self.client.takeoffAsync().join()
            time.sleep(1)

            # 移动到目标高度
            self.client.moveToZAsync(target_z, self.config.movement_speed).join()
            logger.info(f"已稳定在 {target_z} 米")
            return True
        except Exception as e:
            logger.error(f"起飞失败: {e}")
            return False

    def capture_image(self, filename: Optional[str] = None) -> Optional[np.ndarray]:
        """
        从指定相机捕获图像并保存

        Args:
            filename: 保存的文件名，若为None则自动生成时间戳文件名

        Returns:
            RGB图像数组，若失败则返回None
        """
        logger.info("正在捕获图像...")
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    self.config.camera_name,
                    airsim.ImageType.Scene,
                    False, False
                )
            ])

            if not responses or len(responses) == 0:
                logger.error("未收到图像响应")
                return None

            response = responses[0]
            if response.width == 0 or response.height == 0:
                logger.error("图像数据无效")
                return None

            # 转换为numpy数组 (RGB)
            img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(response.height, response.width, 3)

            # 保存图像
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"park_capture_{timestamp}.jpg"

            save_path = Path(self.config.image_save_dir) / filename
            cv2.imwrite(str(save_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            logger.info(f"图像已保存: {save_path}")

            return img_rgb
        except Exception as e:
            logger.error(f"捕获图像失败: {e}")
            return None

    def move_to_position(self, x: float, y: float, z: float,
                         speed: Optional[float] = None) -> bool:
        """
        移动到指定位置

        Args:
            x, y, z: 目标坐标（z为负值表示高度）
            speed: 移动速度，若为None则使用配置速度

        Returns:
            是否成功
        """
        speed = speed or self.config.movement_speed
        try:
            logger.info(f"移动到 ({x}, {y}, {z})...")
            self.client.moveToPositionAsync(x, y, z, speed).join()
            return True
        except Exception as e:
            logger.error(f"移动失败: {e}")
            return False

    def explore(self, waypoints: Optional[List[Tuple[float, float, float]]] = None) -> None:
        """
        沿路径点探索，并在每个点拍照

        Args:
            waypoints: 路径点列表，若为None则使用配置中的路径
        """
        points = waypoints or self.config.waypoints
        logger.info(f"开始探索，共 {len(points)} 个路径点")

        for i, (x, y, z) in enumerate(points, 1):
            logger.info(f"前往路径点 {i}/{len(points)}: ({x}, {y}, {z})")
            if self.move_to_position(x, y, z):
                self.capture_image(filename=f"waypoint_{i}.jpg")
                time.sleep(1)  # 短暂悬停

        logger.info("探索完成")

    def cleanup(self) -> None:
        """清理资源：降落、锁定、释放控制"""
        logger.info("开始清理资源...")
        try:
            if self._api_control_enabled:
                logger.info("正在降落...")
                self.client.landAsync().join()

                logger.info("锁定无人机...")
                self.client.armDisarm(False)

                logger.info("禁用API控制...")
                self.client.enableApiControl(False)
                self._api_control_enabled = False
        except Exception as e:
            logger.error(f"清理过程中出错: {e}")
        finally:
            self._connected = False
            logger.info("清理完成")


def main():
    """主函数：演示如何使用 AbandonedParkSimulator"""
    logger.info("=== AbandonedPark 无人机测试 ===")

    # 创建配置（可自定义）
    config = ParkExplorerConfig(
        takeoff_altitude=10,
        movement_speed=3,
        image_save_dir="park_captures",
        waypoints=[
            (20, 0, -10),
            (20, 15, -10),
            (0, 15, -12),
            (0, 0, -10),
        ]
    )

    # 使用上下文管理器自动处理连接和清理
    try:
        with AbandonedParkSimulator(config) as simulator:
            # 启用无人机模式
            if not simulator.enable_drone_mode():
                logger.error("无法启用无人机模式，退出")
                return

            # 起飞
            if not simulator.takeoff():
                logger.error("起飞失败，退出")
                return

            # 捕获初始图像
            simulator.capture_image(filename="initial.jpg")

            # 执行探索
            simulator.explore()

    except ConnectionError as e:
        logger.error(f"连接错误: {e}")
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"未预期的错误: {e}")


if __name__ == "__main__":
    main()