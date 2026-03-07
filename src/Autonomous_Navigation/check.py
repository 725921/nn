"""
无人机状态检查脚本
用于验证 AirSim 模拟器连接、获取无人机状态并解锁
"""

import time
import logging
from typing import Optional

import airsim

# 配置日志（简单输出到控制台）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def connect_to_simulator(retries: int = 3, delay: float = 1.0) -> Optional[airsim.MultirotorClient]:
    """
    连接到 AirSim 模拟器，支持重试

    Args:
        retries: 最大重试次数
        delay: 重试间隔（秒）

    Returns:
        成功时返回 client 对象，失败返回 None
    """
    for attempt in range(1, retries + 1):
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            # 可选：验证连接是否有效
            if client.ping():
                logger.info(f"连接成功（尝试 {attempt}/{retries}）")
                return client
        except Exception as e:
            logger.warning(f"连接失败（尝试 {attempt}/{retries}）: {e}")
            if attempt < retries:
                time.sleep(delay)
    logger.error("无法连接到模拟器，请确保 AbandonedPark.exe 已运行")
    return None


def get_drone_state(client: airsim.MultirotorClient) -> Optional[airsim.MultirotorState]:
    """
    安全地获取无人机状态

    Args:
        client: AirSim 客户端

    Returns:
        成功返回状态对象，失败返回 None
    """
    try:
        state = client.getMultirotorState()
        return state
    except Exception as e:
        logger.error(f"获取无人机状态失败: {e}")
        return None


def enable_drone_control(client: airsim.MultirotorClient) -> bool:
    """
    启用 API 控制并解锁无人机

    Args:
        client: AirSim 客户端

    Returns:
        是否成功
    """
    try:
        logger.info("启用 API 控制...")
        client.enableApiControl(True)
        logger.info("解锁无人机...")
        client.armDisarm(True)
        return True
    except Exception as e:
        logger.error(f"解锁失败: {e}")
        return False


def print_state_info(state: airsim.MultirotorState) -> None:
    """格式化输出无人机状态信息"""
    pos = state.kinematics_estimated.position
    print(f"当前位置: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}")
    print(f"当前速度: {state.speed:.2f} m/s")
    print(f"电池电量: {state.battery:.1f}%")
    print(f"是否碰撞: {'是' if state.collision.has_collided else '否'}")


def main():
    """主函数：执行状态检查流程"""
    print("=" * 50)
    print("无人机状态检查")
    print("=" * 50)

    # 1. 连接模拟器
    client = connect_to_simulator()
    if not client:
        print("❌ 连接失败，请检查模拟器是否运行")
        return

    print("✓ 已连接到 AbandonedPark 模拟器")

    # 2. 获取并显示状态
    state = get_drone_state(client)
    if state:
        print_state_info(state)
    else:
        print("⚠️ 无法获取完整状态信息")

    # 3. 解锁无人机（可选步骤，可根据需求决定是否执行）
    print("\n准备解锁无人机...")
    if enable_drone_control(client):
        print("✓ 无人机已解锁，准备就绪")
    else:
        print("❌ 解锁失败，请检查模拟器是否处于无人机模式")

    print("\n" + "=" * 50)
    print("状态检查完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()