import pyrealsense2 as rs
import numpy as np
import cv2
import os
import shutil


def save_rgb_and_depth_images(base_save_folder="output", width=640, height=480, fps=30):
    """
    采集 Intel RealSense 的 RGB 和深度图像，并分别保存到独立的文件夹中。
    每次运行时将清空之前保存的文件，并按顺序从 0 开始编号保存文件。

    参数:
    - base_save_folder: 基础保存文件夹路径 (将在该目录下创建 rgb 和 depth 文件夹)。
    - width: 图像宽度 (分辨率)。
    - height: 图像高度 (分辨率)。
    - fps: 帧率 (Frames Per Second)。
    """
    # 创建文件夹路径
    rgb_folder = os.path.join(base_save_folder, "rgb")
    depth_folder = os.path.join(base_save_folder, "depth")

    # 清空并重新创建文件夹
    if os.path.exists(rgb_folder):
        shutil.rmtree(rgb_folder)  # 删除文件夹及其内容
    if os.path.exists(depth_folder):
        shutil.rmtree(depth_folder)
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)

    # 初始化 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置分辨率和流类型
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # 开始摄像头流
    pipeline.start(config)

    print("Press 'q' to quit...")

    rgb_index = 0  # RGB 文件编号
    depth_index = 0  # 深度文件编号

    try:
        # 主循环
        while True:
            # 获取数据帧
            frames = pipeline.wait_for_frames()

            # 提取 RGB 和深度流
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 将帧转换为数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 深度图归一化并转换为可视图像
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 显示 RGB 和深度图像
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow("RealSense RGB & Depth", images)

            # 生成保存文件名（按顺序编号）
            rgb_filename = os.path.join(rgb_folder, f"{rgb_index}.png")
            depth_filename = os.path.join(depth_folder, f"{depth_index}.png")

            # 保存 RGB 和深度图像
            cv2.imwrite(rgb_filename, color_image)
            cv2.imwrite(depth_filename, depth_image)

            # 更新文件编号
            rgb_index += 1
            depth_index += 1

            # 按下 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 停止摄像头流
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 调用函数
    save_rgb_and_depth_images("Data/my_data/tmp_linemod")
