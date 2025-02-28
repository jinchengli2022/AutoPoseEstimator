import os
import time
import json
import cv2
import pyrealsense2 as rs
import numpy as np

def create_task_directory(task_name):
    task_dir = os.path.join(os.getcwd(), task_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    return task_dir

def create_episode_directory(task_dir, episode_index):
    """
    创建episode文件夹
    """
    episode_dir = os.path.join(task_dir, f"episode_{episode_index}")
    if not os.path.exists(episode_dir):
        os.makedirs(episode_dir)
        os.makedirs(os.path.join(episode_dir, "depth"))
        os.makedirs(os.path.join(episode_dir, "rgb"))
    return episode_dir

def save_scene_camera_data(episode_dir, episode_index, timestamp, depth_sensor):
    """
    保存scene_camera.json数据,动态读取相机内参和深度比例因子
    """
    # 获取相机的 depth 传感器的内参
    depth_intrinsics = depth_sensor.get_stream_profiles()[0].as_video_stream_profile().get_intrinsics()

    # 手动构造内参矩阵 K
    cam_K = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                    [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                    [0, 0, 1]])


    # 获取深度比例因子
    depth_scale = depth_sensor.get_depth_scale()

    scene_data = {
        "episode": episode_index,
        "timestamp": timestamp,
        "camera_intrinsics": {
            "cam_K": cam_K.flatten().tolist(),
            "depth_scale": depth_scale
        }
    }

    scene_camera_path = os.path.join(episode_dir, "scene_camera.json")
    with open(scene_camera_path, 'w') as json_file:
        json.dump(scene_data, json_file, indent=4)

def capture_data(episode_dir, episode_index, rgb_frame, depth_frame, frame_count):
    """
    捕获并保存RGB和Depth图像，文件命名为递增的编号
    """
    # 将RGB和深度图像转换为合适的格式
    rgb_image = np.asanyarray(rgb_frame)
    depth_image = np.asanyarray(depth_frame)

    # 保存RGB图像
    rgb_filename = f"{frame_count:06d}.png"
    rgb_path = os.path.join(episode_dir, "rgb", rgb_filename)
    cv2.imwrite(rgb_path, rgb_image.astype(np.uint8))

    # 保存Depth图像
    depth_filename = f"{frame_count:06d}.png"
    depth_path = os.path.join(episode_dir, "depth", depth_filename)
    cv2.imwrite(depth_path, depth_image.astype(np.uint16))

def main():
    """
    主程序
    """
    # 输入任务名
    task_name = input("请输入任务名称: ")
    task_name = task_name.replace(" ", "_")
    task_dir = create_task_directory(task_name)

    # 配置RealSense相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)   # Depth流

    # 启动RealSense相机
    pipeline.start(config)

    # 创建对齐对象，将深度图对齐到RGB图像
    align_to = rs.stream.color
    align = rs.align(align_to)
 
    # 获取深度传感器
    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()

    episode_index = 0
    recording = False
    frame_count = 0  # 记录当前frame的编号

    print("按 'b' 开始录制数据，按 's' 结束录制数据。")
    for i in range(20):
        frames = pipeline.wait_for_frames()
    while True:
        # 获取当前帧
        frames = pipeline.wait_for_frames()

        # 对齐深度图像到RGB图像
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 获取当前时间戳
        timestamp = time.time()

        # 显示RGB图像
        depth_image = np.asanyarray(depth_frame.get_data())
        rgb_image = np.asanyarray(color_frame.get_data())
        # print(rgb_image.shape, depth_image.shape)

        # 可视化深度图像（应用颜色映射）
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 显示RGB图像
        cv2.imshow('RGB Image', rgb_image)
        
        # 显示深度图像
        cv2.imshow('Depth Image', depth_colormap)
        
        # 监听按键
        key = cv2.waitKey(1) & 0xFF

        # 按 'b' 键开始录制
        if key == ord('b') and not recording:
            print(f"开始录制数据，任务: {task_name}, Episode: {episode_index}")
            recording = True
            episode_dir = create_episode_directory(task_dir, episode_index)
            start_time = time.time()

        # 按 's' 键结束录制
        if key == ord('s') and recording:
            print(f"结束录制数据，任务: {task_name}, Episode: {episode_index}")
            recording = False
            end_time = time.time()
            save_scene_camera_data(episode_dir, episode_index, end_time - start_time, depth_sensor)
            episode_index += 1
            frame_count = 0  # 重置帧计数器

        # 持续录制，保存每帧数据
        if recording:
            capture_data(episode_dir, episode_index, rgb_image, depth_image, frame_count)
            frame_count += 1  # 增加帧计数器

        # 按 'q' 键退出
        if key == ord('q'):
            break

    # 停止RealSense相机
    pipeline.stop()

if __name__ == "__main__":
    main()
