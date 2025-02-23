import os
import cv2
import numpy as np
import shutil

def simulate_camera_capture(rgb_data_dir, depth_data_dir, output_dir, frame_id):
    """
    模拟相机拍摄过程，读取指定的RGB和深度图并保存到指定文件夹。
    
    参数:
    - rgb_data_dir (str): RGB图像的文件夹路径。
    - depth_data_dir (str): 深度图像的文件夹路径。
    - output_dir (str): 输出文件夹路径，保存拍摄的RGB和深度图像。
    - frame_ids (list of int): 需要捕捉的帧ID列表。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 构造RGB图像和深度图像的文件路径
    rgb_image_path = os.path.join(rgb_data_dir, f"{frame_id:06d}.png")
    depth_image_path = os.path.join(depth_data_dir, f"{frame_id:06d}.png")

    # 检查RGB图像和深度图像是否存在
    if os.path.exists(rgb_image_path) and os.path.exists(depth_image_path):
        # 读取RGB和深度图像
        rgb_image = cv2.imread(rgb_image_path)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # 深度图通常以无符号整数格式保存

        # 保存RGB图像和深度图像到输出文件夹
        rgb_output_path = os.path.join(output_dir, f"rgb.png")
        depth_output_path = os.path.join(output_dir, f"depth.png")

        cv2.imwrite(rgb_output_path, rgb_image)
        cv2.imwrite(depth_output_path, depth_image)

        # print(f"已保存 RGB 和深度图像：{rgb_output_path} 和 {depth_output_path}")
    else:
        print(f"未找到帧 {frame_id:06d} 的 RGB 或深度图像，跳过...")


def simulate_camera_first_capture(rgb_data_dir, depth_data_dir, output_dir, frame_id=0):
    """
    模拟相机拍摄过程，读取指定的RGB和深度图并保存到指定文件夹。
    
    参数:
    - rgb_data_dir (str): RGB图像的文件夹路径。
    - depth_data_dir (str): 深度图像的文件夹路径。
    - output_dir (str): 输出文件夹路径，保存拍摄的RGB和深度图像。
    - frame_ids (list of int): 需要捕捉的帧ID列表。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 构造RGB图像和深度图像的文件路径
    rgb_image_path = os.path.join(rgb_data_dir, f"{frame_id:06d}.png")
    depth_image_path = os.path.join(depth_data_dir, f"{frame_id:06d}.png")

    # 检查RGB图像和深度图像是否存在
    if os.path.exists(rgb_image_path) and os.path.exists(depth_image_path):
        # 读取RGB和深度图像
        rgb_image = cv2.imread(rgb_image_path)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # 深度图通常以无符号整数格式保存

        # 保存RGB图像和深度图像到输出文件夹
        rgb_output_path = os.path.join(output_dir, f"rgb.png")
        depth_output_path = os.path.join(output_dir, f"depth.png")
        ref_image_output_path = os.path.join(output_dir, f"ref_image.png")

        cv2.imwrite(rgb_output_path, rgb_image)
        cv2.imwrite(ref_image_output_path, rgb_image)
        cv2.imwrite(depth_output_path, depth_image)

        print(f"已保存首帧 RGB 和深度图像：{rgb_output_path} 和 {depth_output_path}")
    else:
        print(f"未找到帧 {frame_id:06d} 的 RGB 或深度图像，跳过...")
# 示例调用
# rgb_data_dir = 'Data/real_data/pour_water/episode_0/rgb'  # RGB图像所在文件夹
# depth_data_dir = 'Data/real_data/pour_water/episode_0/depth'  # 深度图像所在文件夹
# output_dir = '/simulate_captured'  # 保存捕捉到的图像的文件夹
# frame_ids = [0, 1, 2, 3, 4]  # 需要捕捉的帧ID列表

# simulate_camera_capture(rgb_data_dir, depth_data_dir, output_dir, frame_ids)
