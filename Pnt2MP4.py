import cv2
import os
import re

def png_to_video(input_folder, output_file, frame_rate=30):
    """
    将按数字序号排序的 PNG 图像合成为视频
    :param input_folder: 包含 PNG 图像的文件夹路径
    :param output_file: 输出视频文件的路径（例如 output.mp4）
    :param frame_rate: 视频的帧率
    """
    # 获取 PNG 文件列表并按数字序号排序
    def extract_number(filename):
        match = re.search(r'\d+', filename)  # 提取文件名中的数字
        return int(match.group()) if match else float('inf')  # 若无数字则排到最后

    images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    images.sort(key=extract_number)  # 根据数字序号排序

    if not images:
        print("未找到 PNG 图像！")
        return

    # 读取第一张图像以确定视频尺寸
    first_image_path = os.path.join(input_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 MP4 编码
    video = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    # 写入每一帧图像到视频
    for image in images:
        img_path = os.path.join(input_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
        print(f"写入帧: {image}")

    # 释放视频写入器
    video.release()
    print(f"视频已保存到: {output_file}")

# 示例用法
cup = 0
time = 5
# input_folder = f"Data/real_data/pour_water/episode_{time}/track_vis_cup{cup}"  # 替换为包含 PNG 图像的文件夹路径
# output_file = f"Data/real_data/pour_water/episode_{time}/track_vis_cup{cup}.mp4"  # 输出视频文件名

input_folder = f"simulate_captured/track_vis"  # 替换为包含 PNG 图像的文件夹路径
output_file = f"simulate_captured/display.mp4"  # 输出视频文件名
frame_rate = 15  # 设置帧率
png_to_video(input_folder, output_file, frame_rate)

