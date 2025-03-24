import pyrealsense2 as rs
import numpy as np
import cv2
import os
import png
import json

class FrameSaver:
    def __init__(self, folder):
        self.folder = folder
        # self.make_directories(folder)

        # 初始化RealSense管道
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 深度流
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 彩色流
        self.profile = self.pipeline.start(self.config)

        # 获取彩色图像的内参
        color_frame = self.pipeline.wait_for_frames().get_color_frame()
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        # 构建内参矩阵
        cam_K = [
            intrinsics.fx, 0.0, intrinsics.ppx,
            0.0, intrinsics.fy, intrinsics.ppy,
            0.0, 0.0, 1.0
        ]

        self.camera_parameters = {
            "camera_intrinsics": {
                "cam_K": cam_K,
                "depth_scale": self.profile.get_device().first_depth_sensor().get_depth_scale()
            }
        }

        # 保存相机内参到 JSON 文件
        with open(self.folder + 'scene_camera.json', 'w') as fp:
            json.dump(self.camera_parameters, fp, indent=4)

        # 对齐深度图到彩色图
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def make_directories(self, folder):
        """确保文件夹存在，如果不存在则创建"""
        if not os.path.exists(folder):
            os.makedirs(folder)
    #     if not os.path.exists(folder + "depth/"):
    #         os.makedirs(folder + "depth/")

    def save_frame(self):
        """
        每次调用此方法，拍摄一张RGB图像和深度图像，保存为固定名称rgb.png和depth.png。
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 检查帧有效性
        if not aligned_depth_frame or not color_frame:
            print("帧获取失败！")
            return

        # 获取深度图和RGB图数据
        depth_data = np.asanyarray(aligned_depth_frame.get_data())
        color_data = np.asanyarray(color_frame.get_data())

        # 保存RGB图
        filecad = self.folder + "rgb.png"
        cv2.imwrite(filecad, color_data)

        # 保存深度图
        filedepth = self.folder + "depth.png"
        with open(filedepth, 'wb') as f:
            writer = png.Writer(width=depth_data.shape[1], height=depth_data.shape[0],
                                bitdepth=16, greyscale=True)
            zgray2list = depth_data.tolist()
            writer.write(f, zgray2list)

        print(f"保存了文件 {filecad} 和 {filedepth}")

    def stop_recording(self):
        """停止录制并关闭管道"""
        self.pipeline.stop()

# 使用示例
if __name__ == "__main__":
    folder_path = "Data/my_data/"  # 数据保存的路径
    frame_saver = FrameSaver(folder_path)

    # 每次调用保存一张图像
    frame_saver.save_frame()

    # 完成后停止管道
    frame_saver.stop_recording()
