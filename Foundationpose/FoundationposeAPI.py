import os
import time
import argparse
import numpy as np
import trimesh
import cv2
import yaml
from Foundationpose.datareader import LinemodReader
from estimater import *
from Utils import draw_posed_3d_box, draw_xyz_axis, set_seed, NestDict, make_yaml_dumpable
import logging
import torch

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import re
import time as tim

class PoseEstimationAPI:
    def __init__(self, model_path, camera_info):
        """
        初始化姿态估计API
        :param config: 配置字典，包含以下参数：
            - model_path: 3D模型文件路径
            - camera_matrix: 相机内参矩阵
            - use_reconstructed: 是否使用重建模型 (bool)
            - debug_level: 调试级别 (int)
            - device: 计算设备 (e.g. 'cuda:0')
            - mesh_scale: 模型缩放系数 (可选)
        """
        set_seed(0)
        # 读取JSON文件
        # with open(camera_path, 'r') as f:
        #     data = json.load(f)

        # 从camera.json数据中提取相机内参
        cam_K = camera_info['cam_K']
        camera_matrix = np.array([
            [cam_K[0], cam_K[1], cam_K[2]],
            [cam_K[3], cam_K[4], cam_K[5]],
            [cam_K[6], cam_K[7], cam_K[8]]
        ])
        depth_scale = camera_info['depth_scale']

        # 设置相关参数
        config = {
            'model_path': model_path,
            'camera_matrix': camera_matrix,
            'mesh_scale': depth_scale,
            'device': 'cuda:0',
            'debug_level': 0
        }

        self.config = config
        self.device = config.get('device', 'cuda:0')
        self.debug_dir = config.get('debug_dir', './debug')
        os.makedirs(self.debug_dir, exist_ok=True)

        # 初始化模型
        self._init_model(config['model_path'],  config['mesh_scale'])

        # 初始化姿态估计器
        self.estimator = self._create_estimator(config)
        self.inference_times = {}
        self.results = NestDict()

    def _init_model(self, model_path, scale):
        """加载并预处理3D模型"""    # 这里需要留意好是否符合foundaionpose的模型导入流程，做了较大修改
        mesh = trimesh.load(model_path)
        mesh.vertices *= scale   # 输入应为mm，内部处理回归m单位
        self.model_vertices = mesh.vertices.copy()
        self.model_normals = mesh.vertex_normals.copy()
        self.mesh = mesh
        # 计算包围盒
        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(2, 3)

    def _create_estimator(self, config):
        """创建FoundationPose估计器"""
        torch.cuda.set_device(self.device)
        glctx = dr.RasterizeCudaContext(device=self.device)

        return FoundationPose(
            model_pts=self.model_vertices,
            model_normals=self.model_normals,
            mesh=self.mesh,
            glctx=glctx,
            debug_dir=self.debug_dir,
            debug=config.get('debug_level', 0),
            scorer=None,# 会自动加载weights
            refiner=None# 会自动加载weights
        )

    def process_frame(self, color_img, depth_map, mask, obj_id, frame_id, camera_matrix):
        """
        单帧姿态估计处理
        :param color_img: RGB图像 (H,W,3) uint8
        :param depth_map: 深度图 (H,W) float32 (单位：mm)
        :param mask: 物体掩码 (H,W) bool
        :param obj_id: 物体ID
        :param frame_id: 帧ID
        :param camera_matrix: 相机内参矩阵 (3x3)
        :return: (pose_matrix, inference_time)
        """
        start_time = tim.time()

        # 预处理输入
        color_img = np.asarray(color_img, dtype=np.uint8)
        depth_map = np.asarray(depth_map, dtype=np.float32)
        depth_map *= self.config['mesh_scale']      # 将内部处理的depth回归到m单位
        depth_map = np.asarray(depth_map, dtype=np.float16)     # 量化
        mask = np.asarray(mask, dtype=bool)

        # 执行姿态估计
        pose = self.estimator.register(
            K=camera_matrix,
            rgb=color_img,
            depth=depth_map,
            ob_mask=mask,
            ob_id=obj_id,
            index=frame_id
        )

        # odd transfor
        pose = pose @ np.linalg.inv(self.to_origin)

        # 记录时间
        inference_time = tim.time() - start_time
        self._record_statistics(obj_id, frame_id, inference_time, pose)

        return pose, inference_time

    def _record_statistics(self, obj_id, frame_id, time, pose):
        """记录统计信息"""
        if obj_id not in self.inference_times:
            self.inference_times[obj_id] = {}
        self.inference_times[obj_id][frame_id] = time

        # 保存结果到嵌套字典
        self.results[obj_id][frame_id] = pose

    def visualize_result(self, color_img, pose, camera_matrix):
        """
        生成可视化结果
        :param color_img: 原始RGB图像
        :param pose: 估计的位姿矩阵
        :param camera_matrix: 相机内参
        :return: 可视化图像 (H,W,3) uint8
        """
        center_pose = pose
        # center_pose = pose @ np.linalg.inv(self.to_origin)
        vis_img = draw_xyz_axis(
            color_img.copy(),
            ob_in_cam=center_pose,
            scale=0.1,
            K=camera_matrix,
            thickness=3,
            is_input_rgb=False
        )
        vis_img = draw_posed_3d_box(
            camera_matrix,
            img=vis_img,
            ob_in_cam=center_pose,
            bbox=self.bbox
        )
        return vis_img

    def save_results(self, output_dir):
        """保存结果到文件"""
        # 保存姿态估计结果
        with open(os.path.join(output_dir, 'pose_results.yml'), 'w') as f:
            yaml.safe_dump(make_yaml_dumpable(self.results), f)

        # 保存时间统计
        with open(os.path.join(output_dir, 'timing_stats.yml'), 'w') as f:
            yaml.safe_dump(self.inference_times, f)

    @staticmethod
    def get_default_config():
        """获取默认配置"""
        return {
            'model_path': '',
            'camera_matrix': np.eye(3),
            'use_reconstructed': False,
            'debug_level': 0,
            'device': 'cuda:0',
            'mesh_scale': 1.0
        }


if __name__ == '__main__':
    # 使用示例

    # 读取JSON文件
    with open('../Data/my_data/camera.json', 'r') as f:
        data = json.load(f)

    # 从JSON数据中提取相机矩阵
    cam_K = data['cam_K']
    camera_matrix = np.array([
        [cam_K[0], cam_K[1], cam_K[2]],
        [cam_K[3], cam_K[4], cam_K[5]],
        [cam_K[6], cam_K[7], cam_K[8]]
    ])
    depth_scale = data['depth_scale']

    config = {
        'model_path': '../Data/my_data/oneplus_modify.ply',
        'camera_matrix': camera_matrix,
        'mesh_scale': depth_scale,
        'device': 'cuda:0',
        'debug_level': 0
    }

    # 初始化API
    pose_estimator = PoseEstimationAPI(config)

    # 模拟输入数据
    for frame_idx in range(1):
        color = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640).astype(np.float16)
        mask = np.random.choice([True, False], (480, 640))

        # 处理单帧
        pose, time = pose_estimator.process_frame(
            color_img=color,
            depth_map=depth,
            mask=mask,
            obj_id=1,
            frame_id=frame_idx,
            camera_matrix=config['camera_matrix']
        )


        # 可视化
        vis_img = pose_estimator.visualize_result(color, pose, config['camera_matrix'])
        cv2.imwrite(f'frame_{frame_idx:04d}.png', vis_img[..., ::-1])

    # 保存最终结果
    pose_estimator.save_results('./output')