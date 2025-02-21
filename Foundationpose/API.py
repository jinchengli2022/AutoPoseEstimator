# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import logging
import re
import time
import trimesh
import cv2
import imageio
import numpy as np
import torch
import warp as wp
from albumentations import downscale
from Utils import *

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
from datareader import *
from estimater import *
import yaml


# 获取掩码
def get_mask(reader, i_frame, ob_id, detect_type):
    if detect_type == 'box':
        mask = reader.get_mask(i_frame, ob_id)
        H, W = mask.shape[:2]
        vs, us = np.where(mask > 0)
        umin = us.min()
        umax = us.max()
        vmin = vs.min()
        vmax = vs.max()
        valid = np.zeros((H, W), dtype=bool)
        valid[vmin:vmax, umin:umax] = 1
    elif detect_type == 'mask':
        mask = reader.get_mask(i_frame, ob_id)
        if mask is None:
            return None
        valid = mask > 0
    elif detect_type == 'detected':
        mask = cv2.imread(reader.color_files[i_frame].replace('rgb', 'mask_cosypose'), -1)
        valid = mask == ob_id
    else:
        raise RuntimeError
    return valid


class PoseEstimator:
    def __init__(self, args):
        self.args = args
        self.detect_type = 'mask'  # 固定检测类型
        self.inference_times = {}

        # 初始化CUDA上下文
        wp.force_load(device='cuda')

        # 初始化临时读取器获取对象ID
        tmp_reader = LinemodReader(args["linemod_dir"], split=None)
        match = re.search(r'\d+', args["linemod_dir"])
        self.ob_id = int(match.group()) if match else None

        # 初始化主读取器
        self.reader = LinemodReader(args["linemod_dir"], split=None)
        self.video_id = self.reader.get_video_id()

        # 初始化网格模型
        self._init_mesh(tmp_reader)

        # 初始化姿态估计器
        self.estimator = self._init_estimator(tmp_reader)

        # 创建可视化目录
        os.makedirs(f'{args["linemod_dir"]}/track_vis', exist_ok=True)

    def _init_mesh(self, tmp_reader):
        """初始化3D模型"""
        if self.args["use_reconstructed_mesh"]:
            self.mesh = tmp_reader.get_reconstructed_mesh(
                ref_view_dir=self.args["ref_view_dir"])
        else:
            self.mesh = tmp_reader.get_gt_mesh(self.args["mesh_dir"])

        # 计算OBB边界框
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    def _init_estimator(self, tmp_reader):
        """初始化姿态估计器"""
        glctx = dr.RasterizeCudaContext()
        dummy_mesh = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()

        estimator = FoundationPose(
            model_pts=dummy_mesh.vertices.copy(),
            model_normals=dummy_mesh.vertex_normals.copy(),
            mesh=dummy_mesh,
            glctx=glctx,
            debug_dir=self.args["debug_dir"],
            debug=self.args["debug"]
        )

        # 重置为实际使用的模型
        estimator.reset_object(
            model_pts=self.mesh.vertices.copy(),
            model_normals=self.mesh.vertex_normals.copy(),
            mesh=self.mesh
        )
        return estimator

    def _process_frame(self, frame_idx):
        """处理单帧"""
        start_time = time.time()

        # 准备数据
        color = self.reader.get_color(frame_idx)
        depth = self.reader.get_depth(frame_idx)
        ob_mask = get_mask(self.reader, frame_idx, self.ob_id, self.detect_type)

        # 设置真实姿态（用于调试）
        self.estimator.gt_pose = self.reader.get_gt_pose(frame_idx, self.ob_id)

        # **********运行位姿估计器********** #
        pose = self.estimator.register(
            K=self.reader.K,
            rgb=color,
            depth=depth,
            ob_mask=ob_mask,
            ob_id=self.ob_id
        )

        # 记录推理时间
        inference_time = time.time() - start_time
        self._record_inference_time(frame_idx, inference_time)

        # 可视化结果
        self._visualize_result(frame_idx, color, pose)

        return pose

    def _record_inference_time(self, frame_idx, time_cost):
        """记录推理时间"""
        if self.ob_id not in self.inference_times:
            self.inference_times[self.ob_id] = {}
        self.inference_times[self.ob_id][frame_idx] = time_cost

    def _visualize_result(self, frame_idx, color_img, pose):
        """结果可视化"""
        center_pose = pose @ np.linalg.inv(self.to_origin)

        # 绘制3D边界框
        vis_img = draw_posed_3d_box(
            self.reader.K,
            img=color_img,
            ob_in_cam=center_pose,
            bbox=self.bbox
        )

        # 绘制坐标轴
        vis_img = draw_xyz_axis(
            color_img,
            ob_in_cam=center_pose,
            scale=0.1,
            K=self.reader.K,
            thickness=3,
            transparency=0,
            is_input_rgb=True
        )

        # 保存可视化结果
        output_path = f'{self.args["linemod_dir"]}/track_vis/{self.reader.id_strs[frame_idx]}.png'
        imageio.imwrite(output_path, vis_img)

    def run_estimation(self):
        """执行姿态估计流程"""
        results = NestDict()
        num_frames = min(len(self.reader.color_files), self.args["num"])

        for frame_idx in range(num_frames):
            logging.info(f"Processing frame {frame_idx}/{num_frames}")
            print(f"\n### Processing frame {frame_idx}/{num_frames}")

            try:
                pose = self._process_frame(frame_idx)
                # 保存结果
                results[self.video_id][self.reader.id_strs[frame_idx]][self.ob_id] = pose
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
                results[self.video_id][self.reader.id_strs[frame_idx]][self.ob_id] = np.eye(4)

        # 保存结果文件
        self._save_results(results)

    def _save_results(self, results):
        """保存结果到文件"""
        # 保存推理时间
        with open(f'{self.args["debug_dir"]}/linemod_inference_times.yml', 'w') as f:
            yaml.safe_dump(self.inference_times, f)

        # 保存姿态估计结果
        with open(f'{self.args["debug_dir"]}/linemod_res.yml', 'w') as f:
            yaml.safe_dump(make_yaml_dumpable(results), f)


def main():
    # 配置参数
    args = {
        "linemod_dir": "/home/ljc/Git/FoundationPose/my_linemod_test/0000002",
        "use_reconstructed_mesh": False,
        "ref_view_dir": "/home/ljc/Git/FoundationPose/my_linemod/cola_solve/ob_0000001",
        "debug": False,
        "debug_dir": "/home/ljc/Git/FoundationPose/my_linemod_test/debug",
        "num": 1,
        "mesh_dir": "/home/ljc/Git/FoundationPose/my_linemod_test/lm_models/models/oneplus_modify.ply",
    }

    # 设置随机种子
    set_seed(0)

    # 执行姿态估计
    estimator = PoseEstimator(args)
    estimator.run_estimation()


if __name__ == '__main__':
    main()
