# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from albumentations import downscale

from Utils import *
import json, uuid, joblib, os, sys
import scipy.spatial as spatial
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
import itertools
from datareader import *
from estimater import *

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import yaml
import re


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


# 姿态估计
def run_pose_estimation_worker(reader, i_frames, est: FoundationPose = None, debug=0, ob_id=None, device='cuda:0'):
    # 指定GPU
    torch.cuda.set_device(device)
    est.to_device(device)
    est.glctx = dr.RasterizeCudaContext(device=device)

    # 初始化一个字典
    result = NestDict()

    for i, i_frame in enumerate(i_frames):
        logging.info(f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")
        print("\n### ", f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")

        video_id = reader.get_video_id()
        color = reader.get_color(i_frame)
        depth = reader.get_depth(i_frame)
        print(f"Depth before:\n{depth}")

        id_str = reader.id_strs[i_frame]
        H, W = color.shape[:2]

        debug_dir = est.debug_dir

        ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)

        # Debug
        if ob_mask is None:
            print(f"[DEBUG] ob_mask is None for frame {i_frame}, ob_id {ob_id}")
            logging.info("ob_mask not found, skip")
            result[video_id][id_str][ob_id] = np.eye(4)
            return result

        valid_pixel_count = np.sum(ob_mask)

        print(f"[DEBUG] Frame {i_frame}, ob_id {ob_id}, valid pixels: {valid_pixel_count}")
        if valid_pixel_count < 10:  # Example threshold
            print(f"[DEBUG] Valid pixels too small for frame {i_frame}, ob_id {ob_id}, skipping...")
        # if ob_mask is None:
        #     logging.info("ob_mask not found, skip")
        #     result[video_id][id_str][ob_id] = np.eye(4)
        #     return result

        est.gt_pose = reader.get_gt_pose(i_frame, ob_id)    # 读取GT位姿

        # **********运行位姿估计器********** #
        pose = est.register(
            K=reader.K,
            rgb=color,
            depth=depth,
            ob_mask=ob_mask,
            ob_id=ob_id
        )   # 最关键的，获取位姿数据
        logging.info(f"pose:\n{pose}")

        if debug >= 3:
            m = est.mesh_ori.copy()
            tmp = m.copy()
            tmp.apply_transform(pose)
            tmp.export(f'{debug_dir}/model_tf.obj')

        result[video_id][id_str][ob_id] = pose

    return result, pose


# 主函数
def run_pose_estimation():
    # 初始化
    wp.force_load(device='cuda')
    reader_tmp = LinemodReader(opt.linemod_dir, split=None)
    print("## opt.linemod_dir:", opt.linemod_dir)

    # 提取相关参数
    debug = opt.debug
    use_reconstructed_mesh = opt.use_reconstructed_mesh
    debug_dir = opt.debug_dir
    use_num = opt.num

    # 姿态估计器的初始化
    res = NestDict()
    glctx = dr.RasterizeCudaContext()   # CUDA加速
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()  # 创建了一个单位立方体作为空的3D模型，后续会替代
    # 建立FoundationPose对象
    est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(),
                         symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir,
                         debug=debug)   # 在FoundationPose内部调用了scorer和refiner，而不是所看到的未赋值

    # 提取物体id od_id
    match = re.search(r'\d+$', opt.linemod_dir)
    if match:
        last_number = match.group()
        ob_id = int(last_number)
    else:
        print("No digits found at the end of the string")

    # 推理时间记录字典
    inference_times = {}  # 用于记录各帧的推理时间

    # 加载 3D 模型
    if ob_id:
        if use_reconstructed_mesh:# 默认选择使用重建模型
            print("## ob_id:", ob_id)
            print("## opt.linemod_dir:", opt.linemod_dir)
            print("## opt.ref_view_dir:", opt.ref_view_dir)
            mesh = reader_tmp.get_reconstructed_mesh(ref_view_dir=opt.ref_view_dir) # 加载重建模型
        else:
            mesh = reader_tmp.get_gt_mesh(opt.mesh_dir)    # 加载预设路径的模型
        # symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]  # !!!!!!!!!!!!!!!!

        args = []

        reader = LinemodReader(opt.linemod_dir, split=None) # 创建一个LinemodReader对象，方便读取Linemod类型数据集
        video_id = reader.get_video_id()
        # est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=mesh)  # raw

        # 重值3D模型
        est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(),
                         mesh=mesh)  # !!!!!!!!!!!!!!!!将读取到的模型更新

        print("### len(reader.color_files):", len(reader.color_files))
        # 开始遍历帧
        for i in range(len(reader.color_files)):
            args.append((reader, [i], est, debug, ob_id, "cuda:0"))     # 读取所有帧的数据，创建args列表，表示任务

        # 可视化设置，创建OBB盒
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        os.makedirs(f'{opt.linemod_dir}/track_vis', exist_ok=True)

        # 执行姿态估计
        outs = []
        for i, arg in enumerate(args[:use_num]):
            print("### num:", i)

            # 记录推理开始时间
            start_time = time.time()

            # ***********执行推理*********** #
            out, pose = run_pose_estimation_worker(*arg)  # 获取目标位姿

            # 记录推理结束时间
            end_time = time.time()

            # 计算推理时间
            inference_time = end_time - start_time
            print(f"Frame {i}: Inference Time = {inference_time:.6f} seconds")

            # 将推理时间保存到字典中
            if ob_id not in inference_times:
                inference_times[ob_id] = {}
            inference_times[ob_id][i] = inference_time

            # 可视化并保存图片
            center_pose = pose @ np.linalg.inv(to_origin)
            img_color = reader.get_color(i)
            vis = draw_posed_3d_box(reader.K, img=img_color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(img_color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0,
                                is_input_rgb=True)
            imageio.imwrite(f'{opt.linemod_dir}/track_vis/{reader.id_strs[i]}.png', vis)

            # 保存结果
            outs.append(out)

        for out in outs:
            for video_id in out:
                for id_str in out[video_id]:
                    for ob_id in out[video_id][id_str]:
                        res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]

    # 保存推理时间到YAML文件
    with open(f'/home/ljc/Git/FoundationPose/debug/linemod_inference_times.yml', 'w') as time_file:
        yaml.safe_dump(inference_times, time_file)
        print("Save linemod_inference_times.yml OK !!!")

    # 保存姿态估计结果到YAML文件
    with open(f'/home/ljc/Git/FoundationPose/linemod_res.yml', 'w') as ff:
        yaml.safe_dump(make_yaml_dumpable(res), ff)
        print("Save linemod_res.yml OK !!!")


if __name__ == '__main__':
    # 初始化和参数设定
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--linemod_dir', type=str,
                        default="/home/ljc/Git/FoundationPose/my_linemod_test/0000002",
                        help="linemod root dir")  # lm_test_all  lm_test
    parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
    parser.add_argument('--ref_view_dir', type=str,
                        default="/home/ljc/Git/FoundationPose/my_linemod/cola_solve/ob_0000001")
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--debug_dir', type=str,
                        default=f'~/Git/FoundationPose/my_linemod_test/debug')  # lm_test_all  lm_test
    parser.add_argument('--num', type=int,
                        default=1)  # lm_test_all  lm_test
    parser.add_argument('--mesh_dir', type=str,
                        default=f'/home/ljc/Git/FoundationPose/my_linemod_test/lm_models/models/oneplus_modify.ply')  # lm_test_all  lm_test
    opt = parser.parse_args()
    set_seed(0)

    detect_type = 'mask'  # mask / box / detected
    run_pose_estimation()
