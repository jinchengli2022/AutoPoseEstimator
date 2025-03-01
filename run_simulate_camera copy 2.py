import os
import sys
import shutil  # 新增导入shutil模块
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('Grounded_SAM_2')  # 将实际路径添加到 sys.path 中

# import pyrealsense2 as rs
import numpy as np
import cv2
import glob
import trimesh
import logging
import torch
from PIL import Image
from omegaconf import OmegaConf

from Grounded_SAM_2.sam2.build_sam import initialize, compose, instantiate
# from hydra import compose, initialize
# from hydra.utils import instantiate

# from hydra.core.global_hydra import GlobalHydra
# GlobalHydra.instance().clear()  # 清除当前的Hydra实例，以便重新初始化

import argparse
import time as tim

from Instance_Segmentation_Model.model.utils import Detections, convert_npz_to_json
from Instance_Segmentation_Model.model.loss import Similarity
from Instance_Segmentation_Model.utils.inout import save_json_bop23, load_json
from Instance_Segmentation_Model.utils.bbox_utils import CropResizePad
from Instance_Segmentation_Model.utils.poses.pose_utils import get_obj_poses_from_template_level, \
    load_index_level_in_level2
from Instance_Segmentation_Model.segment_anything.utils.amg import rle_to_mask
from Foundationpose.FoundationposeAPI import PoseEstimationAPI as FoundationposeProcessor
from Grounded_SAM_2.TrackingAPI import VideoMaskPredictor as GDSAM2Processor

# from Grounded_SAM_2.TrackingAPI import compose, initialize, instantiate

# from hydra.core.global_hydra import GlobalHydra
# GlobalHydra.instance().clear()
# from hydra import initialize_config_module
# from hydra import initialize, compose
# from hydra.utils import instantiate


from simulate_camera import simulate_camera_capture, simulate_camera_first_capture


# 装饰器
def timed(task_name="任务"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"{task_name}开始...", end='', flush=True)
            start_time = tim.time()
            result = func(*args, **kwargs)
            elapsed_time = tim.time() - start_time
            print(f"完成! 耗时: {elapsed_time:.2f}秒")
            return result
        return wrapper
    return decorator


# 使用装饰器
@timed("初始化SAM")
def create_sam_processor(cad_path, sam_tmp_dir, segmentor_model, stability_score_thresh):
    return SamProcessor(cad_path, sam_tmp_dir, segmentor_model, stability_score_thresh)

@timed("初始化FP")
def create_fp_processor(model_path, camera_info):
    return FoundationposeProcessor(model_path, camera_info)

@timed("初始化GDSAM2")
def create_gdsam2_processor(sam2_checkpoint, model_cfg):
    return GDSAM2Processor(sam2_checkpoint, model_cfg)
    
# 封装调用方法并统计耗时
@timed("SAM切割")
def sam_process_frame(processor, rgb_path, depth_path, mask_path, cam_info):
    return processor.process_frame(rgb_path, depth_path, mask_path, cam_info)

@timed("GDSAM2切割")
def gdsam2_process_frame(processor, ref_image_path, ref_mask_path, input_image_path, output_mask_path, mask_vis_path):
    return processor.process_frames(ref_image_path, ref_mask_path, input_image_path, output_mask_path, mask_vis_path)

@timed("FP位姿估计")  # 这里使用装饰器对方法耗时进行统计
def fp_process_frame(processor, color_img, depth_map, mask, obj_id, frame_id, camera_matrix):
    return processor.process_frame(
        color_img=color_img,
        depth_map=depth_map,
        mask=mask,
        obj_id=obj_id,
        frame_id=frame_id,
        camera_matrix=camera_matrix
    )


def get_mesh_obj(mesh_dir: str):
    # mesh = trimesh.load(os.path.abspath(f'{ref_view_dir}/model/model.obj'))
    # mesh_file = self.get_gt_mesh_file(ob_id)
    mesh = trimesh.load(mesh_dir)
    mesh.vertices *= 1e-3     # 这里为什么要乘1e-3
    # 检查是否加载了纹理
    if hasattr(mesh.visual, 'material') and mesh.visual.material:
        print("纹理加载成功！")
        print("纹理材质信息:", mesh.visual.material)
    else:
        print("纹理未加载。")
    return mesh


def visualize_mask(detections, save_path="mask.png"):
    best_score = 0.0
    best_det = None
    # 遍历 detections.detections 而非 detections 对象
    for det in detections:
        if best_score < det['score']:
            best_score = det['score']
            best_det = det
    if best_det is None:
        raise ValueError("No detections found!")
    mask = rle_to_mask(best_det["segmentation"])
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(save_path)
    return best_score


def load_camera_info(file_path):
    """
    读取相机内参的JSON文件，返回格式化后的cam_info字典。
    
    :param file_path: JSON文件的路径
    :return: 包含相机内参的字典
    """
    # try:
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # print(f"cam_info:{data}")

    cam_info = {}
    # 从JSON中提取相关的cam_K和depth_scale
    cam_info['cam_K'] = data['0']['cam_K']
    cam_info['depth_scale'] = data['0']['depth_scale']

    # cam_info['cam_K'] = data['cam_K']
    # cam_info['depth_scale'] = data['depth_scale']

    # print(cam_info)

    return cam_info

    # except Exception as e:
    #     print(f"No CAMERA INFO: {e}")
    #     return None


# 保存 pose 到 txt 文件
def save_pose(pose, filename):
    """
    将位姿矩阵保存到 txt 文件。
    Args:
        pose: 变换矩阵 (numpy array, shape 4x4)
        filename: 输出的 txt 文件名
    """
    if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
        np.savetxt(filename, pose, fmt='%.8f', delimiter=' ')
        print(f"pose:{pose}")
        print(f"位姿已保存")

    else:
        raise ValueError("Pose must be a 4x4 NumPy array.")
    

class SamProcessor:
    def __init__(self, cad_path, sam_tmp_dir, segmentor_model, stability_score_thresh=0.97):
        # self.device = torch.device("cpu")   # 显存不够的苦😎
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 动态加载模型配置
        with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs"):
            cfg = compose(config_name='run_inference.yaml')
        with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs/model"):
            if segmentor_model == "sam":
                cfg.model = compose(config_name='ISM_sam.yaml')
                cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
            elif segmentor_model == "fastsam":
                cfg.model = compose(config_name='ISM_fastsam.yaml')
            else:
                raise ValueError(f"Unsupported segmentor_model: {segmentor_model}")

        self.model = instantiate(cfg.model)
        self.model.descriptor_model.model = self.model.descriptor_model.model.to(self.device)
        self.model.descriptor_model.model.device = self.device

        if hasattr(self.model.segmentor_model, "predictor"):
            self.model.segmentor_model.predictor.model = (
                self.model.segmentor_model.predictor.model.to(self.device)
            )
        else:
            self.model.segmentor_model.model.setup_model(device=self.device, verbose=True)

        # self.model.segmentor_model.model.setup_model(device=self.device, verbose=True)

        # 加载CAD和初始化模板
        self.mesh = get_mesh_obj(cad_path)     # 理论上来说应该是要乘上系数的
        self.sam_tmp_dir = sam_tmp_dir
        os.makedirs(f"{self.sam_tmp_dir}/sam6d_results", exist_ok=True)
        self._init_template_data()
        self._init_pointcloud_and_poses()

    def _init_template_data(self):
        template_dir = os.path.join(self.sam_tmp_dir, 'templates')
        num_templates = len(glob.glob(f"{template_dir}/*.npy"))
        boxes, masks, templates = [], [], []
        for idx in range(num_templates):
            image = Image.open(os.path.join(template_dir, f'rgb_{idx}.png'))
            mask = Image.open(os.path.join(template_dir, f'mask_{idx}.png'))
            boxes.append(mask.getbbox())
            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
            image = image * mask[:, :, None]
            templates.append(image)
            masks.append(mask.unsqueeze(-1))

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        proposal_processor = CropResizePad(224)
        templates = proposal_processor(images=templates, boxes=boxes).to(self.device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(self.device)

        self.model.ref_data = {
            "descriptors": self.model.descriptor_model.compute_features(templates, "x_norm_clstoken").unsqueeze(0).data,
            "appe_descriptors": self.model.descriptor_model.compute_masked_patch_feature(templates,
                                                                                         masks_cropped[:, 0, :,
                                                                                         :]).unsqueeze(0).data
        }

    # 在RealTimeProcessor类中修改_init_pointcloud_and_poses方法
    def _init_pointcloud_and_poses(self):
        # 初始化姿态数据（强制float32）
        template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
        template_poses[:, :3, 3] *= 0.4
        self.model.ref_data["poses"] = torch.tensor(template_poses, dtype=torch.float32).to(self.device)

        # 初始化点云数据（强制float32）
        model_points = self.mesh.sample(2048).astype(np.float32) / 1000.0
        self.model.ref_data["pointcloud"] = torch.tensor(model_points, dtype=torch.float32).unsqueeze(0).to(self.device)

    # def process_frame(self, rgb_path, depth_path, mask_path, cam_info):
    def process_frame(self, rgb_path, depth_path, mask_path, cam_info):
        # 加载rgb depth 相机内参
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        depth = np.array(Image.open(depth_path)).astype(np.int32)
        cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
        depth_scale = np.array(cam_info['depth_scale'])

        # batch为字典，存储经过转化后的张量，方便后续处理
        batch = {
            "depth": torch.from_numpy(depth).float().unsqueeze(0).to(self.device),  # 添加.float()
            "cam_intrinsic": torch.from_numpy(cam_K).float().unsqueeze(0).to(self.device),
            "depth_scale": torch.from_numpy(depth_scale).float().unsqueeze(0).to(self.device)
        }

        detections = self.model.segmentor_model.generate_masks(rgb)   # 调用模型，从RGB图像生成分割掩码
        detections = Detections(detections) # 将分割模型转换为Detections对象（Detections类：专门封装掩码、边界框、标签等信息）

        query_decriptors, query_appe_descriptors = self.model.descriptor_model.forward(rgb, detections)   # 提取目标的特征描述符。decriptors为几何信息，appe+descriptors为外观信息

        # 计算语义分数
        idx_selected_proposals, pred_idx_objects, semantic_score, best_template = self.model.compute_semantic_score(
            query_decriptors)

        # 权益之技：当视界无法分割时，计算几何分数会有bug
        if idx_selected_proposals.shape[0] == 1:
            print("分割失败：idx_selected_proposals.shape[0]==1")
            return
        elif idx_selected_proposals.shape[0] == 0:
            print("分割失败：idx_selected_proposals.shape[0]==0")
            return

        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # 计算外观分数
        appe_scores, ref_aux_descriptor = self.model.compute_appearance_score(best_template, pred_idx_objects,
                                                                              query_appe_descriptors)

        # 计算几何分数
        image_uv = self.model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)
        geometric_score, visible_ratio = self.model.compute_geometric_score(
            image_uv, detections, query_appe_descriptors, ref_aux_descriptor, self.model.visible_thred
        )

        # 综合分数计算
        final_score = (semantic_score + appe_scores + geometric_score * visible_ratio) / (1 + 1 + visible_ratio)
        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.zeros_like(final_score))
        detections.to_numpy()

        # 保存detections对象
        save_path = f"{self.sam_tmp_dir}/sam6d_results/detection_ism"
        detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
        detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path + ".npz"])
        save_json_bop23(save_path + ".json", detections)

        # 保存掩码图
        # mask_path = f"{data_dir}/mask/mask_{os.path.basename(rgb_path)}"
        best_score = visualize_mask(detections, save_path=mask_path)
        print(f"掩码保存至{mask_path};其分数为{best_score}", end='\t')


def capture_and_process(sam2_checkpoint, model_cfg, cad_path, sam_tmp_dir, data_dir, track_vis_dir, mask_vis_dir, track_pose_path, segmentor_model, stability_score_thresh, use_parallel_mode):
    # 虚拟相机读取的目标数据目录
    rgb_data_dir = os.path.join(data_dir, "rgb")    # RGB图像所在文件夹
    depth_data_dir = os.path.join(data_dir, "depth")  # 深度图像所在文件夹
    camera_dir = f'simulate_captured/{name}'  # 保存捕捉到的图像的文件夹目录

    # 模拟相机拍摄每帧的保存地
    ref_mask_path = "simulate_captured/ref_mask.png"
    ref_image_path = "simulate_captured/ref_image.png"

    # 读取相机内参
    camera_path = os.path.join(data_dir, 'scene_camera.json')
    # # 删除旧目录并重新创建
    for dir_path in [camera_dir, track_vis_dir, mask_vis_dir]:
    # for dir_path in [track_vis_dir, mask_vis_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)  # 删除目录及其内容
        os.makedirs(dir_path, exist_ok=True)  # 重新创建空目录
        print(f"已清空输出目录{dir_path}")


    # 相机内参
    cam_info = load_camera_info(camera_path)
    # cam_info['depth_scale'] = 0.0015

    # 初始化计数器
    frame_count = 0  
    # 初始化poses
    all_poses = []
    # 初始化SAM处理实例
    sam_processor = create_sam_processor(cad_path, sam_tmp_dir, segmentor_model, stability_score_thresh)
    # 初始化foundation处理实例  
    fp_processor = create_fp_processor(model_path=cad_path, camera_info=cam_info)
    # 初始化GDSAM2处理示例
    gdsam2_processor = create_gdsam2_processor(model_cfg=model_cfg, sam2_checkpoint=sam2_checkpoint)

    # 模拟相机拍摄首帧
    simulate_camera_first_capture(rgb_data_dir, depth_data_dir, camera_dir)
    frame_num = len(glob.glob(os.path.join(rgb_data_dir, "*.png")))
    print(f"模拟数据总共{frame_num}帧")

    while frame_count < frame_num:
        # 模拟相机持续拍摄
        simulate_camera_capture(rgb_data_dir, depth_data_dir, camera_dir, frame_count)

        # rgb,depth都是读取，mask即是保存也是读取
        rgb_path = os.path.join(camera_dir, f"rgb.png")
        depth_path = os.path.join(camera_dir, f"depth.png")
        mask_path = os.path.join(camera_dir, f"mask.png")
        mask_vis_path=os.path.join(mask_vis_dir, f'{frame_count:06d}.png')

        ref_mask_path = os.path.join(camera_dir, f"ref_mask.png")
        ref_image_path = os.path.join(camera_dir, f"ref_image.png")


        # ************用sam进行分割************ #
        print(f"\n当前处理帧{frame_count}")

        ### 只用SAM切割
        # sam_process_frame(processor=sam_processor, rgb_path=rgb_path, depth_path=depth_path, mask_path=mask_path, cam_info=cam_info)  # 将读取到的相机参数传输入sam6d的分割文件

        ### 用SAM生成初始帧
        if frame_count == 0:    # 仅对首帧进行处理
            sam_process_frame(processor=sam_processor, rgb_path=ref_image_path, depth_path=depth_path, mask_path=ref_mask_path, cam_info=cam_info)  # 将读取到的相机参数传输入sam6d的分割文件
            print("首帧切割完成")
            del sam_processor   # 工作完成，释放显存
            gdsam2_process_frame(processor=gdsam2_processor, ref_mask_path=ref_mask_path, ref_image_path=ref_image_path, input_image_path=rgb_path, output_mask_path=mask_path, mask_vis_path=mask_vis_path)
        else:
            gdsam2_process_frame(processor=gdsam2_processor, ref_mask_path=mask_path, ref_image_path=rgb_path, input_image_path=rgb_path, output_mask_path=mask_path, mask_vis_path=mask_vis_path)


        # ************用GD SAM2进行checking************ #
        # gdsam2_process_frame(processor=gdsam2_processor, ref_mask_path=ref_mask_path, ref_image_path=ref_image_path, intput_image_path=rgb_path, output_mask_path="output/ljc.png")
        # gdsam2_processor.process_frames(ref_mask_path=ref_mask_path, ref_image_path=ref_image_path, input_image_path=rgb_path, output_mask_path=mask_path)


        # ************用fp进行目标跟踪************ #
        color_bgr = cv2.imread(rgb_path)  # cv2默认读取的是bgr图像
        color_img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            frame_count += 1
            print(f"分割失败，当前帧{frame_count}跳过")
            continue  # sam分割失败，直接跳过
        
        pose, time = fp_process_frame(
            processor=fp_processor,
            color_img=color_img,
            depth_map=depth_map,
            mask=mask,
            obj_id=1,
            frame_id=frame_count,
            camera_matrix=fp_processor.config['camera_matrix']
        )

        # 可视化保存
        vis_img = fp_processor.visualize_result(color_img, pose, fp_processor.config['camera_matrix'])
        save_path=os.path.join(track_vis_dir, f'{frame_count:06d}.png')
        # save_path=os.path.join(mask_vis_dir, f'{frame_count:06d}.png')
        cv2.imwrite(save_path , vis_img[..., ::-1])
        # Pose添加
        all_poses.append(pose)
        # 及时保存poses数据(也可选择最终再保存)
        np.save(os.path.join(track_pose_path, "poses.npy"), np.array(all_poses))
        frame_count += 1

        # # pose_txt形式保存
        # save_pose(pose, f"{output_dir}/poses/pose_{frame_count}.txt")

        # print(f"检测帧{frame_count}.png已保存至{track_vis_dir}")
        # print(f"检测帧{frame_count}.png已保存到{save_path}")

        # SAM可视化结果
        # cv2.imshow("Preview", np.asanyarray(color_frame.get_data()))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    
    # MASK DEBUG
    id = ""
    time = 0
    name = "cup1"
    
    parser = argparse.ArgumentParser()
    # 输入数据(模拟相机输入数据不在此列)
    parser.add_argument("--sam2_checkpoint", default="Grounded_SAM_2/checkpoints/sam2.1_hiera_large.pt", help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg", default="sam2.1_hiera_l.yaml", help="Model configuration file")
    # parser.add_argument("--cad_path", default=f"Data/real_data/cup{cup}_mesh/cup{cup}.obj", help="Path to CAD model")  # 注意输入单位为mm，中间处理为m
    parser.add_argument("--cad_path", default=f"Data/real_data/{name}{id}_mesh/{name}{id}.obj", help="Path to CAD model")  # 注意输入单位为mm，中间处理为m
    # parser.add_argument("--sam_tmp_dir", default=f"Data/real_data/cup{cup}_mesh/cup{cup}_tmp", help="tmp directory")
    parser.add_argument("--sam_tmp_dir", default=f"Data/real_data/{name}{id}_mesh/{name}{id}_tmp", help="tmp directory")
    # 最终输出/中间输出 地址
    # parser.add_argument("--data_dir", default=f"Data/real_data/pour_water/episode_{time}", help="rgb, depth and mask")
    parser.add_argument("--data_dir", default=f"Data/real_data/pour_water/episode_{time}", help="rgb, depth and mask")
    parser.add_argument("--track_vis_dir", default=f"output/{name}{id}/{name}{id}_track_vis", help="rgb, depth and mask")
    parser.add_argument("--mask_vis_dir", default=f"output/{name}{id}/{name}{id}_mask_vis", help="rgb, depth and mask")
    parser.add_argument("--track_pose_path", default=f"output/{name}{id}", help="rgb, depth and mask")
    # 参数配置
    parser.add_argument("--segmentor_model", choices=["sam", "fastsam"], default="sam", help="Segmentor model type")
    parser.add_argument("--stability_score_thresh", type=float, default=0.97, help="SAM stability score threshold")
    args = parser.parse_args()

    # 调整日志输出等级（正常运行时）
    logging.disable(logging.CRITICAL)
    
    pose = capture_and_process(
        sam2_checkpoint=args.sam2_checkpoint,  # Add sam2_checkpoint
        model_cfg=args.model_cfg,  # Add model_cfg
        cad_path=args.cad_path,
        sam_tmp_dir=args.sam_tmp_dir,
        segmentor_model=args.segmentor_model,
        stability_score_thresh=args.stability_score_thresh,
        data_dir=args.data_dir,
        track_vis_dir=args.track_vis_dir,
        mask_vis_dir=args.mask_vis_dir,
        track_pose_path=args.track_pose_path,
        use_parallel_mode=False
    )

    # record
    # cup = 0
    # time = 0

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cad_path", default=f"Data/real_data/cup{cup}_mesh/cup{cup}.obj", help="Path to CAD model")  # 注意输入单位为mm，中间处理为m
    # parser.add_argument("--sam_tmp_dir", default=f"Data/real_data/cup{cup}_mesh/cup{cup}_tmp", help="tmp directory")
    # parser.add_argument("--data_dir", default=f"Data/real_data/pour_water/episode_{time}", help="rgb, depth and mask")
    # parser.add_argument("--track_vis_dir", default=f"Data/real_data/pour_water/episode_{time}/track_vis_cup{cup}", help="rgb, depth and mask")
    # parser.add_argument("--track_pose_dir", default=f"Data/real_data/pour_water/episode_{time}/track_pose_cup{cup}", help="rgb, depth and mask")
    # parser.add_argument("--segmentor_model", choices=["sam", "fastsam"], default="sam", help="Segmentor model type")
    # parser.add_argument("--stability_score_thresh", type=float, default=0.97, help="SAM stability score threshold")
    # args = parser.parse_args()

    
    # for time in range(0, 8):
    #     for cup in range(1, -1, -1):
    #         args.cad_path = f"Data/real_data/cup{cup}_mesh/cup{cup}.obj"
    #         args.sam_tmp_dir = f"Data/real_data/cup{cup}_mesh/cup{cup}_tmp"
    #         args.data_dir = f"Data/real_data/pour_water/episode_{time}"
    #         args.track_vis_dir = f"Data/real_data/pour_water/episode_{time}/track_vis_cup{cup}"
    #         args.track_pose_dir = f"Data/real_data/pour_water/episode_{time}/track_pose_cup{cup}"
    #         args.segmentor_model = "sam"
    #         args.stability_score_thresh = 0.97

    #         # 调整日志输出等级（正常运行时）
    #         logging.disable(logging.CRITICAL)
 
    #         capture_and_process(
    #             cad_path=args.cad_path,
    #             sam_tmp_dir=args.sam_tmp_dir,
    #             segmentor_model=args.segmentor_model,
    #             stability_score_thresh=args.stability_score_thresh,
    #             data_dir=args.data_dir,
    #             track_vis_dir=args.track_vis_dir,
    #             track_pose_dir=args.track_pose_dir,
    #             use_parallel_mode=False
    #         )