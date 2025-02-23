import os
import sys
import cv2
import torch
import time  # 导入时间模块
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
import shutil  # 导入shutil模块，用于删除目录

sys.path.append('..')  # 将实际路径添加到 sys.path 中

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Step 1: 初始化SAM2视频预测器和图像预测器（跳过图像检测，假设第一帧的mask已经可用）
# 使用bfloat16精度
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # 对Ampere GPU启用tfloat32（https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 记录初始化开始时间
start_time = time.time()

# 初始化视频预测器（假设你已经有了第一帧的mask）
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# 记录初始化结束时间
end_time = time.time()
print(f"初始化时间: {end_time - start_time:.2f} 秒")

# 假设你已经有第一帧的mask，这里用PNG格式加载第一帧的mask
first_frame_mask_path = '../Data/real_data/pour_water/episode_0/first_frame_mask_cup1.png'
first_frame_mask = cv2.imread(first_frame_mask_path, cv2.IMREAD_GRAYSCALE)  # 加载为灰度图像

# 确保mask是二值化的（0和255）
first_frame_mask = (first_frame_mask > 127).astype(np.uint8)

# 初始化视频预测器的状态
start_time = time.time()
inference_state = video_predictor.init_state(video_path="../Data/real_data/pour_water/episode_0/rgb")
end_time = time.time()
print(f"状态初始化时间: {end_time - start_time:.2f} 秒")

# 假设第一帧的mask已经准备好，可以直接使用
masks = first_frame_mask[None]  # 添加批次维度
scores = np.ones((1))  # 假设分数为1
logits = np.ones((1))  # 假设logits为1

# 假设你有一个物体id，这里简单假设有一个物体
object_ids = [1]

# Step 3: 为视频传播注册物体的mask
PROMPT_TYPE_FOR_VIDEO = "mask"  # 我们使用mask作为提示

# 记录开始时间
start_time = time.time()

# 将第一帧的mask添加到视频预测器中
for object_id, mask in zip(object_ids, masks):
    labels = np.ones((1), dtype=np.int32)  # 物体的标签
    _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,  # 第一帧的索引
        obj_id=object_id,
        mask=mask,
        # video_path="../Data/real_data/pour_water/episode_0/rgb"
        # ref_image_path="../Data/real_data/pour_water/episode_0/000000.png"
    )

# 记录结束时间
end_time = time.time()
print(f"物体mask注册时间: {end_time - start_time:.2f} 秒")

# Step 4: 推进视频预测器，获取每一帧的分割结果
video_segments = {}  # video_segments存储每一帧的分割结果

# 记录开始时间
start_time = time.time()

for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# 记录结束时间
end_time = time.time()
print(f"视频传播时间: {end_time - start_time:.2f} 秒")

# Step 5: 保存每一帧的mask
save_dir = "../Data/real_data/pour_water/episode_0/gdsam2_mask"

# 清空保存目录
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)  # 删除整个目录及其内容
os.makedirs(save_dir)  # 重新创建目录

# 记录开始时间
start_time = time.time()

for frame_idx, segments in video_segments.items():
    # 获取分割结果的mask
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)  # 合并所有的mask

    # 保存每一帧的mask为PNG图像
    mask_filename = os.path.join(save_dir, f"mask_{frame_idx:06d}.png")
    # 由于每个帧可能有多个mask，合并为一个大的mask（可以选择保存每个物体的单独mask）
    mask_combined = np.max(masks, axis=0)  # 将多个mask合并为一个（取最大值）
    cv2.imwrite(mask_filename, mask_combined * 255)  # 保存为PNG（0或255）

# 记录结束时间
end_time = time.time()
print(f"保存mask时间: {end_time - start_time:.2f} 秒")

# Step 6: 将注释过的帧转换为视频
# output_video_path = "./children_tracking_demo_video.mp4"
# create_video_from_images(save_dir, output_video_path)
