import os
import sys
import cv2
import torch
import numpy as np
import shutil

# from hydra.core.global_hydra import GlobalHydra
# GlobalHydra.instance().clear()
# from hydra import initialize_config_module
# from hydra import initialize, compose
# from hydra.utils import instantiate

sys.path.append('..')  # 将实际路径添加到 sys.path 中
# AutoPoseEstimator

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()  # 清除当前的Hydra实例，以便重新初始化
from sam2.build_sam import build_sam2_video_predictor

class VideoMaskPredictor:
    def __init__(self, sam2_checkpoint, model_cfg):
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.video_predictor = None
        self.inference_state = None
        self.frame_count = 0

        # Initialize the video predictor and first frame mask
        self._initialize()

    def _initialize(self):
        # 初始化
        initialize(config_path="sam2/configs/sam2.1/")  # 请确保配置路径正确

        self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint)
        # GlobalHydra.instance().clear()  # 清除现有的 Hydra 实例
        self.inference_state = self.video_predictor.init_state()

        # 加载掩代
        # self.ref_mask = cv2.imread(self.ref_mask_path, cv2.IMREAD_GRAYSCALE)
        # self.ref_mask = (self.ref_mask > 127).astype(np.uint8)

        # self.frame_count = len(os.listdir(self.video_frames_dir))

        # 清楚保存目录
        # if os.path.exists(self.save_dir):
        #     shutil.rmtree(self.save_dir)
        # os.makedirs(self.save_dir)

    def process_frames(self, ref_image_path, ref_mask_path, input_image_path, output_mask_path):
        # 加载参考掩码
        self.ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
        self.ref_mask = (self.ref_mask > 127).astype(np.uint8)
        mask = self.ref_mask

        # Register the current frame's mask
        object_id = 1  # Assuming the object ID is 1
        # labels = np.ones((1), dtype=np.int32)  # Object labels

        _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=0,
            obj_id=object_id,
            mask=mask,
            ref_image_path=ref_image_path,  # 加载参考image
            video_path=input_image_path
        )

        # 运行推演
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # 保存mask
        for _, segments in video_segments.items():
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)

            mask_combined = np.max(masks, axis=0)  # Combine masks by taking the maximum value
            mask_result = mask_combined * 255
            cv2.imwrite(output_mask_path, mask_result)  # Save as PNG (0 or 255)

        # Update the mask for the next frame
        # self.mask_last = mask_result

        print(f"掩码保存到{output_mask_path}.", end="   ")


def initialize_predictor(sam2_checkpoint, model_cfg):
    predictor = VideoMaskPredictor(sam2_checkpoint=sam2_checkpoint, model_cfg=model_cfg)
    return predictor
    
# Example usage:
if __name__ == "__main__":
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # ref_mask_path = "../Data/real_data/pour_water/episode_0/first_frame_mask_cup1.png"
    # ref_image_path = "../Data/real_data/pour_water/episode_0/000000.png"
    # input_image_path = "../Data/real_data/pour_water/episode_0/gdsam2_rgb/000002.png"
    # output_mask_path = "../Data/real_data/pour_water/episode_0/gdsam2_mask/ljc.png"
    
    ref_mask_path = "../simulate_captured/ref_mask.png"
    ref_image_path = "../simulate_captured/ref_image.png"
    input_image_path = "../simulate_captured/rgb.png"
    output_mask_path = "../output/mask.png"
    


    # Initialize and process the video frames
    predictor = VideoMaskPredictor(
        sam2_checkpoint=sam2_checkpoint, model_cfg=model_cfg
    )
    predictor.process_frames(ref_mask_path=ref_mask_path, ref_image_path=ref_image_path, input_image_path=input_image_path, output_mask_path=output_mask_path)
