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
from PIL import Image
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()  # 清除当前的Hydra实例，以便重新初始化
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class VideoMaskPredictor:
    def __init__(self, sam2_checkpoint, model_cfg, grounding_model_id="IDEA-Research/grounding-dino-tiny", device=None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.video_predictor = None
        self.inference_state = None
        self.frame_count = 0

        # Initialize the video predictor and first frame mask
        initialize(config_path="sam2/configs/sam2.1/")  # 请确保配置路径正确

        self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint)
        # 初始化 SAM2 图像分割模型
        sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint)
        # GlobalHydra.instance().clear()  # 清除现有的 Hydra 实例
        self.inference_state = self.video_predictor.init_state()
        self.image_predictor = SAM2ImagePredictor(sam2_model)
        
        # 初始化 Grounding DINO 模型
        self.processor = AutoProcessor.from_pretrained(grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(self.device)

    def predict_mask_with_box(self, rgb_path, mask_path, box):
        """
        输入一张 RGB 图像，返回对应的单通道二值掩码（uint8）
        :param rgb_image: numpy 数组格式的 RGB 图像
        :return: 单通道二值掩码，前景像素值为 255，背景为 0
        """
        bgr_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # 将 numpy 图像转换为 PIL 格式
        image_pil = Image.fromarray(rgb_image)

        # 使用 SAM2 图像分割器生成掩码
        self.image_predictor.set_image(np.array(image_pil.convert("RGB")))
        masks, scores_pred, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )
        # 处理返回的掩码（若返回多个，则取第一个）
        if masks.ndim >= 3:
            mask = masks[0]
        else:
            mask = masks

        # 将概率/浮点型掩码二值化，并转换为 uint8 格式（0/255）
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        cv2.imwrite(mask_path, mask_binary)

    def predict_mask(self, rgb_path, mask_path, text_prompt):
        """
        输入一张 RGB 图像，返回对应的单通道二值掩码（uint8）
        :param rgb_image: numpy 数组格式的 RGB 图像
        :return: 单通道二值掩码，前景像素值为 255，背景为 0
        """
        bgr_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # 将 numpy 图像转换为 PIL 格式
        image_pil = Image.fromarray(rgb_image)
        
        # 通过 Grounding DINO 根据文本提示获取检测框
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.3,
            target_sizes=[image_pil.size[::-1]]
        )
        # 若未检测到目标，则返回全零掩码
        if not results or len(results[0]["boxes"]) == 0:
            return np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
        
        # 从检测结果中选择得分最高的检测框
        boxes = results[0]["boxes"].cpu().numpy()
        if "scores" in results[0]:
            scores = results[0]["scores"].cpu().numpy()
            idx = np.argmax(scores)
            box = boxes[idx][None, :]  # 调整为 (1, 4)
        else:
            box = boxes[0][None, :]
        
        # 使用 SAM2 图像分割器生成掩码
        self.image_predictor.set_image(np.array(image_pil.convert("RGB")))
        masks, scores_pred, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )
        # 处理返回的掩码（若返回多个，则取第一个）
        if masks.ndim >= 3:
            mask = masks[0]
        else:
            mask = masks
        
        # 将概率/浮点型掩码二值化，并转换为 uint8 格式（0/255）
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        cv2.imwrite(mask_path, mask_binary)
    

    def process_frames(self, ref_image_path, ref_mask_path, input_image_path, output_mask_path, mask_vis_path):
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
            # cv2.imwrite(mask_vis_path, mask_result)  # Save as PNG (0 or 255)


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
