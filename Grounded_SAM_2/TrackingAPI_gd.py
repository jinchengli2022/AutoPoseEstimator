import os
import cv2
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundedSAM2MaskGenerator:
    def __init__(self, text, sam2_checkpoint, model_cfg, grounding_model_id="IDEA-Research/grounding-dino-tiny", device=None):
        """
        初始化时输入文本提示，并加载 SAM2 与 Grounding DINO 模型
        :param text: 分割对象的文本描述（例如 "car."），注意需为小写且以句号结尾
        :param sam2_checkpoint: SAM2 模型的权重路径
        :param model_cfg: SAM2 模型的配置文件路径
        :param grounding_model_id: Grounding DINO 模型ID，默认使用 "IDEA-Research/grounding-dino-tiny"
        :param device: 设备，若为 None 则自动选择 "cuda"（若可用）或 "cpu"
        """

        
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化 SAM2 图像分割模型
        sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_model)
        
        # 初始化 Grounding DINO 模型
        self.processor = AutoProcessor.from_pretrained(grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(self.device)
        
    def predict_mask(self, rgb_path):
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
        inputs = self.processor(images=image_pil, text=self.text, return_tensors="pt").to(self.device)
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
        return mask_binary

# 示例用法
if __name__ == "__main__":
    # 设置参数
    text_prompt = "vase."
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # 初始化生成器，传入文本提示
    mask_generator = GroundedSAM2MaskGenerator(text=text_prompt, sam2_checkpoint=sam2_checkpoint, model_cfg=model_cfg)
    
    # 读取 RGB 图像（注意：cv2.imread 读取的是 BGR 格式，这里需要转换为 RGB）
    name = "vase"
    rgb_path = f"test_data/{name}.png"
    
    
    # 生成单通道掩码
    mask = mask_generator.predict_mask(rgb_path=rgb_path)
    
    # 保存掩码图像
    cv2.imwrite(f"test_data/{name}_mask.png", mask)
    print("单通道掩码已保存到 output_mask.png")
