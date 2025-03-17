import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))     # 添加当前路径到sys.path可检索的模块中区
# sys.path.append(os.path.dirname("Qwen_VL"))     # 添加当前路径到sys.path可检索的模块中区
from Qwen_VL.QwenAPI import *

if __name__ == "__main__":
    print("start")
    local_model_path = "/home/ljc/.cache/huggingface/hub/models--Qwen--Qwen-VL-Chat-Int4/snapshots/cbe5f4e5a742f3019d084e0d53861f72b4e60350"
    pipeline = QwenChat(local_model_path)

    # 根据第一轮图片和文字指令启动流程，再根据第二轮图片完成整个流程
    first_image = 'Qwen_VL/assets/vase_ref_25.png'
    first_text = '不要介绍背景，用一句话描述物体。'
    second_image = 'Qwen_VL/assets/test_0.png'
    box = pipeline.run_pipeline(first_image, first_text, second_image, output_path='0.jpg')
    print(box)