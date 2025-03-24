import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import numpy as np
from PIL import Image

class QwenChat:
    def __init__(self, local_model_path, seed=1234):
        """
        初始化模型和分词器，同时设置随机种子
        :param local_model_path: 本地模型路径
        :param seed: 随机种子
        """
        torch.manual_seed(seed)
        # 加载分词器（注意：分词器默认行为已更改为默认关闭特殊token攻击防护）
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
        # 加载模型，并自动分配设备
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            device_map="auto",
            trust_remote_code=True
        ).eval()

    # ### 单张图片的pipeline
    # def run_pipeline(self, first_image: str, second_image: str, output_path: str = '0.jpg'):
    #     """
    #     执行完整的对话流程：
    #       1. 第一轮对话：根据第一张图片和文字指令获取模型响应
    #       2. 第二轮对话：使用第二张图片以及包含第一轮响应的文字指令，获取最终响应
    #       3. 根据最终响应绘制边框，并保存图片
    #
    #     :param first_image: 第一轮对话使用的图片路径或URL
    #     :param first_text: 第一轮对话使用的文字指令
    #     :param second_image: 第二轮对话使用的图片路径或URL
    #     :param output_path: 保存绘制边框后图片的路径，默认保存为'0.jpg'
    #     :return: 第一轮和第二轮的响应文本
    #     """
    #     # 第一轮对话
    #     first_text = '不要介绍背景，用无标点的一句话描述物体'
    #
    #     query = self.tokenizer.from_list_format([
    #         {'image': first_image},
    #         {'text': first_text},
    #     ])
    #     response, history = self.model.chat(self.tokenizer, query=query, history=None)
    #     print("第一轮响应：", response)
    #
    #     # 第二轮对话，文本中嵌入第一轮的响应
    #     second_text = f'根据以下描述及历史图片:{response}。框出物体'
    #     query = self.tokenizer.from_list_format([
    #         {'image': second_image},
    #         {'text': second_text},
    #     ])
    #     response, history = self.model.chat(self.tokenizer, query=query, history=history)
    #     print("第二轮响应：", response)
    #
    #     # 提取box框
    #     pattern = r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>"
    #     match = re.search(pattern, response)
    #     if match:
    #         # 提取匹配的四个数字，并转换为 float 类型
    #         x1, y1, x2, y2 = map(float, match.groups())     # 注意这里的边框是经过1000归一化的，即
    #
    #         # 打开图片并读取宽度和高度
    #         with Image.open(second_image) as img:
    #             width, height = img.size
    #
    #         # 临时检测
    #         x1 = x1 / 1000 * width
    #         x2 = x2 / 1000 * width
    #         y1 = y1 / 1000 * height
    #         y2 = y2 / 1000 * height
    #
    #         # 生成形状为 (1,4) 的 ndarray，类型为 float32
    #         box_array = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    #         print("提取的 box 坐标为:", box_array)
    #     else:
    #         box_array = np.array([[0, 0, 0, 0]], dtype=np.float32)
    #         print("未匹配到 box 坐标。")
    #
    #     # 绘制边框并保存图片
    #     image = self.tokenizer.draw_bbox_on_latest_picture(response, history)
    #     if image:
    #         image.save(output_path)
    #         print(f"边框绘制成功，图片已保存至 {output_path}")
    #     else:
    #         print("未检测到边框。")
    #
    #     return box_array

    def run_pipeline(self, obj_view_dir: str, scene_img: str, output_path: str = '0.jpg'):
            """
            执行完整的对话流程：
              1. 第一轮对话：根据第一张图片和文字指令获取模型响应
              2. 第二轮对话：使用第二张图片以及包含第一轮响应的文字指令，获取最终响应
              3. 根据最终响应绘制边框，并保存图片

            :param first_image: 第一轮对话使用的图片路径或URL
            :param first_text: 第一轮对话使用的文字指令
            :param second_image: 第二轮对话使用的图片路径或URL
            :param output_path: 保存绘制边框后图片的路径，默认保存为'0.jpg'
            :return: 第一轮和第二轮的响应文本
            """
            # 第一轮对话
            # < img > {image_path_0} < / img >；Picture 1： < img > {image_path_1} < / img >。
            top_view = os.path.join(obj_view_dir, 'rgb_0.png')
            front_view = os.path.join(obj_view_dir, 'rgb_26.png')
            bottom_view = os.path.join(obj_view_dir, 'rgb_41.png')
            # query = f'根据以下视图图，做出描述：视角图1:<img>{view_1}</img>；视角图2:<img>{view_2}</img>,视角图3:<img>{view_3}</img>'
            # query = f'根据以下视图图，做出描述：视角图1:<img>{view_1}</img>；视角图2:<img>{view_2}</img>,视角图3:<img>{view_3}</img>。并框出以下场景图<img>{scene_img}</img>中最符合以下视角图的物体'
            # query = f'框出以下场景图<img>{scene_img}</img>中最符合视角图1:<img>{view_1}</img>；视角图2:<img>{view_2}</img>,视角图3:<img>{view_3}</img>的物体'
            # query = f"<img>{top_view}</img><img>{front_view}</img><img>{bottom_view}</img><img>{scene_img}</img>Draw a bounding box around the object in the following scene image that best the first few pictures"

            # query = f'不要介绍背景，用精准且无标点的一句话描述物体，以下是此物体在不同视角下的图片：视角图1:<img>{view_1}</img>；视角图2:<img>{view_2}</img>,视角图3:<img>{view_3}</img>'
            query = f'<img>{top_view}</img>;front view:<img>{front_view}</img>;bottom view:<img>{bottom_view}</img>,use only one sentence without any background information to describe the object with three views'
            response, history = self.model.chat(self.tokenizer, query=query, history=None)
            print("第一轮响应：", response)

            # 第二轮对话，文本中嵌入第一轮的响应
            # query = f"根据以下描述及历史图片:{response}。在<img>{scene_img}</img>中框出物体"
            query = self.tokenizer.from_list_format([
                {'image': scene_img},
                {'text': f"Based on the following description:{response} and historical pictures:,frame the object that best fits the description"},
            ])
            response, history = self.model.chat(self.tokenizer, query=query, history=history)
            print("第二轮响应：", response)

            # 提取box框
            pattern = r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>"
            match = re.search(pattern, response)
            if match:
                # 提取匹配的四个数字，并转换为 float 类型
                x1, y1, x2, y2 = map(float, match.groups())     # 注意这里的边框是经过1000归一化的，即

                # 打开图片并读取宽度和高度
                with Image.open(scene_img) as img:
                    width, height = img.size

                # 临时检测
                x1 = x1 / 1000 * width
                x2 = x2 / 1000 * width
                y1 = y1 / 1000 * height
                y2 = y2 / 1000 * height

                # 生成形状为 (1,4) 的 ndarray，类型为 float32
                box_array = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                print("提取的 box 坐标为:", box_array)
            else:
                box_array = np.array([[0, 0, 0, 0]], dtype=np.float32)
                print("未匹配到 box 坐标。")

            # 绘制边框并保存图片
            image = self.tokenizer.draw_bbox_on_latest_picture(response, history)
            if image:
                image.save(output_path)
                print(f"边框绘制成功，图片已保存至 {output_path}")
            else:
                print("未检测到边框。")

            return box_array

# 示例用法
if __name__ == '__main__':
    local_model_path = "/home/ljc/.cache/huggingface/hub/models--Qwen--Qwen-VL-Chat-Int4/snapshots/cbe5f4e5a742f3019d084e0d53861f72b4e60350"
    pipeline = QwenChat(local_model_path)

    # 根据第一轮图片和文字指令启动流程，再根据第二轮图片完成整个流程
    obj_name = "flower"
    obj_view_dir = f"Data/real_data/{obj_name}_mesh/{obj_name}_tmp/templates"
    rgb_path = "Data/real_data/ism_test/episode_0/rgb/000000.png"
    pipeline.run_pipeline(obj_view_dir, rgb_path, f'0.png')
