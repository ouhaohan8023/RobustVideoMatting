# 使用SAM切割人脸
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import os
import torch
from tqdm import tqdm
# 从 frames 文件夹中读取所有文件
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
# 定义输入和输出的根目录
root_input_dir = "/media/ext_disk/ouhaohan/video/frames"
root_output_dir = "/media/ext_disk/ouhaohan/video/sams"

if __name__ == '__main__':
    sam = sam_model_registry["vit_h"](checkpoint="/media/ext_disk/ouhaohan/checkpoint/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 遍历 root_input_dir 下的所有 batch 文件夹
    for batch_folder in os.listdir(root_input_dir):
        batch_input_dir = os.path.join(root_input_dir, batch_folder)
        batch_output_dir = os.path.join(root_output_dir, batch_folder)

        # 如果 batch_output_dir 不存在，创建它
        os.makedirs(batch_output_dir, exist_ok=True)

        # 遍历 batch_input_dir 文件夹中的所有子文件夹
        for folder in os.listdir(batch_input_dir):
            folder_path = os.path.join(batch_input_dir, folder)
            # 检查是否为文件夹
            if os.path.isdir(folder_path):
                # 创建对应的输出文件夹
                output_folder_path = os.path.join(batch_output_dir, folder)
                os.makedirs(output_folder_path, exist_ok=True)
                # 遍历 folder_path 文件夹中的所有文件
                # 获取 folder_path 下所有满足条件的文件的数量
                total_files = len([f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")])
                # 创建一个进度条
                progress_bar = tqdm(total=total_files, desc="Processing images", ncols=70)
                for filename in os.listdir(folder_path):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        # 保存处理后的图像到 batch_output_dir 文件夹中，文件名保持不变
                        output_filename = os.path.join(output_folder_path, filename)
                        # 检查输出文件是否已经存在
                        if os.path.exists(output_filename):
                            # print(f"Output file {output_filename} already exists, skipping...")
                            progress_bar.update(1)
                            continue
                        # 读取图像
                        image_bgr = cv2.imread(os.path.join(folder_path, filename))
                        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                        predictor.set_image(image_rgb)
                        # 获取图像的宽度和高度
                        height, width = image_rgb.shape[:2]

                        # 计算图像的中心点坐标
                        center_x = width // 2
                        center_y = height // 2

                        # 将中心点坐标作为 input_point
                        input_point = np.array([[center_x, center_y]])
                        input_label = np.array([1])
                        # 进行 SAM 分割
                        masks, scores, logits = predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            multimask_output=True
                        )

                        # 初始化最佳分数和最佳面具
                        best_score = -1
                        best_face_mask = None
                        for i, (mask, score) in enumerate(zip(masks, scores)):
                            # 如果当前分数比之前的最高分数高，则更新最佳面具
                            if score > best_score:
                                best_score = score
                                best_face_mask = mask

                        # 将图片转换为全黑
                        masked_image = np.zeros_like(image_bgr)

                        # 将 best_face_mask 区域的像素值从原始图像复制到 masked_image
                        masked_image[best_face_mask != 0] = image_rgb[best_face_mask != 0]


                        cv2.imwrite(output_filename, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
                        progress_bar.update(1)
                # 关闭进度条
                progress_bar.close()
                # 打印已处理的文件名
                print(f"Finished processing {folder_path}")
