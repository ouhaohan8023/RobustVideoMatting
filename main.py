import torch
from model import MattingNetwork
from inference import convert_video
import os
import time
import cv2
from tqdm import tqdm

input_dirs = ['/home/ouhaohan/Codes/RobustVideoMatting/media/video/batch1', '/home/ouhaohan/Codes/RobustVideoMatting/media/video/batch3', '/home/ouhaohan/Codes/RobustVideoMatting/media/video/batch2']
output_dirs = ['/home/ouhaohan/Codes/RobustVideoMatting/media/output/batch1', '/home/ouhaohan/Codes/RobustVideoMatting/media/output/batch3', '/home/ouhaohan/Codes/RobustVideoMatting/media/output/batch2']
# 去除背景
def cleanBg():
    # 指定输入和输出的目录
    model = MattingNetwork(variant='mobilenetv3').eval().cuda(0) # 或 variant="resnet50"
    model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))

    for input_dir, output_dir in zip(input_dirs, output_dirs):
        # 如果输出目录不存在，创建它
        os.makedirs(output_dir, exist_ok=True)

        # 遍历输入目录下的所有文件
        for filename in os.listdir(input_dir):
            # 获取文件的完整路径
            input_file = os.path.join(input_dir, filename)

            # 创建输出文件的路径
            output_file = os.path.join(output_dir, filename)

            # 检查输出文件是否已经存在
            if os.path.exists(output_file):
                print(f"Output file {output_file} already exists, skipping...")
                continue

            # 调用convert_video函数处理文件
            convert_video(
                model,                           # The loaded model, can be on any device (cpu or cuda).
                input_source=input_file,         # A video file or an image sequence directory.
                downsample_ratio=None,           # [Optional] If None, make downsampled max size be 512px.
                output_type='video',             # Choose "video" or "png_sequence"
                output_composition=output_file,  # File path if video; directory path if png sequence.
                output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
                seq_chunk=8,                     # Process n frames at once for better parallelism.
                progress=True                    # Print conversion progress.
            )

def mp4ToFrames(input_file, output_dir):
    # 读取视频文件
    cap = cv2.VideoCapture(input_file)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建一个进度条
    progress_bar = tqdm(total=total_frames, desc="Processing frames", ncols=70)
    # 逐帧读取视频并保存为图片
    frame_count = 0
    while True:
        # 读取一帧
        ret, frame = cap.read()

        # 如果无法读取到帧，退出循环
        if not ret:
            break

        # 图片文件名
        filename = f"{frame_count:04d}.jpg"  # 图片文件名从0开始依次排序，例如：0000.jpg, 0001.jpg, ...

        # 保存图片
        cv2.imwrite(os.path.join(output_dir, filename), frame)

        # 更新进度条
        progress_bar.update(1)

        frame_count += 1

    # 释放视频对象
    cap.release()

    # 关闭进度条
    progress_bar.close()
    # 打印已处理的文件名
    print(f"Finished processing {input_file}：output {output_dir}")

def process_directory(directory):
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 获取文件的完整路径
        input_file = os.path.join(directory, filename)
        # 获取输入文件的目录名
        input_dir_name = os.path.basename(directory)
        # 创建输出目录的路径
        output_dir = os.path.join('/home/ouhaohan/Codes/RobustVideoMatting/media/frames', input_dir_name, os.path.splitext(filename)[0])

        # 检查输出目录是否已经存在
        if os.path.exists(output_dir):
            print(f"Output directory {output_dir} already exists, skipping...")
            continue

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        # 调用mp4ToFrames函数处理文件
        mp4ToFrames(input_file, output_dir)

if __name__ == '__main__':

    # 遍历所有的output_dirs
    for output_dir in output_dirs:
        process_directory(output_dir)