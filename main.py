import torch
from model import MattingNetwork
from inference import convert_video


model = MattingNetwork(variant='mobilenetv3').eval() # 或 variant="resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))

convert_video(
    model,                           # The loaded model, can be on any device (cpu or cuda).
    input_source='input.mp4',        # A video file or an image sequence directory.
    downsample_ratio=0.25,           # [Optional] If None, make downsampled max size be 512px.
    output_type='video',             # Choose "video" or "png_sequence"
    output_composition='./output/com1.mp4',    # File path if video; directory path if png sequence.
    # output_alpha="pha.mp4",          # [Optional] Output the raw alpha prediction.
    # output_foreground="fgr.mp4",     # [Optional] Output the raw foreground prediction.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    seq_chunk=12,                    # Process n frames at once for better parallelism.
    progress=True                    # Print conversion progress.
)