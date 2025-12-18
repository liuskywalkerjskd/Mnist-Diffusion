import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import os
import argparse
import cv2 # 需要 pip install opencv-python
import numpy as np

# 导入模型定义
from ddpm_model import ContextUnet, DDPM, ddpm_schedules

def load_model(model_path, device):
    n_T = 400
    n_feat = 128
    n_classes = 10
    model = ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
    ddpm_constants = ddpm_schedules(1e-4, 0.02, n_T)
    ddpm = DDPM(nn_model=model, betas=ddpm_constants, n_T=n_T, device=device, drop_prob=0.0)
    ddpm.to(device)
    ddpm.eval()
    return ddpm

def generate_digit_image(digit, n_samples, model_path, output_dir, device="cuda"):
    """ 生成单张图片模式 """
    ddpm = load_model(model_path, device)
    print(f"Generating {n_samples} images of digit '{digit}'...")
    
    c = torch.tensor([digit]).to(device)
    c = c.repeat(n_samples) 
    c = F.one_hot(c, num_classes=10).float()

    with torch.no_grad():
        # 不需要中间过程，return_all_steps 默认为 False
        x_gen = ddpm.sample(n_samples, (1, 28, 28), device, guide_w=2.0, c_labels=c)

    os.makedirs(output_dir, exist_ok=True)
    save_path_grid = os.path.join(output_dir, f"digit_{digit}_grid.png")
    save_image(x_gen, save_path_grid, nrow=int(n_samples**0.5), normalize=True, value_range=(-1, 1))
    print(f"Saved to {save_path_grid}")

def generate_video_0to9(model_path, output_dir, device="cuda", fps=60):
    """ 生成 0-9 去噪过程视频模式 """
    ddpm = load_model(model_path, device)
    print("Generating denoising video for digits 0-9...")

    # 构造 0-9 的标签
    n_samples = 10
    c = torch.arange(0, 10).to(device) # [0, 1, ..., 9]
    c = F.one_hot(c, num_classes=10).float()

    # 采样，并要求返回所有步骤
    with torch.no_grad():
        # trajectories shape: [T+1, 10, 1, 28, 28]
        trajectories = ddpm.sample(n_samples, (1, 28, 28), device, guide_w=2.0, c_labels=c, return_all_steps=True)

    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "denoising_process_0to9.mp4")
    
    writer = None
    
    print("\nProcessing frames and writing video...")
    # 遍历每一个时间步
    for i in range(trajectories.shape[0]):
        # 取出当前步的 10 张图
        frame_batch = trajectories[i] # [10, 1, 28, 28]
        
        # 拼成网格 (2行5列)，并归一化到 [0, 1]
        # value_range=(-1, 1) 很重要，因为中间噪声可能超出这个范围，需要截断并映射
        grid = make_grid(frame_batch, nrow=5, normalize=True, value_range=(-1, 1), scale_each=False)
        
        # 转换为 OpenCV 可以处理的 numpy uint8 格式
        # [C, H, W] -> [H, W, C] -> 乘255 -> 转uint8 -> 转numpy
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        
        # RGB 转 BGR (OpenCV 默认格式)
        img_bgr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)

        # 初始化视频写入器 (只在第一帧执行)
        if writer is None:
            h, w = img_bgr.shape[:2]
            # 'mp4v' 是一个常用的兼容性较好的编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        writer.write(img_bgr)
        print(f'Processing frame {i}/{trajectories.shape[0]}', end='\r')
        
    writer.release()
    print(f"\nVideo saved to {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 添加一个开关，用于切换视频模式
    parser.add_argument("--video", action="store_true", help="Generate denoising video for 0-9")
    
    parser.add_argument("--digit", type=int, default=5, help="Digit to generate (image mode)")
    parser.add_argument("--count", type=int, default=9, help="Number of images (image mode)")
    parser.add_argument("--weights", type=str, default="./weights/ddpm_mnist.pth")
    parser.add_argument("--outdir", type=str, default="./generated_outputs")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.video:
        # 视频模式
        generate_video_0to9(args.weights, args.outdir, device, fps=60)
    else:
        # 图片模式
        generate_digit_image(args.digit, args.count, args.weights, args.outdir, device)