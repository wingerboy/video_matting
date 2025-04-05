import sys
from birefnet import BiRefNet

# Use codes and weights locally
import torch
import logging
import os
import torch
from torchvision import transforms
import numpy as np
import random
import cv2
from PIL import Image
from time import time
from tqdm import tqdm


def check_state_dict(state_dict, unwanted_prefixes=['module.', '_orig_mod.']):
    for k, v in list(state_dict.items()):
        prefix_length = 0
        for unwanted_prefix in unwanted_prefixes:
            if k[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        state_dict[k[prefix_length:]] = state_dict.pop(k)
    return state_dict

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 图像预处理
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

# 提取前景
def refine_foreground(image, mask, r=90):
    if mask.size != image.size:
        mask = mask.resize(image.size)
    image = np.array(image) / 255.0
    mask = np.array(mask) / 255.0
    estimated_foreground = FB_blur_fusion_foreground_estimator_2(image, mask, r=r)
    image_masked = Image.fromarray((estimated_foreground * 255.0).astype(np.uint8))
    return image_masked

def FB_blur_fusion_foreground_estimator_2(image, alpha, r=90):
    # Thanks to the source: https://github.com/Photoroom/fast-foreground-estimation
    alpha = alpha[:, :, None]
    F, blur_B = FB_blur_fusion_foreground_estimator(
        image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator(image, F, blur_B, alpha, r=6)[0]


def FB_blur_fusion_foreground_estimator(image, F, B, alpha, r=90):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FA = cv2.blur(F * alpha, (r, r))
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    F = blurred_F + alpha * \
        (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    F = np.clip(F, 0, 1)
    return F, blurred_B

# 加载模型
model_name = 'BiRefNet_lite-2K'
birefnet = BiRefNet(bb_pretrained=False)
state_dict = torch.load('BiRefNet_lite-general-2K-epoch_232.pth', map_location=device)
state_dict = check_state_dict(state_dict)
birefnet.load_state_dict(state_dict)
birefnet.to(device)
birefnet.eval()
print("模型加载成功")

# 处理指定视频
video_src_path = "/Users/wingerliu/Downloads/222.mp4"
print('\n处理视频:', video_src_path)

# 创建输出目录和文件
src_dir = os.path.join(f'./frames-{model_name}-video_{os.path.splitext(os.path.basename(video_src_path))[0]}')
video_ext = os.path.splitext(video_src_path)[-1]
video_dst_path_mask = video_src_path.replace(video_ext, f'-preds_mask-{model_name}{video_ext}')
video_dst_path_subject = video_src_path.replace(video_ext, f'-preds_subject-{model_name}{video_ext}')

# 提取帧
vidcap = cv2.VideoCapture(video_src_path)
fps = vidcap.get(cv2.CAP_PROP_FPS)
success, image = vidcap.read()

if not success:
    print(f"无法读取视频: {video_src_path}")
    exit(1)

video_writer_shape = image.shape[:2][::-1]
video_writer_mask = cv2.VideoWriter(video_dst_path_mask, cv2.VideoWriter_fourcc(*'mp4v'), fps, video_writer_shape, isColor=False)
video_writer_subject = cv2.VideoWriter(video_dst_path_subject, cv2.VideoWriter_fourcc(*'mp4v'), fps, video_writer_shape, isColor=True)

count = 0
os.makedirs(src_dir, exist_ok=True)
print("正在提取视频帧...")

while success:
    cv2.imwrite(os.path.join(src_dir, f'frame_{count}.png'), image)
    success, image = vidcap.read()
    count += 1

print(f"共提取 {count} 帧")

# 处理所有帧
image_paths = [os.path.join(src_dir, f'frame_{i}.png') for i in range(count)]
    
time_st = time()
batch_size = 16  # 可以根据GPU内存调整批处理大小
total_frames = len(image_paths)

# 确定是否使用半精度
use_fp16 = device.type == 'cuda'
if use_fp16:
    print("使用半精度 (FP16) 加速处理")
    # 将模型转换为半精度
    birefnet = birefnet.half()

for idx in tqdm(range(0, total_frames, batch_size)):
    # 加载当前批次的图像
    current_batch = image_paths[idx:idx + batch_size]
    input_images_pil = [Image.open(img_path).convert("RGB") for img_path in current_batch]
    input_images = torch.cat([transform_image(img).unsqueeze(0) for img in input_images_pil], dim=0).to(device)
    
    # 如果使用GPU半精度，则转换输入
    if use_fp16:
        input_images = input_images.half()

    # 预测
    with torch.no_grad():
        preds = birefnet(input_images)
        if isinstance(preds, list):
            preds = preds[-1]
        preds = preds.sigmoid().cpu()

    for idx_pred in range(preds.shape[0]):
        pred = preds[idx_pred].squeeze()
        image = input_images_pil[idx_pred]

        # 生成蒙版图像
        pred_pil = transforms.ToPILImage()(pred)
        
        # 生成前景图像
        image_masked = refine_foreground(image, pred_pil)
        
        # 保存到视频
        mask_resized = pred_pil.resize(image.size).convert('L')
        video_writer_mask.write(np.array(mask_resized))
        
        # 合成前景和背景
        array_foreground = np.array(image_masked)[:, :, :3].astype(np.float32)
        array_mask = (np.array(mask_resized).astype(np.float32) / 255.0)[:, :, None]
        
        # 创建绿色背景
        array_background = np.zeros_like(array_foreground)
        array_background[:, :, 1] = 200  # 绿色背景
        
        # 合成
        array_foreground_background = array_foreground * array_mask + array_background * (1 - array_mask)
        video_writer_subject.write(cv2.cvtColor(array_foreground_background.astype(np.uint8), cv2.COLOR_RGB2BGR))

video_writer_mask.release()
video_writer_subject.release()
print(f"视频处理完成!")
print(f"输出文件: \n{video_dst_path_mask}\n{video_dst_path_subject}")
print(f"总耗时: {time() - time_st:.2f}秒")
