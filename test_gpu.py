#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BiRefNet视频抠图GPU加速程序

本程序支持两种类型的BiRefNet模型:
1. Lite版: 默认从birefnet.py导入，适用于参数较少的模型 (例如"BiRefNet_lite-general-2K-epoch_232.pth")
2. 完整版: 从models.birefnet导入，适用于参数较多的模型 (例如"BiRefNet-general-epoch_244.pth")

程序会自动根据模型文件名和权重参数形状判断模型类型，并使用合适的加载方式。

用法示例:
    # 使用默认设置
    python test_gpu.py
    
    # 指定视频和模型
    python test_gpu.py --video path/to/video.mp4 --model path/to/model.pth
    
    # 指定处理分辨率和批大小
    python test_gpu.py --size 1280,720 --batch 8
    
    # 使用完整版模型
    python test_gpu.py --model BiRefNet-general-epoch_244.pth

参数说明:
    --video: 输入视频路径
    --model: 模型路径，支持lite和完整版
    --size: 处理分辨率，格式为width,height
    --batch: 批处理大小，0表示自动根据GPU内存决定
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from time import time
from tqdm import tqdm
import argparse

# 添加models路径到导入路径，以便导入完整版模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    # 尝试导入完整版模型
    from models.birefnet import BiRefNet as FullBiRefNet
    HAS_FULL_MODEL = True
    print("成功导入完整版BiRefNet模型")
except ImportError:
    HAS_FULL_MODEL = False
    print("未找到完整版BiRefNet模型，将只使用lite版本")

# 导入lite版模型
from birefnet import BiRefNet as LiteBiRefNet

# 设置GPU内存管理优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 检查CUDA是否可用
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"可用GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 清理状态字典帮助函数
def check_state_dict(state_dict, unwanted_prefixes=['module.', '_orig_mod.']):
    for k, v in list(state_dict.items()):
        prefix_length = 0
        for unwanted_prefix in unwanted_prefixes:
            if k[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        state_dict[k[prefix_length:]] = state_dict.pop(k)
    return state_dict

# 图像预处理函数
def transform_image(image, size=(512, 512)):
    """将PIL图像转换为模型输入张量"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

# 反标准化函数
def denormalize(tensor):
    """将标准化的张量转换回原始范围"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    return tensor * std + mean

# GPU高斯模糊实现
def create_gaussian_kernel(kernel_size, sigma=1.0):
    """创建高斯卷积核"""
    # 创建一维高斯分布
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # 生成二维核
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d

def blur_tensor(tensor, kernel_size, sigma=None):
    """使用高斯卷积对张量进行模糊处理"""
    if sigma is None:
        sigma = kernel_size / 6.0
        
    # 生成高斯核
    kernel = create_gaussian_kernel(kernel_size, sigma)
    kernel = kernel.to(tensor.device)
    
    # 确保卷积核与输入张量类型一致 - 修复FP16/FP32不匹配
    kernel = kernel.to(dtype=tensor.dtype)
    
    # 为RGB图像准备核
    batch_size, channels = tensor.shape[:2]
    if channels == 3:
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        groups = 3
    else:
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        groups = 1
    
    # 添加填充以保持大小
    padding = kernel_size // 2
    
    # 针对批次中的每个图像应用卷积
    output = F.conv2d(tensor, kernel, padding=padding, groups=groups)
    
    return output

# GPU版本的前景提取函数
def fb_blur_fusion_foreground_estimator_gpu(image, alpha, r=90):
    """
    GPU版的前景估计器 - 第一阶段
    
    参数:
        image: [B,C,H,W] 的torch张量，值范围0-1
        alpha: [B,1,H,W] 的torch张量，值范围0-1
        r: 模糊核大小
    """
    # 确保alpha是4D: [B,1,H,W]
    if alpha.dim() == 3:
        alpha = alpha.unsqueeze(1)
    
    # 确保alpha和image类型一致
    alpha = alpha.to(dtype=image.dtype)
    
    # 计算模糊的alpha
    blurred_alpha = blur_tensor(alpha, r)
    
    # 计算模糊的F*alpha
    FA = image * alpha
    blurred_FA = blur_tensor(FA, r)
    
    # 避免除以0
    eps = 1e-6
    # 计算模糊的F
    blurred_F = blurred_FA / (blurred_alpha + eps)
    
    # 计算模糊的B*(1-alpha)
    B1A = image * (1 - alpha)
    blurred_B1A = blur_tensor(B1A, r)
    
    # 计算模糊的B
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + eps)
    
    # 更新F
    F = blurred_F + alpha * (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    F = torch.clamp(F, 0, 1)
    
    return F, blurred_B

def refine_foreground_gpu(image, mask, r1=45, r2=3):
    """
    完整的GPU版前景提取
    使用两阶段方法优化前景估计
    
    参数:
        image: [B,C,H,W] 图像张量，值范围0-1
        mask: [B,1,H,W] 掩码张量，值范围0-1
    """
    # 第一次运行大模糊核
    F, blur_B = fb_blur_fusion_foreground_estimator_gpu(image, mask, r1)
    
    # 第二次运行小模糊核
    F_refined, _ = fb_blur_fusion_foreground_estimator_gpu(image, mask, r2)
    
    return F_refined

# 添加模型加载函数
def load_birefnet_model(model_path, device):
    """
    智能加载BiRefNet模型，自动适应lite和标准版本
    
    参数:
        model_path: 模型权重文件路径
        device: 计算设备(CPU/GPU)
        
    返回:
        加载好的BiRefNet模型
    """
    # 检测模型类型
    model_filename = os.path.basename(model_path)
    is_lite_model = 'lite' in model_filename.lower()
    model_type = "BiRefNet-lite" if is_lite_model else "BiRefNet(完整版)"
    print(f"根据文件名检测到模型类型: {model_type}")
    
    # 尝试获取模型参数信息
    temp_state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # 通过权重参数维度判断是否为lite版
    bb_weight_shape = None
    for key in temp_state_dict:
        if 'bb.patch_embed.proj.weight' in key:
            bb_weight_shape = temp_state_dict[key].shape
            break
    
    # 根据形状判断是否为lite版
    is_lite_based_on_shape = True
    if bb_weight_shape:
        # lite版是[96,3,4,4]，完整版是[192,3,4,4]
        is_lite_based_on_shape = bb_weight_shape[0] < 150
        print(f"根据参数形状推断: {'Lite版' if is_lite_based_on_shape else '完整版'} (嵌入层大小: {bb_weight_shape[0]})")
    
    # 最终判断是否为lite版 (文件名和参数形状都要考虑)
    final_is_lite = is_lite_model or is_lite_based_on_shape
    
    # 使用适当的模型类加载
    if final_is_lite:
        print("加载Lite版BiRefNet模型")
        from BiRefNet_config import BiRefNetConfig
        config = BiRefNetConfig(bb_pretrained=False)
        model = LiteBiRefNet(bb_pretrained=False, config=config)
        
        # 加载权重
        state_dict = check_state_dict(temp_state_dict)
        try:
            model.load_state_dict(state_dict)
            print("Lite版模型加载成功")
        except Exception as e:
            print(f"Lite版模型加载失败，尝试非严格模式: {e}")
            model.load_state_dict(state_dict, strict=False)
            print("警告：部分参数未加载")
    else:
        # 尝试加载完整版模型
        if HAS_FULL_MODEL:
            print("加载完整版BiRefNet模型")
            # 使用完整版BiRefNet类
            model = FullBiRefNet(bb_pretrained=False)
            
            # 加载权重
            state_dict = check_state_dict(temp_state_dict)
            try:
                model.load_state_dict(state_dict)
                print("完整版模型加载成功")
            except Exception as e:
                print(f"完整版模型加载失败，尝试非严格模式: {e}")
                model.load_state_dict(state_dict, strict=False)
                print("警告：部分参数未加载")
        else:
            print("检测到完整版模型，但models.birefnet模块不可用")
            print("将尝试以Lite版模型加载，部分功能可能无法使用")
            
            # 使用Lite版模型加载
            from BiRefNet_config import BiRefNetConfig
            config = BiRefNetConfig(bb_pretrained=False)
            model = LiteBiRefNet(bb_pretrained=False, config=config)
            
            # 使用非严格模式加载
            state_dict = check_state_dict(temp_state_dict)
            model.load_state_dict(state_dict, strict=False)
            print("警告：以Lite版模型加载完整版权重，部分参数未加载")
    
    # 将模型移至设备
    model = model.to(device)
    model.eval()
    
    return model

# 完整的GPU视频处理函数
def process_video(video_path, output_mask_path, output_composite_path, model_path, image_size=(512, 512), batch_size=4):
    """
    高效的GPU视频处理函数
    
    参数:
        video_path: 输入视频路径
        output_mask_path: 输出掩码视频路径
        output_composite_path: 输出合成视频路径
        model_path: BiRefNet模型路径
        image_size: 模型输入尺寸
        batch_size: 批处理大小
    """
    print(f"处理视频: {video_path}")
    
    # 使用智能加载函数加载模型
    birefnet = load_birefnet_model(model_path, device)
    
    # 使用半精度加速
    use_fp16 = device.type == 'cuda'
    if use_fp16:
        print("使用FP16半精度加速")
        birefnet = birefnet.half()
        print(f"模型已转换为半精度: {next(birefnet.parameters()).dtype}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 获取视频尺寸
    ret, first_frame = cap.read()
    if not ret:
        print("无法读取视频")
        return
        
    original_h, original_w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开始
    
    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mask_writer = cv2.VideoWriter(output_mask_path, fourcc, fps, (original_w, original_h), False)
    composite_writer = cv2.VideoWriter(output_composite_path, fourcc, fps, (original_w, original_h), True)
    
    # 开始计时
    start_time = time()
    
    # 处理视频帧
    with torch.no_grad():
        for frame_idx in tqdm(range(0, total_frames, batch_size)):
            # 收集一批帧
            frames = []
            for _ in range(min(batch_size, total_frames - frame_idx)):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            if not frames:
                break
                
            # 转换批次帧到适合GPU的格式
            batch_tensors = []
            for frame in frames:
                # 转RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                tensor = transform_image(pil_img, image_size).unsqueeze(0)
                batch_tensors.append(tensor)
            
            input_batch = torch.cat(batch_tensors, dim=0).to(device)
            if use_fp16:
                input_batch = input_batch.half()
            
            # 调试信息
            if frame_idx == 0:
                print(f"输入张量类型: {input_batch.dtype}")
            
            # GPU事件计时
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            try:
                # 模型推理计时
                start_event.record()
                preds = birefnet(input_batch)
                if isinstance(preds, list):
                    preds = preds[-1]
                masks = preds.sigmoid()
                end_event.record()
                torch.cuda.synchronize()
                
                # 显示每个批次的推理时间
                if frame_idx == 0:
                    inference_time = start_event.elapsed_time(end_event)
                    print(f"模型推理时间: {inference_time:.2f} ms/批次 ({inference_time/batch_size:.2f} ms/帧)")
                    print(f"掩码张量类型: {masks.dtype}")
                
                # 将标准化的输入转回原始范围 (0-1)
                input_denorm = denormalize(input_batch)
                
                if frame_idx == 0:
                    print(f"反标准化后张量类型: {input_denorm.dtype}")
                
                # 前景提取计时
                start_event.record()
                foregrounds = refine_foreground_gpu(input_denorm, masks)
                end_event.record()
                torch.cuda.synchronize()
                
                # 显示前景提取时间
                if frame_idx == 0:
                    refinement_time = start_event.elapsed_time(end_event)
                    print(f"前景优化时间: {refinement_time:.2f} ms/批次 ({refinement_time/batch_size:.2f} ms/帧)")
                    print(f"前景张量类型: {foregrounds.dtype}")
                
                # 创建绿色背景 - 确保数据类型一致
                background = torch.zeros_like(input_denorm)
                background[:, 1] = 0.8  # 绿色通道设为0.8
                
                # 确保masks与前景和背景类型一致
                masks = masks.to(dtype=foregrounds.dtype)
                
                # 合成前景和背景
                composites = foregrounds * masks + background * (1 - masks)
                
                # 处理每一帧并写入视频
                for i in range(len(frames)):
                    original_size = (original_w, original_h)
                    
                    # 将掩码调整到原始大小并转为numpy
                    mask_np = masks[i, 0].cpu().float().numpy()  # 先转为float32
                    mask_np = np.ascontiguousarray(mask_np)  # 确保数组连续
                    mask_np = cv2.resize(mask_np, original_size, interpolation=cv2.INTER_LINEAR)
                    mask_np = (mask_np * 255).astype(np.uint8)
                    
                    # 将合成图像调整到原始大小并转为numpy
                    comp_np = composites[i].permute(1, 2, 0).cpu().float().numpy()  # 先转为float32
                    comp_np = np.ascontiguousarray(comp_np)  # 确保数组连续
                    comp_np = cv2.resize(comp_np, original_size, interpolation=cv2.INTER_LINEAR)
                    comp_np = (comp_np * 255).astype(np.uint8)
                    comp_np = cv2.cvtColor(comp_np, cv2.COLOR_RGB2BGR)
                    
                    # 写入视频
                    mask_writer.write(mask_np)
                    composite_writer.write(comp_np)
                
            except Exception as e:
                print(f"处理帧 {frame_idx} 时发生错误: {e}")
                import traceback
                traceback.print_exc()
                # 如果是第一批帧就失败，则退出
                if frame_idx == 0:
                    break
            
            # 定期清理GPU缓存
            if frame_idx % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
    
    # 关闭视频写入器
    cap.release()
    mask_writer.release()
    composite_writer.release()
    
    # 计算总时间
    total_time = time() - start_time
    fps_achieved = total_frames / total_time
    
    print(f"视频处理完成!")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均处理速度: {fps_achieved:.2f} FPS")
    print(f"输出文件:")
    print(f"1. 掩码视频: {output_mask_path}")
    print(f"2. 合成视频: {output_composite_path}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='BiRefNet视频抠图处理')
    parser.add_argument('--video', type=str, default="/Users/wingerliu/Downloads/111.mp4", 
                        help='输入视频路径')
    parser.add_argument('--model', type=str, default="BiRefNet_lite-general-2K-epoch_232.pth", 
                        help='模型路径，支持lite和完整版')
    parser.add_argument('--size', type=str, default="2560,1440", 
                        help='处理分辨率，格式为width,height')
    parser.add_argument('--batch', type=int, default=0, 
                        help='批处理大小，0表示自动根据GPU内存决定')
    args = parser.parse_args()
    
    # 视频路径
    video_path = args.video
    model_path = args.model
    
    # 解析处理分辨率
    try:
        width, height = map(int, args.size.split(','))
        image_size = (width, height)
        print(f"设置处理分辨率: {image_size}")
    except:
        image_size = (2560, 1440)
        print(f"使用默认处理分辨率: {image_size}")
    
    # 输出路径
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_mask_path = os.path.join(os.path.dirname(video_path), f"{video_name}-mask-{model_name}.mp4")
    output_composite_path = os.path.join(os.path.dirname(video_path), f"{video_name}-composite-{model_name}.mp4")
    
    # 设置批处理大小 - 根据GPU内存大小调整
    batch_size = args.batch
    if batch_size <= 0 and torch.cuda.is_available():
        # 自动计算合适的批处理大小
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem > 20:  # 高端GPU (4090, A100等)
            batch_size = 4  # 2K分辨率下减小批大小
        elif gpu_mem > 10:  # 中端GPU (3080, 2080等)
            batch_size = 2
        else:  # 入门级GPU
            batch_size = 1
        print(f"根据GPU内存({gpu_mem:.1f}GB)自动设置批处理大小: {batch_size}")
    else:
        if batch_size <= 0:
            batch_size = 1
        print(f"使用指定的批处理大小: {batch_size}")
    
    # 处理视频
    process_video(
        video_path=video_path,
        output_mask_path=output_mask_path,
        output_composite_path=output_composite_path,
        model_path=model_path,
        image_size=image_size,
        batch_size=batch_size
    ) 