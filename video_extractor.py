#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频抠图GPU加速程序

本程序支持两种模型:
1. BiRefNet: 支持lite版和完整版
2. BEN2: 支持BEN2_Base等模型

支持的模型文件格式:
- PyTorch格式 (.pth): 标准PyTorch保存的模型文件
- SafeTensors格式 (.safetensors): 更安全、更快的模型格式，需安装safetensors库

音频处理:
- 支持保留原始视频的音频轨道
- 需要安装moviepy库: pip install moviepy
- 如果未安装moviepy，输出视频将没有声音

背景处理:
- 默认使用绿色背景 (0, 255, 0)
- 支持使用自定义背景图片
- 支持RGB背景颜色值

用法示例:
    # 使用BiRefNet模型
    python video_extractor.py --method birefnet --model path/to/BiRefNet_model.pth --video path/to/video.mp4
    
    # 使用BEN2模型
    python video_extractor.py --method ben2 --model path/to/BEN2_Base.pth --video path/to/video.mp4
    
    # 使用自定义背景图片
    python video_extractor.py --bg path/to/background.jpg --video path/to/video.mp4 --model path/to/model.pth

    # 使用RGB背景颜色 (例如蓝色)
    python video_extractor.py --bg-color 0,0,255 --video path/to/video.mp4 --model path/to/model.pth
    
    # 指定输出目录和前缀
    python video_extractor.py --video path/to/video.mp4 --model path/to/model.pth --output ./results --prefix test01

    # 单独使用背景合成功能 (适用于已有掩码视频)
    from video_extractor import apply_custom_background
    apply_custom_background(
        video_path="原始视频.mp4",
        mask_path="掩码视频.mp4",
        output_path="输出视频.mp4",
        bg_path="背景图片.jpg"  # 可选，不提供则使用默认绿色背景
    )

参数说明:
    --method: 模型类型，支持 'birefnet' 或 'ben2'
    --video: 输入视频路径
    --model: 模型路径
    --output: 输出目录 (可选)
    --prefix: 输出文件前缀，默认使用时间戳 (可选)
    --bg: 背景图片路径 (可选)
    --bg-color: 背景RGB颜色值，格式为r,g,b，例如 0,255,0 表示绿色 (可选)
    --size: 处理分辨率，格式为width,height
    --batch: 批处理大小，0表示自动根据GPU内存决定

核心功能:
    1. predict_video_mask_birefnet    - BiRefNet模型预测前景掩码和生成默认背景视频
    2. predict_video_mask_ben2 - BEN2模型预测前景掩码和生成默认背景视频
    3. apply_custom_background - 将前景掩码与自定义背景合成
    4. extract_video - 统一的视频处理入口函数，支持上述所有功能

依赖库:
    必须: torch, numpy, opencv-python, pillow
    推荐: moviepy (保留音频), safetensors (支持safetensors模型格式)
    BEN2支持: BEN2 (请按照BEN2模型的要求安装)
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

# 尝试导入BEN2模型
try:
    import BEN2
    HAS_BEN2 = True
    print("成功导入BEN2模型")
except ImportError:
    HAS_BEN2 = False
    print("未找到BEN2模型，将只使用BiRefNet模型")

# 添加moviepy库用于处理带音频的视频
try:
    from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip, ImageClip, VideoClip
    HAS_MOVIEPY = True
    print("成功导入moviepy库，支持音频处理")
except ImportError:
    HAS_MOVIEPY = False
    print("未找到moviepy库，输出视频将没有声音。安装: pip install moviepy")

# 尝试导入safetensors库
try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
    print("成功导入safetensors库，支持.safetensors格式模型")
except ImportError:
    HAS_SAFETENSORS = False
    print("未找到safetensors库，将只支持.pth格式模型")

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
def transform_image(image, size=(1024, 1024)):
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
    支持.pth和.safetensors格式的模型文件
    
    参数:
        model_path: 模型权重文件路径
        device: 计算设备(CPU/GPU)
        
    返回:
        加载好的BiRefNet模型
    """
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
    # 检测模型文件格式
    is_safetensors = model_path.endswith('.safetensors')
    model_filename = os.path.basename(model_path)
    
    print(f"模型文件: {model_filename}")
    print(f"模型格式: {'safetensors' if is_safetensors else 'pth/torch'}")
    
    # 检测模型类型
    is_lite_model = 'lite' in model_filename.lower()
    model_type = "BiRefNet-lite" if is_lite_model else "BiRefNet(完整版)"
    print(f"根据文件名检测到模型类型: {model_type}")
    
    # 加载模型权重
    temp_state_dict = {}
    
    if is_safetensors:
        # 使用safetensors加载模型
        if not HAS_SAFETENSORS:
            raise ImportError("需要安装safetensors库以加载.safetensors格式模型: pip install safetensors")
            
        print("使用safetensors加载模型文件...")
        with safe_open(model_path, framework="pt", device="cpu") as f:
            temp_state_dict = {key: f.get_tensor(key) for key in f.keys()}
    else:
        # 使用torch.load加载模型
        print("使用torch加载模型文件...")
        temp_state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # 检查加载是否成功
    if not temp_state_dict:
        raise ValueError(f"模型文件加载失败或为空: {model_path}")
    
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

# 添加背景图像处理函数
def load_background(bg_path, target_size, device):
    """
    加载并处理背景图像
    
    参数:
        bg_path: 背景图像路径
        target_size: 目标大小 (width, height)
        device: 计算设备
        
    返回:
        处理后的背景图像张量 [3, H, W]
    """
    if not os.path.exists(bg_path):
        raise FileNotFoundError(f"背景图像不存在: {bg_path}")
        
    try:
        bg_img = Image.open(bg_path).convert("RGB")
        bg_img = bg_img.resize(target_size)
        
        # 转换为张量
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        bg_tensor = transform(bg_img).to(device)
        print(f"成功加载背景图像: {bg_path}, 大小: {target_size}")
        return bg_tensor
        
    except Exception as e:
        print(f"加载背景图像失败: {e}")
        print("将使用默认绿色背景")
        return None

def create_color_background(color, target_size, device):
    """
    创建纯色背景
    
    参数:
        color: RGB颜色值 [r, g, b]，范围0-255
        target_size: 目标大小 (width, height)
        device: 计算设备
        
    返回:
        背景图像张量 [3, H, W]
    """
    # 将颜色值归一化为0-1
    color = [c / 255.0 for c in color]
    
    # 创建RGB背景
    bg = torch.zeros((3, target_size[1], target_size[0]), device=device)
    bg[0, :, :] = color[0]  # R
    bg[1, :, :] = color[1]  # G
    bg[2, :, :] = color[2]  # B
    
    return bg

# 添加默认回调函数
def default_callback(status, progress=0, message=""):
    """默认回调函数，打印状态信息
    
    参数:
        status: 状态类型，如'init', 'progress', 'complete', 'error'
        progress: 进度百分比，0-100
        message: 状态信息
    """
    if status == 'init':
        print(f"初始化: {message}")
    elif status == 'progress':
        print(f"进度: {progress:.1f}%, {message}")
    elif status == 'complete':
        print(f"完成: {message}")
    elif status == 'error':
        print(f"错误: {message}")
    else:
        print(f"状态: {status}, {message}")

def simple_callback(status, progress=0, message=""):
    """简化版回调函数，只显示进度条
    
    参数:
        status: 状态类型，如'init', 'progress', 'complete', 'error'
        progress: 进度百分比，0-100
        message: 状态信息
    """
    import sys
    if status == 'progress':
        bar_length = 50
        filled_length = int(progress / 100 * bar_length)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        sys.stdout.write(f"\r进度: [{bar}] {progress:.1f}% {message}")
        sys.stdout.flush()
    elif status in ['complete', 'error']:
        sys.stdout.write('\n')
        print(f"{status}: {message}")
    elif status == 'init':
        print(f"开始处理: {message}")

def verbose_callback(status, progress=0, message=""):
    """详细版回调函数，显示所有状态变化和时间戳
    
    参数:
        status: 状态类型，如'init', 'progress', 'complete', 'error'
        progress: 进度百分比，0-100
        message: 状态信息
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    if status == 'progress':
        print(f"[{timestamp}] 进度更新: {progress:.1f}%, {message}")
    else:
        print(f"[{timestamp}] 状态变化: {status}, {message}")

# 拆分预测和背景合成部分
def predict_video_mask_birefnet(
    video_path, 
    output_mask_path, 
    output_composite_path, 
    model_path, 
    image_size=(512, 512), 
    batch_size=4, 
    bg_color=(0, 255, 0),
    callback=default_callback
):
    """
    BiRefNet模型预测视频前景掩码和合成默认背景的视频，不处理音频
    
    参数:
        video_path: 输入视频路径
        output_mask_path: 输出掩码视频路径
        output_composite_path: 输出合成视频路径
        model_path: BiRefNet模型路径
        image_size: 模型输入尺寸
        batch_size: 批处理大小
        bg_color: 默认背景颜色RGB值
        callback: 回调函数，用于报告进度
        
    返回:
        tuple: (掩码视频路径, 合成视频路径)
    """
    # 初始化状态
    callback('init', 0, f"开始处理视频: {video_path}")
    print(f"处理视频: {video_path}")
    
    try:
        # 使用智能加载函数加载模型
        callback('loading', 10, "加载模型...")
        birefnet = load_birefnet_model(model_path, device)
        
        # 使用半精度加速
        use_fp16 = device.type == 'cuda'
        if use_fp16:
            print("使用FP16半精度加速")
            birefnet = birefnet.half()
            print(f"模型已转换为半精度: {next(birefnet.parameters()).dtype}")
        
        # 打开视频
        callback('loading', 20, "读取视频信息...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 获取视频尺寸
        ret, first_frame = cap.read()
        if not ret:
            callback('error', 0, "无法读取视频")
            print("无法读取视频")
            
        original_h, original_w = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开始
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_composite_path), exist_ok=True)
        
        # 创建输出视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        mask_writer = cv2.VideoWriter(output_mask_path, fourcc, fps, (original_w, original_h), False)
        composite_writer = cv2.VideoWriter(output_composite_path, fourcc, fps, (original_w, original_h), True)
        
        # 开始计时
        start_time = time()
        
        callback('processing', 30, "开始处理视频帧...")
        # 处理视频帧
        with torch.no_grad():
            for frame_idx in tqdm(range(0, total_frames, batch_size)):
                # 计算并报告进度
                progress = 30 + (frame_idx / total_frames) * 60  # 占总进度的60%
                callback('progress', progress, f"处理帧 {frame_idx}/{total_frames}")
                
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
                
                try:
                    # 模型推理
                    preds = birefnet(input_batch)
                    if isinstance(preds, list):
                        preds = preds[-1]
                    masks = preds.sigmoid()
                    
                    # 将标准化的输入转回原始范围 (0-1)
                    input_denorm = denormalize(input_batch)
                    
                    # 前景提取
                    foregrounds = refine_foreground_gpu(input_denorm, masks)
                    
                    # 获取前景的尺寸，用于创建匹配的背景
                    _, _, fg_h, fg_w = foregrounds.shape
                    
                    # 创建与前景相同尺寸的背景
                    background = create_color_background(bg_color, (fg_w, fg_h), device)
                    background = background.unsqueeze(0).expand(foregrounds.shape[0], -1, -1, -1)
                    background = background.to(dtype=foregrounds.dtype)
                    
                    # 确保masks与前景和背景类型一致
                    masks = masks.to(dtype=foregrounds.dtype)
                    
                    # 合成前景和背景
                    composites = foregrounds * masks + background * (1 - masks)
                    
                    # 处理每一帧并保存/写入
                    for i, frame_index in enumerate(range(frame_idx, frame_idx + len(frames))):
                        original_size = (original_w, original_h)
                        
                        # 将掩码调整到原始大小并转为numpy
                        mask_np = masks[i, 0].cpu().float().numpy()
                        mask_np = np.ascontiguousarray(mask_np)
                        mask_np = cv2.resize(mask_np, original_size, interpolation=cv2.INTER_LINEAR)
                        mask_np = (mask_np * 255).astype(np.uint8)
                        
                        # 将合成图像调整到原始大小并转为numpy
                        comp_np = composites[i].permute(1, 2, 0).cpu().float().numpy()
                        comp_np = np.ascontiguousarray(comp_np)
                        comp_np = cv2.resize(comp_np, original_size, interpolation=cv2.INTER_LINEAR)
                        comp_np = (comp_np * 255).astype(np.uint8)
                        comp_np = cv2.cvtColor(comp_np, cv2.COLOR_RGB2BGR)
                        
                        # 写入视频文件
                        mask_writer.write(mask_np)
                        composite_writer.write(comp_np)
                    
                except Exception as e:
                    error_msg = f"处理帧 {frame_idx} 时发生错误: {e}"
                    callback('error', progress, error_msg)
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                    # 如果是第一批帧就失败，则退出
                    if frame_idx == 0:
                        break
                
                # 定期清理GPU缓存
                if frame_idx % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
        
        # 关闭资源
        cap.release()
        mask_writer.release()
        composite_writer.release()
        
        # 计算总时间
        total_time = time() - start_time
        fps_achieved = total_frames / total_time
        
        callback('complete', 100, f"处理完成，平均速度: {fps_achieved:.2f} FPS")
        print(f"视频处理完成!")
        print(f"总时间: {total_time:.2f}秒")
        print(f"平均处理速度: {fps_achieved:.2f} FPS")
        print(f"输出文件:")
        print(f"1. 掩码视频: {output_mask_path}")
        print(f"2. 合成视频: {output_composite_path}")
        
        # 验证输出文件存在
        if not os.path.exists(output_mask_path) or not os.path.exists(output_composite_path):
            callback('warning', 100, "输出文件可能未正确生成")
            print("警告: 输出文件可能未正确生成")
    
    except Exception as e:
        callback('error', 0, f"处理视频时发生错误: {e}")
        print(f"处理视频时发生错误: {e}")
        import traceback
        traceback.print_exc()

def predict_video_mask_ben2(
    video_path, 
    output_mask_path, 
    output_composite_path, 
    model_path, 
    batch_size=4, 
    bg_color=(0, 255, 0),
    callback=default_callback
):
    """
    BEN2模型预测视频前景掩码和合成默认背景的视频，不处理音频
    
    参数:
        video_path: 输入视频路径
        output_mask_path: 输出掩码视频路径
        output_composite_path: 输出前景视频路径
        model_path: BEN2模型路径
        batch_size: 批处理大小
        bg_color: 默认背景颜色RGB值
        callback: 回调函数，用于报告进度
        
    返回:
        tuple: (掩码视频路径, 合成视频路径)
    """
    if not HAS_BEN2:
        error_msg = "错误: 未安装BEN2模型，无法使用此方法"
        callback('error', 0, error_msg)
        print(error_msg)
        print("请安装BEN2模型后重试")
        
    callback('init', 0, f"开始处理视频: {video_path}")
    
    try:
        # 初始化BEN2模型
        callback('loading', 10, "加载BEN2模型...")
        model = BEN2.BEN_Base().to(device).eval()
        model.loadcheckpoints(model_path)
        
        print(f"成功加载BEN2模型: {model_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_composite_path), exist_ok=True)
        
        # 直接使用BEN2的视频分割功能
        callback('processing', 30, "提取视频前景和掩码...")
        try:
            # 处理进度监控
            def progress_monitor(frame, total, message="处理中"):
                progress = 30 + (frame / total) * 60
                if callable(callback):
                    callback('progress', progress, f"BEN2 {message}: {frame}/{total}")
            
            # 使用BEN2直接处理视频
            result_mask_path, result_composite_path = model.segment_video_v2(
                video_path=video_path,
                output_mask_path=output_mask_path,
                output_composite_path=output_composite_path,
                fps=0,  # 自动检测FPS
                refine_foreground=True,
                batch=batch_size,
                rgb_value=bg_color,  # 默认背景色
                progress_callback=progress_monitor
            )
            
            # 检查输出文件是否存在
            if not os.path.exists(output_mask_path):
                callback('warning', 90, "掩码视频文件未生成")
                print("警告: 掩码视频文件未生成")
                result_mask_path = None
                
            if not os.path.exists(output_composite_path):
                callback('warning', 90, "合成视频文件未生成")
                print("警告: 合成视频文件未生成")
                result_composite_path = None
            
            callback('complete', 100, "BEN2视频处理完成")
            print(f"BEN2视频处理完成")
            print(f"输出文件:")
            if result_mask_path:
                print(f"1. 掩码视频: {result_mask_path}")
            if result_composite_path:
                print(f"2. 合成视频: {result_composite_path}")
                
            return result_mask_path, result_composite_path
            
        except Exception as e:
            error_msg = f"BEN2视频处理出错: {e}"
            callback('error', 60, error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, None
            
    except Exception as e:
        error_msg = f"BEN2处理视频时出错: {e}"
        callback('error', 0, error_msg)
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, None

def apply_custom_background(
    video_path,
    mask_video_path,
    background_path,
    output_composite_path,
    include_audio=True,
    callback=default_callback
):
    """
    将视频前景与自定义背景合成，同时保留原始音频
    
    参数:
        video_path: 原始视频路径
        mask_video_path: 掩码视频路径
        background_path: 背景图片或视频路径
        output_composite_path: 输出合成视频路径
        include_audio: 是否包含原始视频的音频
        callback: 回调函数，用于报告进度
        
    返回:
        str: 合成视频路径，失败时返回None
    """
    try:        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_composite_path), exist_ok=True)
        
        callback('processing', 5, "开始合成视频和背景")
        
        # 打开视频
        video_clip = VideoFileClip(video_path)
        mask_clip = VideoFileClip(mask_video_path)
        
        callback('processing', 15, "正在加载背景图片")
        bg_image = Image.open(background_path)
        
        # 确保背景尺寸和视频一致
        bg_image = bg_image.resize((video_clip.size[0], video_clip.size[1]), Image.LANCZOS)
        
        # 创建持续时间与视频相同的图像clip
        bg_clip = ImageClip(np.array(bg_image)).set_duration(video_clip.duration)
        
        # 创建掩码函数，用于将mask_clip作为alpha通道应用到video_clip
        callback('processing', 25, "正在合成前景和背景")
        
        def make_frame(t):
            fg_frame = video_clip.get_frame(t)
            bg_frame = bg_clip.get_frame(t)
            mask_frame = mask_clip.get_frame(t)
            
            # 确保掩码为灰度图
            if len(mask_frame.shape) == 3 and mask_frame.shape[2] > 1:
                mask_gray = np.mean(mask_frame, axis=2) / 255.0
            else:
                mask_gray = mask_frame / 255.0
                
            # 扩展维度以匹配RGB通道
            mask_3d = np.stack([mask_gray] * 3, axis=-1)
            
            # 使用掩码混合前景和背景
            composite = fg_frame * mask_3d + bg_frame * (1 - mask_3d)
            return composite
        
        # 创建合成视频
        composite_clip = VideoClip(make_frame, duration=video_clip.duration)
        
        # 添加原始音频（如果需要）
        if include_audio and video_clip.audio is not None:
            callback('processing', 75, "正在添加音频")
            composite_clip = composite_clip.set_audio(video_clip.audio)
        
        # 写入输出文件
        callback('processing', 80, "正在保存合成视频")
        composite_clip.write_videofile(
            output_composite_path,
            codec='libx264',
            audio_codec='aac' if include_audio and video_clip.audio is not None else None,
            fps=video_clip.fps,
            progress_bar=False
        )
        
        # 清理资源
        video_clip.close()
        mask_clip.close()
        bg_clip.close()
        composite_clip.close()
        
        callback('complete', 100, "合成视频处理完成")
        print(f"合成视频处理完成: {output_composite_path}")
    
    except Exception as e:
        error_msg = f"合成视频处理出错: {e}"
        callback('error', 0, error_msg)
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        # 尝试清理可能打开的资源
        try:
            if 'video_clip' in locals() and video_clip:
                video_clip.close()
            if 'mask_clip' in locals() and mask_clip:
                mask_clip.close()
            if 'bg_clip' in locals() and bg_clip:
                bg_clip.close()
            if 'composite_clip' in locals() and composite_clip:
                composite_clip.close()
        except:
            pass

# 修改原有process_video函数为调用新拆分的函数
def extract_video(
    video_path,
    model_path,
    foreground_video_path,
    output_mask_path,
    method="birefnet",
    image_size=(512, 512),
    batch_size=4,
    bg_color=(0, 255, 0),
    callback=default_callback
):
    """
    统一的视频处理入口函数，支持多种模型和背景处理
    
    参数:
        video_path: 输入视频路径
        model_path: 模型路径，支持.pth和.safetensors格式
        foreground_video_path: 指定的前景视频输出路径
        output_mask_path: 指定的掩码视频输出路径
        method: 模型类型，支持'birefnet'或'ben2'
        image_size: 模型输入尺寸，格式为(width, height)
        batch_size: 批处理大小
        bg_color: 背景颜色RGB值，默认为绿色(0,255,0)
        callback: 回调函数，用于报告进度
        
    返回:
        tuple: (掩码视频路径, None)，处理失败时返回(None, None)
    """
    try:        
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        
        # 根据方法选择处理函数
        callback('processing', 10, f"使用{method.upper()}模型处理视频")
        if method.lower() == "ben2":
            if not HAS_BEN2:
                error_msg = "错误: 未安装BEN2模型，无法使用此方法"
                callback('error', 0, error_msg)
                print(error_msg)
                return None, None
                
            # 禁用合成视频输出，只生成掩码
            predict_video_mask_ben2(
                video_path=video_path,
                output_mask_path=output_mask_path,
                output_composite_path=foreground_video_path,
                model_path=model_path,
                batch_size=batch_size,
                bg_color=bg_color,
                callback=callback
            )
            
        elif method.lower() == "birefnet":
            # 禁用合成视频输出，只生成掩码
            predict_video_mask_birefnet(
                video_path=video_path,
                output_mask_path=output_mask_path,
                output_composite_path=foreground_video_path,
                model_path=model_path,
                image_size=image_size, 
                batch_size=batch_size, 
                bg_color=bg_color,
                callback=callback
            )
        
        # 检查输出文件是否存在
        if os.path.exists(mask_path) and os.path.exists(foreground_video_path):
            callback('complete', 100, "掩码视频生成完成")
            print(f"掩码视频生成完成: {mask_path}, 前景视频生成完成: {foreground_video_path}")
        else:
            warn_msg = f"警告: 掩码视频or前景视频未生成: {output_mask_path} or {foreground_video_path}"
            callback('warning', 95, warn_msg)
            print(warn_msg)
    
    except Exception as e:
        error_msg = f"视频处理出错: {e}"
        callback('error', 0, error_msg)
        print(error_msg)
        import traceback
        traceback.print_exc()
