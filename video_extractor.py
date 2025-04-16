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

model_config = {
    # 耗时 半小时左右
    "BEN2_Base": {
        "method": "ben2",
        "model_path": "/root/gpufree-data/models/BEN2_Base.pth",
        "image_size": (1024, 1024),
        "max_batch_size": 16,
    },
    # 耗时 616.55秒左右
    "BiRefNet-HRSOD_DHU": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet-HRSOD_DHU-epoch_115.pth",
        "image_size": (1024, 1024),
        "max_batch_size": 16,
    },
    # 耗时 619s 左右
    "BiRefNet-massive-TR_DIS5K_TR_TEs": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.pth",
        "image_size": (1024, 1024),
        "max_batch_size": 16,
    },
    # 耗时 657.93s 左右
    "BiRefNet": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet.safetensors",
        "image_size": (1024, 1024),
        "max_batch_size": 16,
    },
    # 耗时 609.84 左右
    "BiRefNet_dynamic-matting": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet_dynamic-matting-epoch_159.pth",
        "image_size": (1024, 1024),
        "max_batch_size": 16,
    },
    # 耗时  左右  --代码错误，待修复
    "BiRefNet-COD": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet-COD-epoch_125.pth",
        "image_size": (1024, 1024),
    },
    # 耗时 526.06s 左右
    "BiRefNet-general-bb_swin_v1_tiny": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth",
        "image_size": (1024, 1024),
        "max_batch_size": 16,
    },
    "BiRefNet-matting": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet-matting-epoch_100.pth",
        "image_size": (1024, 1024),
    },
    "BiRefNet_HR-general": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet_HR-general-epoch_130.pth",
        "image_size": (2048, 2048),
    },
    "BiRefNet_lite-general-2K": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet_lite-general-2K-epoch_232.pth",
        "image_size": (2560, 1440),
    },
    "BiRefNet-DIS": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet-DIS-epoch_590.pth",
        "image_size": (1024, 1024),
    },
    "BiRefNet-general": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet-general-epoch_244.pth",
        "image_size": (1024, 1024),
    },  
    "BiRefNet-portrait": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet-portrait-epoch_150.pth",
        "image_size": (1024, 1024),
    },   
    "BiRefNet_HR-matting": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet_HR-matting-epoch_135.pth",
        "image_size": (2048, 2048),
    },   
    "BiRefNet_lite": {
        "method": "birefnet",
        "model_path": "/root/gpufree-data/models/BiRefNet_lite.safetensors",
        "image_size": (1024, 1024),
    }    
}

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
            return None, None
            
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
            
        return output_mask_path, output_composite_path
    
    except Exception as e:
        callback('error', 0, f"处理视频时发生错误: {e}")
        print(f"处理视频时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

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
        output_composite_path: 输出合成视频路径
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
        return None, None
        
    callback('init', 0, f"开始处理视频: {video_path}")
    print(f"使用BEN2模型处理视频: {video_path}")
    
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
    mask_path,
    output_path,
    bg_path=None,
    bg_color=(0, 255, 0),
    callback=default_callback,
    include_audio=True
):
    """
    将视频前景与自定义背景合成，并保留原始视频的音频
    
    参数:
        video_path: 原始视频路径
        mask_path: 预测的掩码视频路径
        output_path: 输出合成视频路径
        bg_path: 背景图像路径，如果为None则使用bg_color
        bg_color: 背景颜色RGB值，默认为绿色(0,255,0)
        callback: 回调函数，用于报告进度
        include_audio: 是否包含音频，默认为True
        
    返回:
        str: 成功时返回输出视频路径，失败时返回None
    """
    if not include_audio or not HAS_MOVIEPY:
        # 无音频模式处理或没有安装moviepy库
        if include_audio and not HAS_MOVIEPY:
            error_msg = "警告: 未安装moviepy库，无法处理音频"
            callback('warning', 0, error_msg)
            print(error_msg)
            print("将生成无音频视频，如需音频请安装: pip install moviepy")
            
        return apply_custom_background_no_audio(
            video_path, mask_path, output_path, bg_path, bg_color, callback
        )
    
    # 以下为包含音频的处理流程
    # 资源管理变量，用于清理
    original_video = None
    mask_video = None
    background_clip = None
    composite_clip = None
    
    try:
        callback('init', 0, f"开始背景合成处理")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 检查输入文件是否存在
        if not os.path.exists(video_path):
            error_msg = f"错误: 原始视频不存在: {video_path}"
            callback('error', 0, error_msg)
            print(error_msg)
            return None
            
        if not os.path.exists(mask_path):
            error_msg = f"错误: 掩码视频不存在: {mask_path}"
            callback('error', 0, error_msg)
            print(error_msg)
            return None
            
        # 检查背景图片路径（如果指定）
        if bg_path and not os.path.exists(bg_path):
            warn_msg = f"警告: 指定的背景图片不存在: {bg_path}，将使用纯色背景"
            callback('warning', 5, warn_msg)
            print(warn_msg)
            bg_path = None
        
        # 加载原始视频和掩码视频
        callback('loading', 10, "加载视频...")
        original_video = VideoFileClip(video_path)
        mask_video = VideoFileClip(mask_path, has_mask=False)
        
        # 检查两个视频的尺寸是否匹配
        if original_video.size != mask_video.size:
            warn_msg = f"警告: 原始视频({original_video.size})和掩码视频({mask_video.size})尺寸不匹配"
            callback('warning', 15, warn_msg)
            print(warn_msg)
            print("尝试调整掩码视频尺寸以匹配原始视频...")
            # 调整掩码视频尺寸
            mask_video = mask_video.resize(original_video.size)
        
        # 视频基本信息
        width, height = original_video.size
        fps = original_video.fps
        duration = original_video.duration
        
        # 检查是否有音频
        has_audio = original_video.audio is not None
        if not has_audio:
            print("原始视频没有音频轨道")
        else:
            print(f"检测到音频轨道, 时长: {original_video.audio.duration}秒")
        
        # 处理背景
        callback('processing', 30, "处理背景...")
        if bg_path:
            # 检查图片格式
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            if not any(bg_path.lower().endswith(ext) for ext in valid_extensions):
                warn_msg = f"警告: 背景图片格式可能不受支持: {bg_path}"
                callback('warning', 32, warn_msg)
                print(warn_msg)
            
            # 加载背景图像
            try:
                # 先获取图片原始尺寸
                bg_img_orig = Image.open(bg_path)
                orig_width, orig_height = bg_img_orig.size
                
                # 检查尺寸是否匹配
                if orig_width != width or orig_height != height:
                    aspect_ratio_video = width / height
                    aspect_ratio_bg = orig_width / orig_height
                    
                    # 检查纵横比差异
                    ratio_diff = abs(aspect_ratio_video - aspect_ratio_bg) / aspect_ratio_video
                    
                    if ratio_diff > 0.1:  # 纵横比差异超过10%
                        warn_msg = (f"警告: 背景图片({orig_width}x{orig_height})与视频({width}x{height})纵横比差异较大，"
                                  f"可能导致拉伸变形")
                        callback('warning', 35, warn_msg)
                        print(warn_msg)
                    else:
                        info_msg = f"背景图片将从 {orig_width}x{orig_height} 调整为 {width}x{height}"
                        callback('info', 35, info_msg)
                        print(info_msg)
                
                # 调整大小并转换格式
                background_img = bg_img_orig.convert("RGB")
                background_img = background_img.resize((width, height))
                background_array = np.array(background_img)
                
                # 创建背景视频 - 静态图像
                background_clip = ImageClip(background_array, duration=duration)
                callback('processing', 40, f"使用自定义背景: {bg_path}")
            except Exception as e:
                error_msg = f"加载背景图像失败: {e}"
                callback('error', 40, error_msg)
                print(error_msg)
                print("将使用默认纯色背景")
                
                # 创建纯色背景
                color_array = np.zeros((height, width, 3), dtype=np.uint8)
                color_array[:, :, 0] = bg_color[0]  # R
                color_array[:, :, 1] = bg_color[1]  # G
                color_array[:, :, 2] = bg_color[2]  # B
                background_clip = ImageClip(color_array, duration=duration)
                callback('processing', 40, f"使用纯色背景: RGB{bg_color}")
        else:
            # 创建纯色背景
            color_array = np.zeros((height, width, 3), dtype=np.uint8)
            color_array[:, :, 0] = bg_color[0]  # R
            color_array[:, :, 1] = bg_color[1]  # G
            color_array[:, :, 2] = bg_color[2]  # B
            background_clip = ImageClip(color_array, duration=duration)
            callback('processing', 40, f"使用纯色背景: RGB{bg_color}")
        
        # 创建合成函数
        callback('processing', 50, "创建合成视频...")
        def blend_frame(get_frame, t):
            # 获取原视频帧
            original_frame = original_video.get_frame(t)
            # 获取掩码帧并确保是单通道
            mask_frame = mask_video.get_frame(t)
            if len(mask_frame.shape) == 3 and mask_frame.shape[2] > 1:
                mask_frame = mask_frame[:, :, 0]
            # 将掩码归一化到0-1
            mask_frame = mask_frame / 255.0
            # 获取背景帧
            bg_frame = background_clip.get_frame(t)
            # 扩展掩码到3通道
            mask_3ch = np.stack([mask_frame] * 3, axis=2)
            # 合成
            return original_frame * mask_3ch + bg_frame * (1 - mask_3ch)
        
        # 创建合成视频
        callback('processing', 60, "生成合成视频...")
        composite_clip = VideoClip(blend_frame, duration=duration)
        
        # 添加音频
        if has_audio:
            callback('processing', 70, "添加音频...")
            composite_clip = composite_clip.set_audio(original_video.audio)
        
        # 写入视频
        callback('finalizing', 80, "保存合成视频...")
        composite_clip.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac')
        
        # 清理资源
        callback('finalizing', 90, "清理资源...")
        if original_video: original_video.close()
        if mask_video: mask_video.close()
        if background_clip: background_clip.close()
        if composite_clip: composite_clip.close()
        
        callback('complete', 100, "背景合成完成")
        print(f"背景合成完成!")
        print(f"输出文件: {output_path}")
        return output_path
        
    except Exception as e:
        error_msg = f"背景合成处理出错: {e}"
        callback('error', 0, error_msg)
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        # 确保资源被清理
        try:
            if original_video: original_video.close()
            if mask_video: mask_video.close()
            if background_clip: background_clip.close()
            if composite_clip: composite_clip.close()
        except Exception as cleanup_error:
            print(f"清理资源时出错: {cleanup_error}")
        
        return None

def apply_custom_background_no_audio(
    video_path,
    mask_path,
    output_path,
    bg_path=None,
    bg_color=(0, 255, 0),
    callback=default_callback
):
    """
    将视频前景与自定义背景合成（不包含音频），使用纯OpenCV实现
    
    参数:
        video_path: 原始视频路径
        mask_path: 预测的掩码视频路径
        output_path: 输出合成视频路径
        bg_path: 背景图像路径，如果为None则使用bg_color
        bg_color: 背景颜色RGB值，默认为绿色(0,255,0)
        callback: 回调函数，用于报告进度
        
    返回:
        str: 成功时返回输出视频路径，失败时返回None
    """
    try:
        callback('init', 0, f"开始背景合成处理 (无音频模式)")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 检查输入文件是否存在
        if not os.path.exists(video_path):
            error_msg = f"错误: 原始视频不存在: {video_path}"
            callback('error', 0, error_msg)
            print(error_msg)
            return None
            
        if not os.path.exists(mask_path):
            error_msg = f"错误: 掩码视频不存在: {mask_path}"
            callback('error', 0, error_msg)
            print(error_msg)
            return None
        
        # 打开原始视频和掩码视频
        callback('loading', 10, "加载视频...")
        original_cap = cv2.VideoCapture(video_path)
        mask_cap = cv2.VideoCapture(mask_path)
        
        # 视频基本信息
        fps = original_cap.get(cv2.CAP_PROP_FPS)
        total_frames = min(
            int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT)), 
            int(mask_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        
        # 获取视频尺寸
        ret, first_frame = original_cap.read()
        if not ret:
            callback('error', 0, "无法读取原始视频")
            print("无法读取原始视频")
            return None
            
        original_h, original_w = first_frame.shape[:2]
        original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开始
        
        # 检查掩码视频尺寸
        ret, first_mask = mask_cap.read()
        if not ret:
            callback('error', 0, "无法读取掩码视频")
            print("无法读取掩码视频")
            return None
            
        mask_h, mask_w = first_mask.shape[:2]
        mask_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开始
        
        # 检查尺寸是否匹配
        if original_w != mask_w or original_h != mask_h:
            warn_msg = f"警告: 原始视频({original_w}x{original_h})和掩码视频({mask_w}x{mask_h})尺寸不匹配"
            callback('warning', 15, warn_msg)
            print(warn_msg)
        
        # 处理背景
        callback('processing', 30, "准备背景...")
        
        # 如果提供了背景图片，尝试加载
        background = None
        if bg_path and os.path.exists(bg_path):
            try:
                bg_img = cv2.imread(bg_path)
                if bg_img is not None:
                    # 调整背景大小以匹配视频
                    background = cv2.resize(bg_img, (original_w, original_h))
                    print(f"使用自定义背景: {bg_path}")
                else:
                    print(f"无法加载背景图片: {bg_path}，将使用纯色背景")
            except Exception as e:
                print(f"加载背景图片错误: {e}，将使用纯色背景")
        
        # 如果没有背景图片或加载失败，创建纯色背景
        if background is None:
            background = np.zeros((original_h, original_w, 3), dtype=np.uint8)
            background[:, :] = bg_color[::-1]  # OpenCV使用BGR格式
            print(f"使用纯色背景: BGR{bg_color[::-1]}")
        
        # 创建输出视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(output_path, fourcc, fps, (original_w, original_h), True)
        
        # 处理视频帧
        callback('processing', 40, "开始合成视频...")
        frame_count = 0
        
        while frame_count < total_frames:
            # 计算并报告进度
            progress = 40 + (frame_count / total_frames) * 50
            if frame_count % 10 == 0:
                callback('progress', progress, f"合成帧 {frame_count}/{total_frames}")
            
            # 读取原始帧和掩码帧
            ret1, original_frame = original_cap.read()
            ret2, mask_frame = mask_cap.read()
            
            if not ret1 or not ret2:
                break
            
            # 将掩码帧转为灰度图，如果它不是灰度图
            if len(mask_frame.shape) == 3 and mask_frame.shape[2] > 1:
                mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask_frame
            
            # 如果尺寸不一致，调整掩码尺寸
            if mask_gray.shape[:2] != original_frame.shape[:2]:
                mask_gray = cv2.resize(mask_gray, (original_w, original_h))
            
            # 创建前景掩码和背景掩码
            # 将掩码转换为三通道，以便应用
            mask_3ch = cv2.merge([mask_gray, mask_gray, mask_gray])
            mask_3ch = mask_3ch / 255.0  # 归一化到0-1

            # 创建背景掩码 (反转)
            bg_mask_3ch = 1.0 - mask_3ch
            
            # 合成
            foreground = cv2.multiply(original_frame.astype(float), mask_3ch)
            bg = cv2.multiply(background.astype(float), bg_mask_3ch)
            
            # 组合前景和背景
            composite = cv2.add(foreground, bg).astype(np.uint8)
            
            # 写入输出视频
            output_writer.write(composite)
            
            frame_count += 1
        
        # 释放资源
        original_cap.release()
        mask_cap.release()
        output_writer.release()
        
        callback('complete', 100, "背景合成完成 (无音频)")
        print(f"背景合成完成!")
        print(f"输出文件: {output_path}")
        return output_path
        
    except Exception as e:
        error_msg = f"无音频背景合成处理出错: {e}"
        callback('error', 0, error_msg)
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        # 尝试释放资源
        try:
            if 'original_cap' in locals() and original_cap is not None:
                original_cap.release()
            if 'mask_cap' in locals() and mask_cap is not None:
                mask_cap.release()
            if 'output_writer' in locals() and output_writer is not None:
                output_writer.release()
        except Exception as cleanup_error:
            print(f"清理资源时出错: {cleanup_error}")
        
        return None

# 修改原有process_video函数为调用新拆分的函数
def extract_video(
    video_path,
    model_path,
    output_dir="./output",
    method="birefnet",
    image_size=(512, 512),
    batch_size=4,
    bg_path=None,
    bg_color=(0, 255, 0),
    callback=default_callback
):
    """
    统一的视频处理入口函数，支持多种模型和背景处理
    
    参数:
        video_path: 输入视频路径
        model_path: 模型路径，支持.pth和.safetensors格式
        output_dir: 输出目录，默认为./output
        method: 模型类型，支持'birefnet'或'ben2'
        image_size: 模型输入尺寸，格式为(width, height)
        batch_size: 批处理大小
        bg_path: 背景图片路径，如果为None则使用bg_color
        bg_color: 背景颜色RGB值，默认为绿色(0,255,0)
        callback: 回调函数，用于报告进度
        
    返回:
        tuple: (掩码视频路径, 合成视频路径)，处理失败时返回(None, None)
    """
    try:
        # 参数验证
        if not os.path.exists(video_path):
            error_msg = f"错误: 输入视频不存在: {video_path}"
            callback('error', 0, error_msg)
            print(error_msg)
            return None, None
            
        if not os.path.exists(model_path):
            error_msg = f"错误: 模型文件不存在: {model_path}"
            callback('error', 0, error_msg)
            print(error_msg)
            return None, None
            
        if method.lower() not in ["birefnet", "ben2"]:
            error_msg = f"错误: 不支持的方法: {method}"
            callback('error', 0, error_msg)
            print(error_msg)
            return None, None
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置文件名
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        model_name = os.path.splitext(os.path.basename(model_path))[0]
            
        # 处理背景名称
        bg_name = "default"
        if bg_path:
            if os.path.exists(bg_path):
                bg_name = os.path.splitext(os.path.basename(bg_path))[0]
            else:
                print(f"警告: 背景图片不存在: {bg_path}，将使用默认背景")
                bg_path = None
                
        output_mask_path = os.path.join(
            output_dir, 
            f"mask-{video_name}-{bg_name}-{model_name}-{image_size[0]}x{image_size[1]}.mp4"
        )
        output_composite_path = os.path.join(
            output_dir, 
            f"composite-{video_name}-{bg_name}-{model_name}-{image_size[0]}x{image_size[1]}.mp4"
        )
        
        # 根据方法选择处理函数
        if method.lower() == "ben2":
            if not HAS_BEN2:
                error_msg = "错误: 未安装BEN2模型，无法使用此方法"
                callback('error', 0, error_msg)
                print(error_msg)
                return None, None
                
            mask_path, composite_path = predict_video_mask_ben2(
                video_path=video_path,
                output_mask_path=output_mask_path,
                output_composite_path=output_composite_path,
                model_path=model_path,
                batch_size=batch_size,
                bg_color=bg_color,
                callback=callback
            )
        elif method.lower() == "birefnet":
            mask_path, composite_path = predict_video_mask_birefnet(
                video_path=video_path,
                output_mask_path=output_mask_path,
                output_composite_path=output_composite_path,
                model_path=model_path,
                image_size=image_size, 
                batch_size=batch_size, 
                bg_color=bg_color,
                callback=callback
            )
        
        # 应用自定义背景（如果指定）
        if bg_path and os.path.exists(bg_path) and mask_path and os.path.exists(mask_path):
            callback('processing', 85, f"应用自定义背景: {bg_path}")
            print(f"应用自定义背景: {bg_path}")
            
            # 生成自定义背景输出路径
            custom_bg_output_path = os.path.join(
                output_dir, 
                f"{prefix}custom_bg-{video_name}-{bg_name}-{model_name}-{image_size[0]}x{image_size[1]}{timestamp}.mp4"
            )
            
            # 应用自定义背景
            custom_bg_path = apply_custom_background(
                video_path=video_path,
                mask_path=mask_path,
                output_path=custom_bg_output_path,
                bg_path=bg_path,
                bg_color=bg_color,
                callback=callback
            )
            
            if custom_bg_path:
                # 更新合成视频路径为自定义背景版本
                composite_path = custom_bg_path
                
        # 检查输出文件是否都存在
        if mask_path and composite_path and os.path.exists(mask_path) and os.path.exists(composite_path):
            callback('complete', 100, "视频处理完成")
            print(f"视频处理完成")
            print(f"输出文件:")
            print(f"1. 掩码视频: {mask_path}")
            print(f"2. 合成视频: {composite_path}")
            return mask_path, composite_path
        else:
            missing_files = []
            if not mask_path or not os.path.exists(mask_path):
                missing_files.append(output_mask_path)
            if not composite_path or not os.path.exists(composite_path):
                missing_files.append(output_composite_path)
                
            warn_msg = f"警告: 以下输出文件未生成: {missing_files}"
            callback('warning', 95, warn_msg)
            print(warn_msg)
            
            # 返回存在的文件
            return (mask_path if mask_path and os.path.exists(mask_path) else None,
                    composite_path if composite_path and os.path.exists(composite_path) else None)
    
    except Exception as e:
        error_msg = f"视频处理出错: {e}"
        callback('error', 0, error_msg)
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='BiRefNet/BEN2视频抠图处理')
    parser.add_argument('--video', type=str, required=True, 
                        help='输入视频路径')
    parser.add_argument('--model', type=str, required=True, 
                        help='模型路径，支持.pth和.safetensors格式，自动识别lite和完整版')
    parser.add_argument('--output', type=str, default="", 
                        help='输出目录')
    parser.add_argument('--size', type=str, default="", 
                        help='处理分辨率，格式为width,height')
    parser.add_argument('--batch', type=int, default=0, 
                        help='批处理大小，0表示自动根据GPU内存决定')
    # 背景参数
    parser.add_argument('--bg', type=str, default=None,
                        help='背景图片路径，如果指定将使用该图片作为背景')
    parser.add_argument('--bg-color', type=str, default="0,255,0",
                        help='背景RGB颜色值，格式为r,g,b，例如 0,255,0 表示绿色')
    # 回调函数类型
    parser.add_argument('--callback-type', type=str, default="default", choices=['default', 'simple', 'verbose'],
                        help='回调函数类型: default, simple, verbose')
    args = parser.parse_args()
    
    # 根据参数选择回调函数
    if args.callback_type == 'simple':
        callback = simple_callback
    elif args.callback_type == 'verbose':
        callback = verbose_callback
    else:
        callback = default_callback
    
    # 提示必要的库
    if not HAS_SAFETENSORS and args.model.endswith('.safetensors'):
        print("警告: 您正在尝试加载.safetensors格式的模型，但未安装safetensors库")
        print("请运行: pip install safetensors")
        print("然后重新运行此脚本")
        sys.exit(1)
        
    if not HAS_MOVIEPY:
        print("警告: 未安装moviepy库，处理后的视频将没有声音")
        print("要保留原始视频的音频，请运行: pip install moviepy")
        print("然后重新运行此脚本")
        print("继续处理，但输出视频将没有声音...")
    else:
        print("已检测到moviepy库，将保留原始视频的音频轨道")
    
    
    # 解析背景颜色
    bg_color = (0, 255, 0)  # 默认绿色
    try:
        bg_color = tuple(map(int, args.bg_color.split(',')))
        if len(bg_color) != 3:
            print(f"警告: 背景颜色格式错误，应为'r,g,b'，使用默认绿色(0,255,0)")
            bg_color = (0, 255, 0)
    except:
        print(f"警告: 无法解析背景颜色'{args.bg_color}'，使用默认绿色(0,255,0)")
        bg_color = (0, 255, 0)
    
    # 解析处理分辨率
    try:
        width, height = map(int, args.size.split(','))
        image_size = (width, height)
        print(f"设置处理分辨率: {image_size}")
    except:
        image_size = model_config[args.model]["image_size"]
        print(f"使用默认处理分辨率: {image_size}")
    
    # 输出目录
    output_dir = args.output if args.output else os.path.dirname(args.video)
    
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
    
    # 调用统一入口函数进行处理
    mask_path, composite_path = extract_video(
        video_path=args.video,
        model_path=model_config[args.model]["model_path"],
        output_dir=output_dir,
        method=model_config[args.model]["method"],
        image_size=image_size,
        batch_size=batch_size,
        bg_path=args.bg,
        bg_color=bg_color,
        callback=callback
    )
    
    if mask_path and composite_path:
        print("处理完成!")
        print(f"输出文件:")
        print(f"1. 掩码视频: {mask_path}")
        print(f"2. 合成视频: {composite_path}")
    else:
        print("处理失败")
        sys.exit(1)

'''
python test_gpu.py --model "/root/gpufree-data/BiRefNet-DIS-epoch_590.pth" --video "/root/gpufree-data/111.mp4" --size "1024,1024" --batch 8

python test_gpu_all.py --method ben2 --model "/root/gpufree-data/models/BiRefNet.safetensors" --video "/root/gpufree-data/samplevideo/111.mp4" --size "1024,1024" --batch 8 --output "/root/gpufree-data/output/"

python video_extractor.py --model "BiRefNet" --video "/root/gpufree-data/samplevideo/111.mp4" --size "1024,1024" --batch 8 --output "/root/gpufree-data/output/"
python video_extractor.py --model "BEN2_Base" --video "/root/gpufree-data/samplevideo/111.mp4" --size "1024,1024" --batch 8 --output "/root/gpufree-data/output/"

python video_extractor.py --model "BiRefNet_lite-general-2K" --video "/root/gpufree-data/samplevideo/111.mp4" --size "1024,1024" --batch 8 --output "/root/gpufree-data/output/"
''' 