# -*- coding: utf-8 -*-
import os
import time
import uuid
import json
import threading
import requests
import torch
import re
from urllib.parse import urlparse
from typing import Dict, List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn
import traceback
from video_extractor import extract_video, apply_custom_background, HAS_MOVIEPY
from logger_handler import get_logger


POD_IP = os.environ.get("pod_ip") or "100.116.215.243"

# 初始化日志
logger = get_logger(f"server-{INSTANCE_ID}")
# 统一的日志记录函数
def log_with_task_id(task_id, message, level='info'):
    """
    使用任务ID记录日志
    
    参数:
        task_id: 任务ID
        message: 日志消息
        level: 日志级别 (debug, info, warning, error, critical)
        logger_name: 日志记录器名称，不提供则使用__name__
    """
    extra = {'task_id': task_id if task_id else 'no-task-id'}
    
    if level == 'debug':
        logger.debug(message, extra=extra)
    elif level == 'info':
        logger.info(message, extra=extra)
    elif level == 'warning':
        logger.warning(message, extra=extra)
    elif level == 'error':
        logger.error(message, extra=extra)
    elif level == 'critical':
        logger.critical(message, extra=extra)
    else:
        logger.info(message, extra=extra) 

app = FastAPI(title="AI视频处理服务")

# 存储任务状态的字典
tasks_status = {}

# 后端回调地址 (可配置)
# BACKEND_URL="https://to74zigu-nx6sqm6b-6001.zjrestapi.gpufree.cn:8443"
BACKEND_URL="http://100.116.215.243:6001"
# 当前AI server服务对外地址，按需修改
# CURRENT_WORKER_URL = f"https://{INSTANCE_ID}-6002.zjrestapi.gpufree.cn:8443"
CURRENT_WORKER_URL = f"http://{POD_IP}:6002"

# 判断两个URL是否属于同一台机器
def is_same_host(url1, url2):
    """
    判断两个URL是否属于同一台机器
    支持以下格式:
    1. 常规IP: 1.2.3.4:9090
    2. 容器化域名: to74zigu-nx6sqm6b-6001.zjrestapi.gpufree.cn
    """
    # 解析URL获取主机名和端口
    parsed1 = urlparse(url1)
    parsed2 = urlparse(url2)
    
    host1 = parsed1.netloc.split(':')[0]
    host2 = parsed2.netloc.split(':')[0]
    
    # 如果是IP地址格式，直接比较IP部分
    ip_pattern = re.compile(r'\d+\.\d+\.\d+\.\d+')
    if ip_pattern.match(host1) and ip_pattern.match(host2):
        return host1 == host2
    
    # 针对容器化域名格式: to74zigu-nx6sqm6b-6001.zjrestapi.gpufree.cn
    container_pattern = re.compile(r'(.*?)-\d+\.(.*)$')
    match1 = container_pattern.match(host1)
    match2 = container_pattern.match(host2)
    
    if match1 and match2:
        # 提取前缀和后缀，忽略端口部分
        prefix1, suffix1 = match1.groups()
        prefix2, suffix2 = match2.groups()
        return prefix1 == prefix2 and suffix1 == suffix2
    
    # 其他情况直接比较主机名
    return host1 == host2

# 获取用于访问后端的URL
def get_backend_url(target_url, endpoint=None, port=6001):
    """
    根据目标URL判断是否为同一台机器，是则返回localhost地址
    
    参数:
        target_url: 目标服务器URL
        endpoint: API端点路径 (可选)
        port: 如果是本地访问，使用的端口号
    """
    if is_same_host(CURRENT_WORKER_URL, target_url):
        # 同一台机器，使用localhost
        url = f"http://localhost:{port}"
        log_with_task_id("system", f"检测到同一台机器，使用本地地址: {url}")
    else:
        # 不同机器，使用原URL
        url = target_url
        
    # 添加端点路径
    if endpoint:
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
        url = url + endpoint
    
    return url

# 心跳相关配置
ENABLE_HEARTBEAT = True  # 是否启用心跳
BACKEND_PORT = 6001      # 后端服务端口

# 生成回调和心跳URL
BACKEND_CALLBACK_URL = get_backend_url(BACKEND_URL, "/api/task/callback", BACKEND_PORT)
BACKEND_HEARTBEAT_URL = get_backend_url(BACKEND_URL, "/api/tasks/interface/heartbeat", BACKEND_PORT)
BACKEND_IDENTIFICATION = "wingerboy"  # 身份标识，按需修改
HEARTBEAT_INTERVAL = 60  # 心跳间隔秒

# 打印重要的URL配置信息
log_with_task_id("system", f"后端回调URL: {BACKEND_CALLBACK_URL}")
log_with_task_id("system", f"心跳URL: {BACKEND_HEARTBEAT_URL}")

BG_SAVE_PATH="/root/gpufree-share/videos/background"
ORIGIN_VIDEO_PATH="/root/gpufree-share/videos/originvideo"
FORE_VIDEO_PATH="/root/gpufree-share/videos/forevideo"
MASK_VIDEO_PATH="/root/gpufree-share/videos/maskvideo"
OUTPUT_PATH="/root/gpufree-share/videos/output"

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

# 定义请求模型
class VideoSegmentTaskRequest(BaseModel):
    taskId: str
    videoPath: str
    foregroundPath: Optional[str] = None
    backgroundPath: Optional[str] = None
    modelName: str
    modelAlias: Optional[str] = None
    callbackUrl: Optional[str] = None
    workerUrl: Optional[str] = None

# 视频分割任务状态枚举
class TaskStatus:
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def update_task_status(task_id: str, status: str, progress: float = 0, message: str = ""):
    """更新任务状态并通知Java后端"""
    tasks_status[task_id] = {
        "status": status,
        "progress": progress,
        "message": message,
        "updated_at": time.time()
    }
    
    # 向Java后端报告进度
    try:
        payload = {
            "Identification": BACKEND_IDENTIFICATION,
            "taskId": task_id,
            "workerUrl": CURRENT_WORKER_URL,
            "status": status,
            "progress": progress,
            "message": message
        }
        
        log_with_task_id(task_id, f"向后端报告进度: {json.dumps(payload)}")
        
        # 实际环境中取消注释下面的代码
        response = requests.post(BACKEND_CALLBACK_URL, json=payload, timeout=10)
        if response.status_code != 200:
            log_with_task_id(task_id, f"向后端报告进度失败: {response.text}", 'error')
    except Exception as e:
        log_with_task_id(task_id, f"向Java后端报告进度出错: {str(e)}", 'error')

# 添加进度区间管理器
class ProgressManager:
    """
    进度区间管理器，用于管理整个任务的进度分配和转换
    将各子流程的进度(0-100%)映射到全局统一的进度范围内
    """
    def __init__(self, task_id):
        self.task_id = task_id
        self.stages = []  # [(stage_name, start_percent, end_percent, weight)]
        self.current_stage = 0
        self.stage_progress = 0
        self.total_progress = 0
        self.last_reported_progress = 0
        
    def add_stage(self, name, start_percent, end_percent, weight=1.0):
        """添加处理阶段及其进度区间"""
        self.stages.append((name, start_percent, end_percent, weight))
        
    def set_stage(self, stage_index):
        """设置当前处理阶段"""
        if 0 <= stage_index < len(self.stages):
            self.current_stage = stage_index
            self.stage_progress = 0
            stage_name = self.stages[stage_index][0]
            log_with_task_id(self.task_id, f"进入处理阶段: {stage_name}")
            return True
        return False
        
    def next_stage(self):
        """进入下一处理阶段"""
        return self.set_stage(self.current_stage + 1)
        
    def update_progress(self, progress, message=""):
        """
        更新当前阶段的进度并转换为全局进度
        
        参数:
            progress: 当前阶段的进度(0-100%)
            message: 进度消息
        """
        # 确保进度在有效范围内
        progress = max(0, min(100, progress))
        self.stage_progress = progress
        
        # 获取当前阶段信息
        if self.current_stage < len(self.stages):
            stage_name, start, end, _ = self.stages[self.current_stage]
            
            # 计算全局进度
            stage_contribution = (progress / 100.0) * (end - start)
            self.total_progress = start + stage_contribution
            
            # 确保进度始终向前，避免后退
            self.total_progress = max(self.total_progress, self.last_reported_progress)
            
            # 限制进度突变幅度
            if self.total_progress - self.last_reported_progress > 5 and self.last_reported_progress > 0:
                self.total_progress = self.last_reported_progress + 5
                
            # 更新上次报告的进度
            self.last_reported_progress = self.total_progress
            
            # 构建进度消息
            if not message:
                message = f"{stage_name}: {progress:.1f}%"
            else:
                message = f"{stage_name}: {message}"
                
            return self.total_progress, message
        
        return 0, message
    
    def create_stage_callback(self, status_handler):
        """
        为当前阶段创建进度回调适配器
        
        参数:
            status_handler: 负责处理状态的函数，接收(status, progress, message)
        """
        def progress_adapter(status, progress, message=""):
            # 将子流程进度适配到当前阶段的整体进度区间
            if status in ('error', 'failed'):
                # 错误状态直接传递
                status_handler(status, progress, message)
                return
                
            if status in ('complete', 'completed') and progress >= 100:
                # 完成状态，自动进入下一阶段
                self.next_stage()
                
            # 更新并转换进度
            global_progress, enhanced_message = self.update_progress(progress, message)
            
            # 调用状态处理函数
            status_handler(status, global_progress, enhanced_message)
        
        return progress_adapter

# 修改现有进度回调工厂函数
def create_progress_callback(task_id):
    """创建针对特定任务的进度回调函数"""
    
    # 内部状态处理函数
    def handle_status(status, progress, message=""):
        # 记录进度日志
        log_with_task_id(task_id, f"任务进度: {status} - {progress:.1f}% - {message}")
        
        # 将status转换为TaskStatus
        task_status = TaskStatus.PROCESSING
        if status == 'error' or status == 'failed':
            task_status = TaskStatus.FAILED
        elif status == 'complete' or status == 'completed':
            if progress >= 100:
                task_status = TaskStatus.COMPLETED
        
        # 限制进度范围在5-95之间，预留开始和结束的余量
        if progress < 5:
            progress = 5
        elif progress > 95 and task_status != TaskStatus.COMPLETED:
            progress = 95
            
        # 更新并通知后端
        update_task_status(task_id, task_status, progress, message)
    
    return handle_status

def process_video_segment(task_id, task_params):
    """处理视频分割任务"""
    # 提取参数
    origin_video_path = task_params["origin_video_path"]
    mask_video_path = task_params["mask_video_path"]
    foreground_video_path = task_params["foreground_video_path"]
    composite_video_path = task_params["composite_video_path"]
    bg_path = task_params["bg_path"]
    model_name = task_params["model_name"]
    image_size = task_params["image_size"]
    
    # 初始化返回值
    result = {
        "mask_video_path": mask_video_path,
        "composite_video_path": None
    }
    
    # 创建进度管理器，并配置处理阶段
    progress_mgr = ProgressManager(task_id)
    # 定义处理阶段和进度分配
    progress_mgr.add_stage("准备阶段", 0, 5)       # 0-5%: 初始检查和准备
    progress_mgr.add_stage("视频分割", 5, 65)      # 5-65%: 视频提取和掩码生成(最耗时)
    progress_mgr.add_stage("背景合成", 65, 95)     # 65-95%: 背景应用
    progress_mgr.add_stage("完成阶段", 95, 100)    # 95-100%: 清理和完成
    
    # 创建基础回调处理函数
    base_callback = create_progress_callback(task_id)
    
    # 更新任务状态为处理中，开始第一阶段
    progress_mgr.set_stage(0)
    base_callback('processing', progress_mgr.update_progress(0, "正在准备任务资源...")[0], "正在准备任务资源...")
    
    try:        
        # 更新第一阶段完成进度
        base_callback('processing', progress_mgr.update_progress(100, "准备工作完成")[0], "准备工作完成")
        
        # 检查掩码视频是否已存在
        skip_segmentation = False
        if os.path.exists(mask_video_path):
            log_with_task_id(task_id, f"掩码视频已存在，跳过分割阶段: {mask_video_path}")
            skip_segmentation = True
            # 直接进入背景合成阶段
            progress_mgr.set_stage(2)
            base_callback('processing', progress_mgr.update_progress(0, "已找到现有掩码视频，跳过分割阶段")[0], "已找到现有掩码视频，跳过分割阶段")
        else:
            # 进入视频分割阶段
            progress_mgr.set_stage(1)
            base_callback('processing', progress_mgr.update_progress(0, "正在进行视频分割...")[0], "正在进行视频分割...")
        
        # 阶段1: 视频分割，生成掩码视频
        if not skip_segmentation:
            try:
                log_with_task_id(task_id, f"开始提取视频: {origin_video_path}")
                model_path = model_config[model_name]["model_path"]
                method = model_config[model_name].get("method", "birefnet")
                batch_size = model_config[model_name].get("max_batch_size", 4)
                log_with_task_id(task_id, f"使用模型 {model_name}, 方法 {method}")
                
                # 获取当前阶段的进度适配器
                extraction_callback = progress_mgr.create_stage_callback(base_callback)
                
                # 执行视频分割
                extract_video(
                    video_path=origin_video_path,
                    model_path=model_path,
                    foreground_video_path=foreground_video_path,
                    output_mask_path=mask_video_path,
                    method=method,
                    image_size=image_size,
                    batch_size=batch_size,
                    callback=extraction_callback
                )
                
                # 检查掩码是否成功创建
                if not os.path.exists(mask_video_path):
                    error_msg = f"视频分割失败，无法创建掩码视频: {mask_video_path}"
                    log_with_task_id(task_id, error_msg, 'error')
                    base_callback('error', 100, error_msg)
                    return result
                
                log_with_task_id(task_id, f"视频分割完成，掩码生成在: {mask_video_path}")
                # 确保视频分割阶段进度为100%
                base_callback('processing', progress_mgr.update_progress(100, "掩码视频生成完成")[0], "掩码视频生成完成")
                
            except Exception as e:
                error_msg = f"视频分割过程中出错: {str(e)}"
                log_with_task_id(task_id, error_msg, 'error')
                log_with_task_id(task_id, traceback.format_exc(), 'error')
                base_callback('error', 100, error_msg)
                return result
        
        # 阶段2: 应用背景
        try:
            # 如果没有跳过分割阶段，需要主动设置当前阶段
            if not skip_segmentation:
                progress_mgr.set_stage(2)
                base_callback('processing', progress_mgr.update_progress(0, "正在应用自定义背景...")[0], "正在应用自定义背景...")
            
            # 创建背景合成阶段的进度适配器
            compositing_callback = progress_mgr.create_stage_callback(base_callback)
            
            # 应用背景
            composite_path = apply_custom_background(
                video_path=origin_video_path,
                mask_video_path=mask_video_path,
                background_path=bg_path,
                output_composite_path=composite_video_path,
                include_audio=True,
                callback=compositing_callback
            )
            
            result["composite_video_path"] = composite_path
            
            # 检查合成视频是否成功创建
            if not os.path.exists(composite_video_path):
                error_msg = f"应用背景失败，无法创建合成视频: {composite_video_path}"
                log_with_task_id(task_id, error_msg, 'error')
                base_callback('error', 100, error_msg)
                return result
            
            log_with_task_id(task_id, f"背景应用完成，合成视频生成在: {composite_path}")
            # 确保背景合成阶段进度为100%
            base_callback('processing', progress_mgr.update_progress(100, "背景合成完成")[0], "背景合成完成")
            
        except Exception as e:
            error_msg = f"应用背景过程中出错: {str(e)}"
            log_with_task_id(task_id, error_msg, 'error')
            log_with_task_id(task_id, traceback.format_exc(), 'error')
            base_callback('error', 100, error_msg)
            return result
        
        # 进入完成阶段
        progress_mgr.set_stage(3)
        base_callback('processing', progress_mgr.update_progress(50, "正在完成任务...")[0], "正在完成任务...")
        
        # 任务完成
        base_callback('completed', 100, "视频分割与背景合成完成")
        
    except Exception as e:
        error_msg = f"任务处理过程中出错: {str(e)}"
        log_with_task_id(task_id, error_msg, 'error')
        log_with_task_id(task_id, traceback.format_exc(), 'error')
        base_callback('error', 100, error_msg)
    
    return result

@app.post("/api/video/segment")
async def video_segment_task_executor(
    request: VideoSegmentTaskRequest,
    background_tasks: BackgroundTasks
):
    """启动视频分割任务"""
    task_id = request.taskId
    
    # 如果没有提供task_id，生成一个
    if not task_id:
        log_with_task_id("no-task-id", "请求中无taskId", 'error')
        return {
            "taskId": "error",
            "status": "fail",
            "message": "错误: 请求中无taskId"
        }
    
    try:
        # 记录开始处理的任务
        log_with_task_id(task_id, f"接收到视频分割任务请求")
        log_with_task_id(task_id, f"请求参数: 视频={request.videoPath}, 模型={request.modelName}, 背景={request.backgroundPath}")
        
        # 检查请求参数
        # 检查图片格式
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if not request.backgroundPath or not any(request.backgroundPath.lower().endswith(ext) for ext in valid_extensions):
            log_with_task_id(task_id, f"警告: 背景图片格式可能不受支持或未提供: {request.backgroundPath}", 'error')
            return {
                "taskId": task_id,
                "status": "fail",
                "message": "错误: 背景图片格式可能不受支持或未提供"
            }
        
        # 参数验证
        if not os.path.exists(request.videoPath):
            error_msg = f"错误: 输入视频不存在: {request.videoPath}"
            log_with_task_id(task_id, error_msg, 'error')
            return {
                "taskId": task_id,
                "status": "fail",
                "message": error_msg
            }
            
        # 检查模型是否存在
        if request.modelName not in model_config:
            error_msg = f"错误: 未知的模型名称: {request.modelName}"
            log_with_task_id(task_id, error_msg, 'error')
            return {
                "taskId": task_id,
                "status": "fail",
                "message": error_msg
            }
            
        # 检查模型是否可用
        model_info = model_config[request.modelName]
        if model_info.get("available") is False:
            error_msg = f"错误: 模型暂时不可用: {request.modelName}. 原因: {model_info.get('reason', '未知')}"
            log_with_task_id(task_id, error_msg, 'error')
            return {
                "taskId": task_id,
                "status": "fail",
                "message": error_msg
            }
            
        # 确保mask_video_path有值
        mask_video_path = request.foregroundPath
        # 构建默认路径
        video_name = os.path.splitext(os.path.basename(request.videoPath))[0]
        bg_name = os.path.splitext(os.path.basename(request.backgroundPath))[0]
        model_name = request.modelName
        image_size = model_config[model_name].get("image_size", (1024, 1024))
        if not mask_video_path:
            
            mask_video_path = os.path.join(
                MASK_VIDEO_PATH, 
                f"mask-{video_name}-{bg_name}-{model_name}-{image_size[0]}x{image_size[1]}.mp4"
            )
            log_with_task_id(task_id, f"使用默认掩码路径: {mask_video_path}")
        # 构建合成视频输出路径
        composite_video_path = os.path.join(
            OUTPUT_PATH, 
            f"composite-{video_name}-{bg_name}-{model_name}-{image_size[0]}x{image_size[1]}.mp4"
        )
        foreground_video_path = os.path.join(
            FORE_VIDEO_PATH, 
            f"foreground-{video_name}-{bg_name}-{model_name}-{image_size[0]}x{image_size[1]}.mp4"
        )
        log_with_task_id(task_id, f"使用默认合成视频路径: {composite_video_path}")
        log_with_task_id(task_id, f"使用默认前景视频路径: {foreground_video_path}")
        # 确保目录存在
        os.makedirs(os.path.dirname(mask_video_path), exist_ok=True)
        os.makedirs(os.path.dirname(foreground_video_path), exist_ok=True)
        os.makedirs(os.path.dirname(composite_video_path), exist_ok=True)
        
        # 如果提供了回调URL，更新全局回调URL
        if request.callbackUrl:
            global BACKEND_CALLBACK_URL
            BACKEND_CALLBACK_URL = request.callbackUrl
            log_with_task_id(task_id, f"更新回调URL: {BACKEND_CALLBACK_URL}")
        
        # 如果提供了worker地址，更新全局worker地址
        if request.workerUrl:
            global CURRENT_WORKER_URL
            CURRENT_WORKER_URL = request.workerUrl
            log_with_task_id(task_id, f"更新worker地址: {CURRENT_WORKER_URL}")
        
        # 准备任务参数
        task_params = {
            "task_id": task_id,
            "origin_video_path": request.videoPath,
            "mask_video_path": mask_video_path,
            "foreground_video_path": foreground_video_path,
            "composite_video_path": composite_video_path,
            "bg_path": request.backgroundPath,
            "model_name": request.modelName,
            "image_size": image_size
        }
        
        # 更新任务状态为等待中
        update_task_status(task_id, TaskStatus.PENDING, 0, "任务已接收，等待处理")
        
        # 在后台启动任务
        background_tasks.add_task(process_video_segment, task_id, task_params)
        
        # 立即返回响应
        log_with_task_id(task_id, f"视频分割任务已接收并开始处理: {task_params}")
        return {
            "taskId": task_id,
            "status": "accepted",
            "message": "视频分割任务已接收并开始处理",
            "maskVideoPath": mask_video_path,
            "compositeVideoPath": composite_video_path
        }
    except Exception as e:
        error_msg = f"处理请求时出错: {str(e)}"
        log_with_task_id(task_id, error_msg, 'error')
        log_with_task_id(task_id, traceback.format_exc(), 'error')
        return {
            "taskId": task_id,
            "status": "fail",
            "message": error_msg
        }

@app.get("/api/task/{task_id}/status")
async def get_task_status(task_id: str):
    """获取任务的当前状态"""
    if task_id in tasks_status:
        log_with_task_id(task_id, f"查询任务状态: {tasks_status[task_id]}")
        return {
            "taskId": task_id,
            "status": tasks_status[task_id]["status"],
            "progress": tasks_status[task_id]["progress"],
            "message": tasks_status[task_id]["message"],
            "updated": tasks_status[task_id]["updated_at"]
        }
    else:
        log_with_task_id(task_id, "未找到任务状态", 'warning')
        return {
            "taskId": task_id,
            "status": "unknown",
            "message": "未找到任务"
        }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    log_with_task_id("health", "执行健康检查")
    return {"status": "ok", "service": "AI视频处理服务"}

def heartbeat_loop():
    """心跳线程，定期向后端发送心跳包"""
    if not ENABLE_HEARTBEAT:
        log_with_task_id("heartbeat", "心跳功能已禁用")
        return
        
    log_with_task_id("heartbeat", f"心跳服务启动 - URL: {BACKEND_HEARTBEAT_URL}")
    
    while True:
        try:
            # 按照测试案例格式构建payload
            payload = {
                "interfaceAddress": CURRENT_WORKER_URL,
                "Identification": BACKEND_IDENTIFICATION
            }
            log_with_task_id("heartbeat", f"发送心跳: {payload}")
            
            # 发送请求
            resp = requests.post(BACKEND_HEARTBEAT_URL, json=payload, timeout=10)
            
            # 处理响应
            if resp.status_code == 200:
                log_with_task_id("heartbeat", f"心跳成功: {resp.text}")
            else:
                log_with_task_id("heartbeat", f"心跳失败: {resp.status_code} {resp.text}", 'warning')
        except Exception as e:
            log_with_task_id("heartbeat", f"心跳异常: {e}", 'warning')
        
        # 等待下一次心跳
        time.sleep(HEARTBEAT_INTERVAL)

# 启动心跳线程
threading.Thread(target=heartbeat_loop, daemon=True).start()

if __name__ == "__main__":
    # 服务器配置
    if not HAS_MOVIEPY:
        log_with_task_id("server", "警告: 未安装moviepy库，输出视频将没有音频")
        
    host = os.environ.get("AI_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("AI_SERVER_PORT", "6002"))
    
    # 启动服务器
    uvicorn.run(app, host=host, port=port)
