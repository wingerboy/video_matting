import os
import time
import uuid
import json
import logging
import threading
import requests
import torch
from typing import Dict, List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn
import traceback
from video_extractor import extract_video, apply_custom_background, default_callback, HAS_MOVIEPY

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI视频处理服务")

# 存储任务状态的字典
tasks_status = {}

# Java回调地址 (可配置)
JAVA_CALLBACK_URL="http://localhost:6000/api/task/update"
BG_SAVE_PATH="/root/gpufree-data/videos/background"
ORIGIN_VIDEO_PATH="/root/gpufree-data/videos/originvideo"
FORE_VIDEO_PATH="/root/gpufree-data/videos/forevideo"
MASK_VIDEO_PATH="/root/gpufree-data/videos/maskvideo"
OUTPUT_PATH="/root/gpufree-data/videos/output"

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
    task_id: str
    origin_video_path: str
    mask_video_path: str
    bg_path: str
    model_name: str

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
            "taskId": task_id,
            "status": status,
            "progress": progress,
            "message": message
        }
        
        logger.info(f"向Java后端报告进度: {json.dumps(payload)}")
        
        # 实际环境中取消注释下面的代码
        # response = requests.post(JAVA_CALLBACK_URL, json=payload)
        # if response.status_code != 200:
        #     logger.error(f"向Java后端报告进度失败: {response.text}")
    except Exception as e:
        logger.error(f"向Java后端报告进度出错: {str(e)}")

# 自定义进度回调函数
def create_progress_callback(task_id):
    """创建针对特定任务的进度回调函数"""
    def progress_callback(status, progress, message=""):
        # 将status转换为TaskStatus
        task_status = TaskStatus.PROCESSING
        if status == 'complete':
            task_status = TaskStatus.COMPLETED
        elif status == 'error':
            task_status = TaskStatus.FAILED
        
        # 更新并通知Java后端
        update_task_status(task_id, task_status, progress, message)
    
    return progress_callback

def process_video_segment(task_id, task_params):
    """处理视频分割任务"""
    # 提取参数
    origin_video_path = task_params["origin_video_path"]
    mask_video_path = task_params["mask_video_path"]
    bg_path = task_params["bg_path"]
    model_name = task_params["model_name"]
    
    # 初始化返回值
    result = {
        "mask_video_path": mask_video_path,
        "composite_video_path": None
    }
    
    # 更新任务状态为处理中
    update_task_status(task_id, TaskStatus.PROCESSING, 5, "正在准备任务资源...")
    
    try:
        # 检查输入文件是否存在
        if not os.path.exists(origin_video_path):
            error_msg = f"输入视频不存在: {origin_video_path}"
            logger.error(error_msg)
            update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
            return result
        
        if not os.path.exists(bg_path):
            error_msg = f"背景图片不存在: {bg_path}"
            logger.error(error_msg)
            update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
            return result
        
        # 检查模型配置是否存在
        if model_name not in model_config:
            error_msg = f"未知的模型名称: {model_name}"
            logger.error(error_msg)
            update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
            return result
        
        # 创建进度回调函数
        def progress_callback(status, progress, message=""):
            logger.info(f"任务 {task_id}: {status} - {progress}% - {message}")
            task_status = TaskStatus.PROCESSING
            if status == "failed":
                task_status = TaskStatus.FAILED
            elif status == "completed":
                if progress >= 100:
                    task_status = TaskStatus.COMPLETED
            
            # 限制进度范围在5-95之间，预留开始和结束的余量
            if progress < 5:
                progress = 5
            elif progress > 95 and task_status != TaskStatus.COMPLETED:
                progress = 95
                
            update_task_status(task_id, task_status, progress, message)
        
        # 构建输出路径
        video_name = os.path.splitext(os.path.basename(origin_video_path))[0]
        bg_name = os.path.splitext(os.path.basename(bg_path))[0]
        image_size = model_config[model_name].get("image_size", (1024, 1024))
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(mask_video_path), exist_ok=True)
        
        # 构建合成视频输出路径
        composite_video_path = os.path.join(
            OUTPUT_PATH, 
            f"composite-{video_name}-{bg_name}-{model_name}-{image_size[0]}x{image_size[1]}.mp4"
        )
        foreground_video_path = os.path.join(
            OUTPUT_PATH, 
            f"foreground-{video_name}-{bg_name}-{model_name}-{image_size[0]}x{image_size[1]}.mp4"
        )
        
        # 检查掩码视频是否已存在
        skip_segmentation = False
        if os.path.exists(mask_video_path):
            logger.info(f"掩码视频已存在，跳过分割阶段: {mask_video_path}")
            skip_segmentation = True
            update_task_status(task_id, TaskStatus.PROCESSING, 50, "已找到现有掩码视频，跳过分割阶段")
        
        # 阶段1: 视频分割，生成掩码视频
        if not skip_segmentation:
            update_task_status(task_id, TaskStatus.PROCESSING, 10, "正在进行视频分割...")
            
            try:
                logger.info(f"开始提取视频: {origin_video_path}")
                model_path = model_config[model_name]["model_path"]
                method = model_config[model_name].get("method", "birefnet")
                logger.info(f"使用模型 {model_name}, 方法 {method}")
                
                # 提取视频，生成掩码
                mask_path = extract_video(
                    video_path=origin_video_path,
                    model_path=model_path,
                    foreground_video_path=foreground_video_path,
                    output_mask_path=mask_video_path,
                    method=method,
                    image_size=image_size,
                    batch_size=4,
                    callback=progress_callback
                )
                
                # 检查掩码是否成功创建
                if not os.path.exists(mask_video_path):
                    error_msg = f"视频分割失败，无法创建掩码视频: {mask_video_path}"
                    logger.error(error_msg)
                    update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
                    return result
                
                logger.info(f"视频分割完成，掩码生成在: {mask_path}")
                
            except Exception as e:
                error_msg = f"视频分割过程中出错: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
                return result
        
        # 阶段2: 应用背景
        try:
            update_task_status(task_id, TaskStatus.PROCESSING, 60, "正在应用自定义背景...")
            
            # 应用背景
            composite_path = apply_custom_background(
                video_path=origin_video_path,
                mask_video_path=mask_video_path,
                background_path=bg_path,
                output_composite_path=composite_video_path,
                include_audio=True,
                callback=progress_callback
            )
            
            result["composite_video_path"] = composite_path
            
            # 检查合成视频是否成功创建
            if not os.path.exists(composite_video_path):
                error_msg = f"应用背景失败，无法创建合成视频: {composite_video_path}"
                logger.error(error_msg)
                update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
                return result
            
            logger.info(f"背景应用完成，合成视频生成在: {composite_path}")
            
        except Exception as e:
            error_msg = f"应用背景过程中出错: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
            return result
        
        # 任务完成
        update_task_status(task_id, TaskStatus.COMPLETED, 100, "视频分割与背景合成完成")
        
    except Exception as e:
        error_msg = f"任务处理过程中出错: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
    
    return result

@app.post("/api/video/segment")
async def video_segment_task_executor(
    request: VideoSegmentTaskRequest,
    background_tasks: BackgroundTasks
):
    """启动视频分割任务"""
    task_id = request.task_id
    
    # 如果没有提供task_id，生成一个
    if not task_id:
        logger.error(f"请求中无task_id")
        return {
            "taskId": "error",
            "status": "fail",
            "message": "错误: 请求中无task_id"
        }
    
    try:
        # 记录开始处理的任务
        logger.info(f"接收到视频分割任务请求: {task_id}")
        logger.info(f"请求参数: 视频={request.origin_video_path}, 模型={request.model_name}, 背景={request.bg_path}")
        
        # 检查请求参数
        # 检查图片格式
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if not any(request.bg_path.lower().endswith(ext) for ext in valid_extensions):
            logger.error(f"警告: 背景图片格式可能不受支持: {request.bg_path}")
            return {
                "taskId": task_id,
                "status": "fail",
                "message": "错误: 背景图片格式可能不受支持"
            }
        
        # 参数验证
        if not os.path.exists(request.origin_video_path):
            error_msg = f"错误: 输入视频不存在: {request.origin_video_path}"
            logger.error(error_msg)
            return {
                "taskId": task_id,
                "status": "fail",
                "message": error_msg
            }
            
        # 检查模型是否存在
        if request.model_name not in model_config:
            error_msg = f"错误: 未知的模型名称: {request.model_name}"
            logger.error(error_msg)
            return {
                "taskId": task_id,
                "status": "fail",
                "message": error_msg
            }
            
        # 确保mask_video_path有值
        mask_video_path = request.mask_video_path
        if not mask_video_path:
            # 构建默认路径
            video_name = os.path.splitext(os.path.basename(request.origin_video_path))[0]
            bg_name = os.path.splitext(os.path.basename(request.bg_path))[0]
            model_name = request.model_name
            image_size = model_config[model_name].get("image_size", (1024, 1024))
            
            mask_video_path = os.path.join(
                MASK_VIDEO_PATH, 
                f"mask-{video_name}-{bg_name}-{model_name}-{image_size[0]}x{image_size[1]}.mp4"
            )
            logger.info(f"使用默认掩码路径: {mask_video_path}")
            
        # 确保目录存在
        os.makedirs(os.path.dirname(mask_video_path), exist_ok=True)
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        # 准备任务参数
        task_params = {
            "task_id": task_id,
            "origin_video_path": request.origin_video_path,
            "mask_video_path": mask_video_path,
            "bg_path": request.bg_path,
            "model_name": request.model_name
        }
        
        # 更新任务状态为等待中
        update_task_status(task_id, TaskStatus.PENDING, 0, "任务已接收，等待处理")
        
        # 在后台启动任务
        background_tasks.add_task(process_video_segment, task_id, task_params)
        
        # 立即返回响应
        return {
            "taskId": task_id,
            "status": "accepted",
            "message": "视频分割任务已接收并开始处理",
            "maskVideoPath": mask_video_path
        }
    except Exception as e:
        error_msg = f"处理请求时出错: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            "taskId": task_id,
            "status": "fail",
            "message": error_msg
        }

@app.get("/api/task/{task_id}/status")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail=f"找不到任务 {task_id}")
    
    return tasks_status[task_id]

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "AI视频处理服务"}

if __name__ == "__main__":
    # 服务器配置
    if not HAS_MOVIEPY:
        logger.warning("警告: 未安装moviepy库，输出视频将没有音频")
        
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "6001"))
    
    # 启动服务器
    uvicorn.run(app, host=host, port=port)
