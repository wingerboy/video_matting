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
class TaskIDFilter(logging.Filter):
    """
    添加任务ID到日志记录的过滤器
    """
    def __init__(self, name=''):
        super().__init__(name)
        self.task_id = ''
    
    def filter(self, record):
        record.task_id = getattr(record, 'task_id', 'no-task-id')
        return True

# 创建过滤器实例
task_id_filter = TaskIDFilter()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(task_id)s] - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.addFilter(task_id_filter)

app = FastAPI(title="AI视频处理服务")

# 存储任务状态的字典
tasks_status = {}

# Java回调地址 (可配置)
BACKEND_CALLBACK_URL="https://to74zigu-nx6sqm6b-6001.zjrestapi.gpufree.cn:8443/api/task/callback"
# 心跳相关配置
ENABLE_HEARTBEAT = True  # 是否启用心跳
HEARTBEAT_URL = "https://to74zigu-nx6sqm6b-6001.zjrestapi.gpufree.cn:8443/api/tasks/interface/heartbeat"  # 后端心跳接口地址
BACKEND_IDENTIFICATION = "wingerboy"  # 身份标识，按需修改
HEARTBEAT_INTERVAL = 60  # 心跳间隔秒
# 当前AI server服务对外地址，按需修改
CURRENT_WORKER_URL = "https://to74zigu-nx6sqm6b-6002.zjrestapi.gpufree.cn:8443"

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

# 创建一个辅助函数，用于在上下文中设置任务ID
def log_with_task_id(task_id, message, level='info'):
    """使用任务ID记录日志"""
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
        
        log_with_task_id(task_id, f"向Java后端报告进度: {json.dumps(payload)}")
        
        # 实际环境中取消注释下面的代码
        # response = requests.post(BACKEND_CALLBACK_URL, json=payload)
        # if response.status_code != 200:
        #     log_with_task_id(task_id, f"向后端报告进度失败: {response.text}", 'error')
    except Exception as e:
        log_with_task_id(task_id, f"向Java后端报告进度出错: {str(e)}", 'error')

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
            log_with_task_id(task_id, error_msg, 'error')
            update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
            return result
        
        if not os.path.exists(bg_path):
            error_msg = f"背景图片不存在: {bg_path}"
            log_with_task_id(task_id, error_msg, 'error')
            update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
            return result
        
        # 检查模型配置是否存在
        if model_name not in model_config:
            error_msg = f"未知的模型名称: {model_name}"
            log_with_task_id(task_id, error_msg, 'error')
            update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
            return result
        
        # 检查模型是否可用
        model_info = model_config[model_name]
        if model_info.get("available") is False:
            error_msg = f"错误: 模型暂时不可用: {model_name}. 原因: {model_info.get('reason', '未知')}"
            log_with_task_id(task_id, error_msg, 'error')
            update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
            return result
        
        # 创建进度回调函数
        def progress_callback(status, progress, message=""):
            log_with_task_id(task_id, f"任务进度: {status} - {progress}% - {message}")
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
            log_with_task_id(task_id, f"掩码视频已存在，跳过分割阶段: {mask_video_path}")
            skip_segmentation = True
            update_task_status(task_id, TaskStatus.PROCESSING, 50, "已找到现有掩码视频，跳过分割阶段")
        
        # 阶段1: 视频分割，生成掩码视频
        if not skip_segmentation:
            update_task_status(task_id, TaskStatus.PROCESSING, 10, "正在进行视频分割...")
            
            try:
                log_with_task_id(task_id, f"开始提取视频: {origin_video_path}")
                model_path = model_config[model_name]["model_path"]
                method = model_config[model_name].get("method", "birefnet")
                batch_size = model_config[model_name].get("max_batch_size", 4)
                log_with_task_id(task_id, f"使用模型 {model_name}, 方法 {method}")
                
                # 提取视频，生成掩码
                mask_path = extract_video(
                    video_path=origin_video_path,
                    model_path=model_path,
                    output_mask_path=mask_video_path,
                    method=method,
                    image_size=image_size,
                    batch_size=batch_size,
                    callback=progress_callback
                )
                
                # 检查掩码是否成功创建
                if not os.path.exists(mask_video_path):
                    error_msg = f"视频分割失败，无法创建掩码视频: {mask_video_path}"
                    log_with_task_id(task_id, error_msg, 'error')
                    update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
                    return result
                
                log_with_task_id(task_id, f"视频分割完成，掩码生成在: {mask_path}")
                
            except Exception as e:
                error_msg = f"视频分割过程中出错: {str(e)}"
                log_with_task_id(task_id, error_msg, 'error')
                log_with_task_id(task_id, traceback.format_exc(), 'error')
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
                log_with_task_id(task_id, error_msg, 'error')
                update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
                return result
            
            log_with_task_id(task_id, f"背景应用完成，合成视频生成在: {composite_path}")
            
        except Exception as e:
            error_msg = f"应用背景过程中出错: {str(e)}"
            log_with_task_id(task_id, error_msg, 'error')
            log_with_task_id(task_id, traceback.format_exc(), 'error')
            update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
            return result
        
        # 任务完成
        update_task_status(task_id, TaskStatus.COMPLETED, 100, "视频分割与背景合成完成")
        
    except Exception as e:
        error_msg = f"任务处理过程中出错: {str(e)}"
        log_with_task_id(task_id, error_msg, 'error')
        log_with_task_id(task_id, traceback.format_exc(), 'error')
        update_task_status(task_id, TaskStatus.FAILED, 100, error_msg)
    
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
        if not mask_video_path:
            # 构建默认路径
            video_name = os.path.splitext(os.path.basename(request.videoPath))[0]
            bg_name = os.path.splitext(os.path.basename(request.backgroundPath))[0]
            model_name = request.modelName
            image_size = model_config[model_name].get("image_size", (1024, 1024))
            
            mask_video_path = os.path.join(
                MASK_VIDEO_PATH, 
                f"mask-{video_name}-{bg_name}-{model_name}-{image_size[0]}x{image_size[1]}.mp4"
            )
            log_with_task_id(task_id, f"使用默认掩码路径: {mask_video_path}")
            
        # 确保目录存在
        os.makedirs(os.path.dirname(mask_video_path), exist_ok=True)
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
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
            "bg_path": request.backgroundPath,
            "model_name": request.modelName
        }
        
        # 更新任务状态为等待中
        update_task_status(task_id, TaskStatus.PENDING, 0, "任务已接收，等待处理")
        
        # 在后台启动任务
        background_tasks.add_task(process_video_segment, task_id, task_params)
        
        # 立即返回响应
        log_with_task_id(task_id, "视频分割任务已接收并开始处理")
        return {
            "taskId": task_id,
            "status": "accepted",
            "message": "视频分割任务已接收并开始处理",
            "maskVideoPath": mask_video_path
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
        
    log_with_task_id("heartbeat", f"心跳服务启动 - URL: {HEARTBEAT_URL}")
    
    while True:
        try:
            # 按照测试案例格式构建payload
            payload = {
                "interfaceAddress": CURRENT_WORKER_URL,
                "Identification": BACKEND_IDENTIFICATION
            }
            log_with_task_id("heartbeat", f"发送心跳: {payload}")
            
            # 发送请求
            resp = requests.post(HEARTBEAT_URL, json=payload, timeout=10)
            
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
        logger.warning("警告: 未安装moviepy库，输出视频将没有音频")
        
    host = os.environ.get("AI_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("AI_SERVER_PORT", "6002"))
    
    # 启动服务器
    uvicorn.run(app, host=host, port=port)
