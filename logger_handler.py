#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一日志处理模块

为项目提供统一的日志记录功能：
1. 支持按日期自动分割日志文件
2. 支持关联任务ID
3. 同时输出到控制台和日志文件
4. 提供简单易用的接口
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler

# 确保日志目录存在
def ensure_log_dir(log_dir='logs'):
    """确保日志目录存在"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

class TaskIDFilter(logging.Filter):
    """添加任务ID到日志记录的过滤器"""
    def __init__(self, name=''):
        super().__init__(name)
        self.task_id = ''
    
    def filter(self, record):
        record.task_id = getattr(record, 'task_id', 'no-task-id')
        return True

# 创建单例日志记录器
def get_logger(name=None, log_dir='logs', log_level=logging.INFO):
    """获取配置好的日志记录器"""
    logger_name = name if name else __name__
    logger = logging.getLogger(logger_name)
    
    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger
    
    # 确保日志目录存在
    log_dir = ensure_log_dir(log_dir)
    log_file = os.path.join(log_dir, f"{logger_name}.log")
    
    # 配置日志记录器
    logger.setLevel(log_level)
    
    # 添加日志过滤器
    task_id_filter = TaskIDFilter()
    logger.addFilter(task_id_filter)
    
    # 日志格式
    formatter = logging.Formatter('%(asctime)s [%(task_id)s] - %(name)s - %(levelname)s - %(message)s')
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器 (按时间分割)
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',  # 每天午夜分割
        interval=1,       # 1天一个日志文件
        backupCount=10,   # 保留30天的日志
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# 统一的日志记录函数
def log_with_task_id(task_id, message, level='info', logger_name=None):
    """
    使用任务ID记录日志
    
    参数:
        task_id: 任务ID
        message: 日志消息
        level: 日志级别 (debug, info, warning, error, critical)
        logger_name: 日志记录器名称，不提供则使用__name__
    """
    logger = get_logger(logger_name)
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