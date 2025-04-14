#!/bin/bash

# 设置通用参数
VIDEO_PATH="/root/gpufree-data/samplevideo/111.mp4"
OUTPUT_BASE="/root/gpufree-data/output"
BATCH_SIZE=8
LOG_FILE="model_comparison_$(date +%Y%m%d_%H%M%S).log"

# 创建日志文件
echo "模型比较开始: $(date)" > $LOG_FILE
echo "===============================" >> $LOG_FILE

# 模型列表 (从model_config中提取)
MODELS=(
    "BEN2_Base"
    "BiRefNet-HRSOD_DHU"
    "BiRefNet-massive-TR_DIS5K_TR_TEs"
    "BiRefNet"
    "BiRefNet_dynamic-matting"
    "BiRefNet-COD"
    "BiRefNet-general-bb_swin_v1_tiny"
    "BiRefNet-matting"
    "BiRefNet_HR-general"
    "BiRefNet_lite-general-2K"
    "BiRefNet-DIS"
    "BiRefNet-general"
    "BiRefNet-portrait"
    "BiRefNet_HR-matting"
    "BiRefNet_lite"
)

# 为每个模型创建单独的输出目录
for MODEL in "${MODELS[@]}"; do
    MODEL_OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}"
    mkdir -p $MODEL_OUTPUT_DIR
    
    echo ""
    echo "=========================================="
    echo "处理模型: $MODEL"
    echo "开始时间: $(date)"
    echo "输出目录: $MODEL_OUTPUT_DIR"
    echo "=========================================="
    
    # 记录到日志
    echo "" >> $LOG_FILE
    echo "模型: $MODEL" >> $LOG_FILE
    echo "开始时间: $(date)" >> $LOG_FILE
    
    # 执行处理命令，捕获输出和错误
    {
        python video_extractor.py \
            --model "$MODEL" \
            --video "$VIDEO_PATH" \
            --batch "$BATCH_SIZE" \
            --output "$MODEL_OUTPUT_DIR"
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            STATUS="成功"
        else
            STATUS="失败 (错误码: $EXIT_CODE)"
        fi
        
        END_TIME=$(date)
        
        echo "处理状态: $STATUS"
        echo "结束时间: $END_TIME"
        echo "----------------------------------------"
        
        # 记录到日志
        echo "处理状态: $STATUS" >> $LOG_FILE
        echo "结束时间: $END_TIME" >> $LOG_FILE
        echo "----------------------------------------" >> $LOG_FILE
        
    } 2>&1 | tee -a "${MODEL_OUTPUT_DIR}/processing.log"
done

echo ""
echo "所有模型处理完成，摘要信息保存在 $LOG_FILE"
echo "各模型的详细日志保存在各自的输出目录中"