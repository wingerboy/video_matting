#!/bin/bash

# Create necessary directories
mkdir -p models
mkdir -p uploads
mkdir -p outputs

# Check if models exist
if [ ! -f "models/ben2_lite_2K.safetensors" ] || [ ! -f "models/BiRefNet_lite_2K.safetensors" ]; then
    echo "警告: 模型文件不存在。请确保以下文件已放置在models目录中:"
    echo "  - models/ben2_lite_2K.safetensors"
    echo "  - models/BiRefNet_lite_2K.safetensors"
fi

# Run the application
streamlit run webui.py --server.port=3001 --server.address=0.0.0.0 