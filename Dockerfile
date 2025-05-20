FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# 环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 更新并安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN python -m pip install --upgrade pip

# 单独安装所需 Python 依赖（显式列出，torch/torchvision 已随基础镜像提供）
RUN python -m pip install \
    accelerate==1.6.0 \
    annotated-types==0.7.0 \
    anyio==4.9.0 \
    certifi==2025.1.31 \
    charset-normalizer==3.4.1 \
    click==8.1.8 \
    decorator==4.4.2 \
    einops==0.8.1 \
    fastapi==0.115.12 \
    ffmpeg-python==0.2.0 \
    filelock==3.18.0 \
    fsspec==2025.3.2 \
    future==1.0.0 \
    h11==0.16.0 \
    huggingface-hub==0.30.2 \
    idna==3.10 \
    imageio==2.37.0 \
    imageio-ffmpeg==0.6.0 \
    Jinja2==3.1.6 \
    kornia==0.8.0 \
    kornia_rs==0.1.8 \
    lazy_loader==0.4 \
    MarkupSafe==3.0.2 \
    moviepy==1.0.3 \
    mpmath==1.3.0 \
    networkx==3.4.2 \
    numpy==1.26.4 \
    opencv-python-headless==4.11.0.86 \
    packaging==24.2 \
    pillow==10.4.0 \
    prettytable==3.16.0 \
    proglog==0.1.11 \
    psutil==7.0.0 \
    pydantic==2.11.3 \
    pydantic_core==2.33.1 \
    python-dotenv==1.1.0 \
    PyYAML==6.0.2 \
    regex==2024.11.6 \
    requests==2.32.3 \
    safetensors==0.5.3 \
    scikit-image==0.25.2 \
    scipy==1.15.2 \
    setuptools==75.8.0 \
    sniffio==1.3.1 \
    starlette==0.46.2 \
    sympy==1.13.1 \
    tifffile==2025.3.30 \
    timm==1.0.15 \
    tokenizers==0.21.1 \
    tqdm==4.67.1 \
    transformers==4.51.2 \
    triton==3.1.0 \
    typing_extensions==4.13.2 \
    typing-inspection==0.4.0 \
    urllib3==2.4.0 \
    uvicorn==0.34.2 \
    wcwidth==0.2.13 \
    wheel==0.45.1
