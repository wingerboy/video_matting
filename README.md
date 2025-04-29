---
library_name: birefnet
tags:
- background-removal
- mask-generation
- Dichotomous Image Segmentation
- Camouflaged Object Detection
- Salient Object Detection
- pytorch_model_hub_mixin
- model_hub_mixin
repo_url: https://github.com/ZhengPeng7/BiRefNet
pipeline_tag: image-segmentation
---
<h1 align="center">Bilateral Reference for High-Resolution Dichotomous Image Segmentation</h1>

<div align='center'>
    <a href='https://scholar.google.com/citations?user=TZRzWOsAAAAJ' target='_blank'><strong>Peng Zheng</strong></a><sup> 1,4,5,6</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=0uPb8MMAAAAJ' target='_blank'><strong>Dehong Gao</strong></a><sup> 2</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=kakwJ5QAAAAJ' target='_blank'><strong>Deng-Ping Fan</strong></a><sup> 1*</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=9cMQrVsAAAAJ' target='_blank'><strong>Li Liu</strong></a><sup> 3</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=qQP6WXIAAAAJ' target='_blank'><strong>Jorma Laaksonen</strong></a><sup> 4</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=pw_0Z_UAAAAJ' target='_blank'><strong>Wanli Ouyang</strong></a><sup> 5</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=stFCYOAAAAAJ' target='_blank'><strong>Nicu Sebe</strong></a><sup> 6</sup>
</div>

<div align='center'>
    <sup>1 </sup>Nankai University&ensp;  <sup>2 </sup>Northwestern Polytechnical University&ensp;  <sup>3 </sup>National University of Defense Technology&ensp; <sup>4 </sup>Aalto University&ensp;  <sup>5 </sup>Shanghai AI Laboratory&ensp;  <sup>6 </sup>University of Trento&ensp; 
</div>

<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap;">
  <a href='https://arxiv.org/pdf/2401.03407'><img src='https://img.shields.io/badge/arXiv-BiRefNet-red'></a>&ensp; 
  <a href='https://drive.google.com/file/d/1aBnJ_R9lbnC2dm8dqD0-pzP2Cu-U1Xpt/view?usp=drive_link'><img src='https://img.shields.io/badge/中文版-BiRefNet-red'></a>&ensp; 
  <a href='https://www.birefnet.top'><img src='https://img.shields.io/badge/Page-BiRefNet-red'></a>&ensp; 
  <a href='https://drive.google.com/drive/folders/1s2Xe0cjq-2ctnJBR24563yMSCOu4CcxM'><img src='https://img.shields.io/badge/Drive-Stuff-green'></a>&ensp; 
  <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-yellow'></a>&ensp; 
  <a href='https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Spaces-BiRefNet-blue'></a>&ensp; 
  <a href='https://huggingface.co/ZhengPeng7/BiRefNet'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Models-BiRefNet-blue'></a>&ensp; 
  <a href='https://colab.research.google.com/drive/14Dqg7oeBkFEtchaHLNpig2BcdkZEogba?usp=drive_link'><img src='https://img.shields.io/badge/Single_Image_Inference-F9AB00?style=for-the-badge&logo=googlecolab&color=525252'></a>&ensp; 
  <a href='https://colab.research.google.com/drive/1MaEiBfJ4xIaZZn0DqKrhydHB8X97hNXl#scrollTo=DJ4meUYjia6S'><img src='https://img.shields.io/badge/Inference_&_Evaluation-F9AB00?style=for-the-badge&logo=googlecolab&color=525252'></a>&ensp; 
</div>

## This repo holds the official weights of BiRefNet_lite trained in 2K resolution.

### Training Sets:
+ DIS5K (except DIS-VD)
+ HRS10K
+ UHRSD
+ P3M-10k (except TE-P3M-500-NP)
+ TR-humans
+ AM-2k
+ AIM-500
+ Human-2k (synthesized with BG-20k)
+ Distinctions-646 (synthesized with BG-20k)
+ HIM2K
+ PPM-100

HR samples selection:
```
size_h, size_w = 1440, 2560
ratio = 0.8
h, w = image.shape[:2]
h >= size_h and w >= size_w or (h > size_h * ratio and w > size_w * ratio)
```

### Validation Sets:
+ DIS-VD
+ TE-P3M-500-NP

### Performance:
|    Dataset    |                Method               | maxFm | wFmeasure | MAE  | Smeasure | meanEm | HCE  | maxEm | meanFm | adpEm | adpFm | mBA  | maxBIoU | meanBIoU |
|     :------:    | :------: |  :------: |  :------: |  :------: |  :------: |  :------: |  :------: |  :------: |  :------: |  :------: |  :------: |  :------: |  :------: |  :------: |
|     DIS-VD    | BiRefNet_lite-2K-general--epoch_232 |  .867 |    .831   | .045 |   .879   |  .919  | 952  |  .925 |  .858  |  .916 |  .847 | .796 |   .750  |   .739   |
| TE-P3M-500-NP | BiRefNet_lite-2K-general--epoch_232 |  .993 |    .986   | .009 |   .975   |  .986  | .000 |  .993 |  .985  |  .833 |  .873 | .825 |   .921  |   .891   |


**Check the main BiRefNet model repo for more info and how to use it:**  
https://huggingface.co/ZhengPeng7/BiRefNet/blob/main/README.md  
> Remember to set the resolution of input images to 2K (2560, 1440) for better results when using this model.

**Also check the GitHub repo of BiRefNet for all things you may want:**  
https://github.com/ZhengPeng7/BiRefNet

## Acknowledgement:

+ Many thanks to @freepik for their generous support on GPU resources for training this model!


## Citation

```
@article{zheng2024birefnet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  volume = {3},
  pages = {9150038},
  year={2024}
}
```

# 视频前景提取工具

这是一个用于从视频中提取前景的工具，使用先进的深度学习模型BEN2-Lite和BiRefNet-Lite来处理视频，生成带透明通道的前景或带有指定背景的合成视频。

## 功能特点

- 支持多种深度学习模型处理
- 自动GPU加速（如果可用）
- 批处理加速视频处理
- 支持WebM格式（带透明通道）或MP4格式（带背景）
- 自定义背景颜色或背景图片
- 进度显示和状态更新
- 友好的Web界面

## 安装要求

1. Python 3.8+
2. PyTorch 2.0+ (带CUDA支持更佳)
3. Streamlit

### 安装依赖

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
pip install streamlit opencv-python pillow safetensors ffmpeg-python
```

## 准备模型文件

在运行应用前，您需要下载模型文件并放置在`models`目录中：

1. 创建`models`文件夹：
   ```bash
   mkdir -p models
   ```

2. 将BEN2-Lite和BiRefNet-Lite模型文件放入`models`文件夹
   - `ben2_lite_2K.safetensors`
   - `BiRefNet_lite_2K.safetensors`

## 运行Web界面

启动Streamlit Web界面，并将端口设置为3001以允许远程访问：

```bash
streamlit run webui.py --server.port=3001 --server.address=0.0.0.0
```

访问 `http://您的IP地址:3001` 来使用Web界面。

## 通过命令行使用

您也可以通过命令行直接处理视频：

```bash
python video_extractor.py --video 视频路径.mp4 --model BEN2-Lite --webm
```

参数说明：
- `--video`: 输入视频路径
- `--model`: 使用的模型 (BEN2-Lite 或 BiRefNet-Lite)
- `--output`: 输出目录
- `--webm`: 输出WebM格式（带透明通道）

## 使用说明

1. 上传视频文件
2. 选择模型类型(BEN2-Lite或BiRefNet-Lite)
3. 调整处理参数
4. 可选择上传背景图片
5. 点击"开始处理"按钮
6. 等待处理完成，下载生成的视频

## 系统要求

- 内存: 最低8GB，推荐16GB+
- GPU: 推荐NVIDIA GPU (6GB+ VRAM)
- 存储: 根据视频大小和时长，至少需要几GB可用空间

## 故障排除

- 如果出现"CUDA out of memory"错误，请减小批处理大小
- 如果模型加载失败，请确保模型文件已正确放置在models目录中
- 如果处理速度过慢，请考虑使用具有更多VRAM的GPU

## 授权

本项目使用的BEN2和BiRefNet模型有其各自的授权条款，请在使用前查阅相关信息。

# AI视频处理服务

这是一个基于FastAPI开发的AI视频处理后端服务，提供视频分割功能，可被Java后端服务调用。

## 功能特点

- 视频前景分割与抠图
- 异步任务处理，支持长时间运行的任务
- 实时进度报告
- 支持自定义背景
- 任务状态跟踪
- 基于BEN2模型的高质量视频处理

## 安装与配置

### 系统要求

- Python 3.8+
- CUDA 11.0+ (推荐，用于GPU加速)
- 足够的存储空间用于处理视频

### 安装步骤

1. 克隆代码库：

```bash
git clone https://github.com/your-repo/ai-video-service.git
cd ai-video-service
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 下载预训练模型：

下载BEN2预训练模型并放置于指定目录。

### 环境变量配置

可通过环境变量自定义服务配置：

- `HOST`: 服务监听地址，默认为 "0.0.0.0"
- `PORT`: 服务端口，默认为 8000
- `JAVA_CALLBACK_URL`: Java后端回调URL，默认为 "http://localhost:8080/api/task/update"

## 启动服务

```bash
python server.py
```

或使用uvicorn：

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

## API接口

### 1. 视频分割任务

**POST** `/api/video/segment`

启动一个视频分割任务。

请求参数：
```json
{
  "task_id": "任务ID",
  "origin_video_path": "原始视频路径",
  "fore_video_path": "输出前景视频路径",
  "origin_video_download_url": "原始视频下载URL",
  "model_path": "模型路径",
  "bg_path_download_url": "背景图片下载URL(可选)"
}
```

响应：
```json
{
  "taskId": "任务ID",
  "status": "accepted",
  "message": "视频分割任务已接收并开始处理"
}
```

### 2. 查询任务状态

**GET** `/api/task/{task_id}/status`

查询指定任务的处理状态。

响应：
```json
{
  "status": "PROCESSING",
  "progress": 45.5,
  "message": "处理中: 150/300",
  "updated_at": 1639012345.678
}
```

### 3. 健康检查

**GET** `/health`

检查服务是否正常运行。

响应：
```json
{
  "status": "ok",
  "service": "AI视频处理服务"
}
```

## 与Java后端集成

本服务会通过HTTP POST请求向Java后端发送任务进度和结果通知，格式如下：

```json
{
  "taskId": "任务ID",
  "status": "PROCESSING|COMPLETED|FAILED",
  "progress": 进度百分比,
  "message": "进度信息"
}
```

## 错误处理

- 如果任务处理失败，状态将设置为"FAILED"，并提供错误详情
- 服务会处理所有异常并通过日志记录详细信息
- API接口会返回标准HTTP错误码和错误信息

## 开发与调试

- 服务使用异步任务处理，不会阻塞主线程
- 可通过FastAPI的自动生成文档页面进行接口测试: `http://localhost:8000/docs`
- 日志会输出到控制台，可用于调试和监控
