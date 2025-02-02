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
