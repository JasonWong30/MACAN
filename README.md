# MACAN
Multiple Attention Channels Aggregated Network for Multimodal Medical Image Fusion

# MACAN (Medical Physics 2024)

Codes for ***Multiple Attention Channels Aggregated Network for Multimodal Medical Image Fusion. (Medical Physics 2024)*** 

[Jingxue Huang](https://github.com/JasonWong30), [Tianshu Tan](), [Xiaosong Li](https://github.com/lxs6), [Tao Ye](https://jdxy.cumtb.edu.cn/info/1011/3212.htm), [Yanxiong Wu](https://www.fosu.edu.cn/spoe/yanjiu/ssds/2625.html)

-[*[Paper]*](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.17607)   

## Update
- [2025-1] README.md is updated.
- [2024-12] Codes and config files are public available.

## Citation

```
@article{huangmultiple,
  title={Multiple attention channels aggregated network for multimodal medical image fusion},
  author={Huang, Jingxue and Tan, Tianshu and Li, Xiaosong and Ye, Tao and Wu, Yanxiong},
  journal={Medical Physics},
  publisher={Wiley Online Library}
}
```

## Abstract
### Background
In clinical practices, doctors usually need to synthesize several single-modality medical images for diagnosis, which is a time-consuming and costly process. With this background, multimodal medical image fusion (MMIF) techniques have emerged to synthesize medical images of different modalities, providing a comprehensive and objective interpretation of the lesion.
### Purpose
Although existing MMIF approaches have shown promising results, they often overlook the importance of multiscale feature diversity and attention interaction, which are essential for superior visual outcomes. This oversight can lead to diminished fusion performance. To bridge the gaps, we introduce a novel approach that emphasizes the integration of multiscale features through a structured decomposition and attention interaction.
### Methods
Our method first decomposes the source images into three distinct groups of multiscale features by stacking different numbers of diverse branch blocks. Then, to extract global and local information separately for each group of features, we designed the convolutional and Transformer block attention branch. These two attention branches make full use of channel and spatial attention mechanisms and achieve attention interaction, enabling the corresponding feature channels to fully capture local and global information and achieve effective inter-block feature aggregation.
### Results
For the MRI-PET fusion type, MACAN achieves average improvements of 24.48%, 27.65%, 19.24%, 27.32%, 18.51%, and 10.33% over the compared methods in terms of Qcb, AG, SSIM, SF, Qabf, and VIF metrics, respectively. Similarly, for the MRI-SPECT fusion type, MACAN outperforms the compared methods with average improvements of 29.13%, 26.43%, 18.20%, 27.71%, 16.79%, and 10.38% in the same metrics. In addition, our method demonstrates promising results in segmentation experiments. Specifically, for the T2-T1ce fusion, it achieves a Dice coefficient of 0.60 and a Hausdorff distance of 15.15. Comparable performance is observed for the Flair-T1ce fusion, with a Dice coefficient of 0.60 and a Hausdorff distance of 13.27.
### Conclusion
The proposed multiple attention channels aggregated network (MACAN) can effectively retain the complementary information from source images. The evaluation of MACAN through medical image fusion and segmentation experiments on public datasets demonstrated its superiority over the state-of-the-art methods, both in terms of visual quality and objective metrics. 

### 🌐 Usage

### ⚙ 1. Virtual Environment

```
 - [ ] torch  1.12.1
 - [ ] torchvision 0.13.1
 - [ ] numpy 1.24.2
 - [ ] Pillow  8.4.0
```

### 🏊 2. Data Preparation

Download the Multi-modal Medical Image Fusion (MMIF) and place the paired images in your own path.

### 🏄 3. Inference

If you want to infer with our MCAFusion and obtain the fusion results in our paper, please run

```
CUDA_VISIBLE_DEVICES=0 python Test2.py
```

## 🙌 MACAN

### Illustration of our MACAN model.

<img src="image//Workflow1.png" width="60%" align=center />

### Detail of MACAN.

<img src="image//Workflow2.png" width="60%" align=center />

<img src="image//Algorithm1.png" width="60%" align=center />

### Qualitative fusion results.

<img src="image//IVF1.png" width="100%" align=center />


### Quantitative fusion results.

Medical Image Fusion

<img src="image//Quantitative_MIF.png" width="60%" align=center />


## 📖 Related Work
- Zixiang Zhao, Lilun Deng, Haowen Bai, Yukun Cui, Zhipeng Zhang, Yulun Zhang, Haotong Qin, Dongdong Chen, Jiangshe Zhang, Peng Wang, Luc Van Gool. *title.* **ICML 2024**. https://arxiv.org/abs/2402.02235.
