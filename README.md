# MACAN
Multiple Attention Channels Aggregated Network for Multimodal Medical Image Fusion

# MACAN (Medical Physics 2024)

Codes for ***Multiple Attention Channels Aggregated Network for Multimodal Medical Image Fusion. (Medical Physics 2024)*** 

[Jingxue Huang](https://github.com/JasonWong30), [Tianshu Tan](), [Xiaosong Li](https://github.com/lxs6), [Tao Ye](https://jdxy.cumtb.edu.cn/info/1011/3212.htm), [Yanxiong Wu](https://www.fosu.edu.cn/spoe/yanjiu/ssds/2625.html)

-[*[Paper]*](hhttps://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.17607)   

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

### Abstract
## Background
In clinical practices, doctors usually need to synthesize several single-modality medical images for diagnosis, which is a time-consuming and costly process. With this background, multimodal medical image fusion (MMIF) techniques have emerged to synthesize medical images of different modalities, providing a comprehensive and objective interpretation of the lesion.
## Purpose
Although existing MMIF approaches have shown promising results, they often overlook the importance of multiscale feature diversity and attention interaction, which are essential for superior visual outcomes. This oversight can lead to diminished fusion performance. To bridge the gaps, we introduce a novel approach that emphasizes the integration of multiscale features through a structured decomposition and attention interaction.
## Methods
Our method first decomposes the source images into three distinct groups of multiscale features by stacking different numbers of diverse branch blocks. Then, to extract global and local information separately for each group of features, we designed the convolutional and Transformer block attention branch. These two attention branches make full use of channel and spatial attention mechanisms and achieve attention interaction, enabling the corresponding feature channels to fully capture local and global information and achieve effective inter-block feature aggregation.
## Results
For the MRI-PET fusion type, MACAN achieves average improvements of 24.48%, 27.65%, 19.24%, 27.32%, 18.51%, and 10.33% over the compared methods in terms of Qcb, AG, SSIM, SF, Qabf, and VIF metrics, respectively. Similarly, for the MRI-SPECT fusion type, MACAN outperforms the compared methods with average improvements of 29.13%, 26.43%, 18.20%, 27.71%, 16.79%, and 10.38% in the same metrics. In addition, our method demonstrates promising results in segmentation experiments. Specifically, for the T2-T1ce fusion, it achieves a Dice coefficient of 0.60 and a Hausdorff distance of 15.15. Comparable performance is observed for the Flair-T1ce fusion, with a Dice coefficient of 0.60 and a Hausdorff distance of 13.27.
## Conclusion
The proposed multiple attention channels aggregated network (MACAN) can effectively retain the complementary information from source images. The evaluation of MACAN through medical image fusion and segmentation experiments on public datasets demonstrated its superiority over the state-of-the-art methods, both in terms of visual quality and objective metrics. 

### 🌐 Usage

### ⚙ 1. Virtual Environment

```
# create virtual environment
conda create -n DDFM python=3.8.10
conda activate DDFM
# select pytorch version yourself
# install DDFM requirements
pip install -r requirements.txt
```

### 📃 2. Pre-trained Checkpoint Preparation

From [the link](https://github.com/openai/guided-diffusion), download the checkpoint "256x256_diffusion_uncond.pt" and paste it to ``'./models/'``.

### 🏊 3. Data Preparation

Download the Infrared-Visible Fusion (IVF) and Medical Image Fusion (MIF) dataset and place the paired images in the folder ``'./input/'``.

### 🏄 4. Inference (Sampling)

If you want to infer with our DDFM and obtain the fusion results in our paper, please run

```
python sample.py
```

Then, the fused results will be saved in the ``'./output/recon/'`` folder.

Additionally,

- **[Random seed]:** For the randomly generated seed settings, we adopt the settings [seed = 3407](https://arxiv.org/abs/2109.08203) 😆😆😆(purely for fun, you can change it arbitrarily to obtain different sampling results).
- **[Sampling speed]:** Regarding the sampling speed, we use ``timestep_respacing: 100`` in the ``configs`` file (with a maximum setting of 1000). A larger ``timestep_respacing`` will result in better generation outcomes, but will take more time to sample.
- **[Step-by-step sampling results]:** If you want to save the sampling results of each step, please set ``record=True`` in ``sample.py``. The step-by-step sampling results will be saved in the ``'./output/progress/'`` folder.

## 🙌 DDFM

### Illustration of our DDFM model.

<img src="image//Workflow1.png" width="60%" align=center />

### Detail of DDFM.

<img src="image//Workflow2.png" width="60%" align=center />

<img src="image//Algorithm1.png" width="60%" align=center />

### Qualitative fusion results.

<img src="image//IVF1.png" width="100%" align=center />

<img src="image//IVF2.png" width="100%" align=center />

<img src="image//MIF.png" width="60%" align=center />

### Quantitative fusion results.

Infrared-Visible Image Fusion

<img src="image//Quantitative_IVF.png" width="100%" align=center />

Medical Image Fusion

<img src="image//Quantitative_MIF.png" width="60%" align=center />

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Zhaozixiang1228/MMIF-DDFM&type=Date)](https://star-history.com/#Zhaozixiang1228/MMIF-DDFM&Date)

## 📖 Related Work
- Zixiang Zhao, Lilun Deng, Haowen Bai, Yukun Cui, Zhipeng Zhang, Yulun Zhang, Haotong Qin, Dongdong Chen, Jiangshe Zhang, Peng Wang, Luc Van Gool. *Image Fusion via Vision-Language Model.* **ICML 2024**. https://arxiv.org/abs/2402.02235.
- Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang, Kai Zhang, Shuang Xu, Dongdong Chen, Radu Timofte, Luc Van Gool. *Equivariant Multi-Modality Image Fusion.* **CVPR 2024**. https://arxiv.org/abs/2305.11443
- Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang, Shuang Xu, Zudi Lin, Radu Timofte, Luc Van Gool.
  *CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion.* **CVPR 2023**. https://arxiv.org/abs/2211.14461
- Zixiang Zhao, Shuang Xu, Chunxia Zhang, Junmin Liu, Jiangshe Zhang and Pengfei Li. *DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion.* **IJCAI 2020**. https://www.ijcai.org/Proceedings/2020/135.
- Zixiang Zhao, Shuang Xu, Jiangshe Zhang, Chengyang Liang, Chunxia Zhang and Junmin Liu. *Efficient and Model-Based Infrared and Visible Image Fusion via Algorithm Unrolling.* **IEEE Transactions on Circuits and Systems for Video Technology 2021**. https://ieeexplore.ieee.org/document/9416456.
- Zixiang Zhao, Jiangshe Zhang, Haowen Bai, Yicheng Wang, Yukun Cui, Lilun Deng, Kai Sun, Chunxia Zhang, Junmin Liu, Shuang Xu. *Deep Convolutional Sparse Coding Networks for Interpretable Image Fusion.* **CVPR Workshop 2023**. https://robustart.github.io/long_paper/26.pdf.
- Zixiang Zhao, Shuang Xu, Chunxia Zhang, Junmin Liu, Jiangshe Zhang. *Bayesian fusion for infrared and visible images.* **Signal Processing**. https://doi.org/10.1016/j.sigpro.2020.107734.
