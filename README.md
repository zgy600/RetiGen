# RetiGen: A Framework for Generalized Retinal Diagnosis Using Multi-View Fundus Images
## Introduction

Pytorch implementation for paper [**RetiGen**: A Framework for Generalized Retinal Diagnosis Using Multi-View Fundus Images](https://arxiv.org/abs/2403.15647) ![fig2](figures/framework.jpg)

## Abstract

This study introduces a novel framework for enhancing domain generalization in medical imaging, specifically focusing on utilizing unlabelled multi-view colour fundus photographs. Unlike traditional approaches that rely on single-view imaging data and face challenges in generalizing across diverse clinical settings, our method leverages the rich information in the unlabelled multi-view imaging data to improve model robustness and accuracy. By incorporating a class balancing method, a test-time adaptation technique and a multi-view optimization strategy, we address the critical issue of domain shift that often hampers the performance of machine learning models in real-world applications. Experiments comparing various state-of-the-art domain generalization and test-time optimization methodologies show that our approach consistently outperforms when combined with existing baseline and state-of-the-art methods. We also show our online method improves all existing techniques. Our framework demonstrates improvements in domain generalization capabilities and offers a practical solution for real-world deployment by facilitating online adaptation to new, unseen datasets.

## Getting started

### Install

1. Clone this repository and navigate to retigen folder

   ```
   git clone https://github.com/zgy600/RetiGen.git
   cd RetiGen/retigen
   ```
2. Install Package

   ```
   conda create -n retigen python=3.10 -y
   conda activate retigen
   pip install --upgrade pip
   conda install pytorch==2.0.1 torchvision==0.15.1 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
   ```
3. Install other dependencies

   ```
   pip install -r requirements.txt
   ```

### Data preparation

Your dataset should be organized as:

```
.
├── images
│   ├── DATASET1
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── DATASET2
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── DATASET3
│   │    ...
│   ...  ...
│   
├── masks
│   ├── DATASET1
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── DATASET2
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── DATASET3
│   │    ...
│   ...  ...
│   
└── splits
    ├── DATASET1_crossval.txt
    ├── DATASET1_train.txt
    ├── DATASET2_crossval.txt
    ├── DATASET2_train.txt
    ├── DATASET3_crossval.txt
    ├── DATASET3_train.txt
    ...

```

## Train and validate

### Training source domain
   ```
cd run/
table_1_origin_DG_MFIDDR_DRTiD.sh
   ```
### Training target domain
   ```
cd run/
table_1_target_DG_MFIDDR.sh
   ```

## Citation
If this repo is useful for your research, please consider citing our paper:
```bibtex
@misc{chen2024retigen,
      title={RetiGen: A Framework for Generalized Retinal Diagnosis Using Multi-View Fundus Images}, 
      author={Ze Chen and Gongyu Zhang and Jiayu Huo and Joan Nunez do Rio and Charalampos Komninos and Yang Liu and Rachel Sparks and Sebastien Ourselin and Christos Bergeles and Timothy Jackson},
      year={2024},
      eprint={2403.15647},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
