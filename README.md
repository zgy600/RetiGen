[toc]

### Introduction

Pytorch implementation for paper [**RetiGen**: A Framework for Generalized Retinal Diagnosis Using Multi-View Fundus Images](https://arxiv.org/abs/2403.15647) ![fig2](figures/framework.jpg)

### Abstract

This study introduces a novel framework for enhancing domain generalization in medical imaging, specifically focusing on utilizing unlabelled multi-view colour fundus photographs. Unlike traditional approaches that rely on single-view imaging data and face challenges in generalizing across diverse clinical settings, our method leverages the rich information in the unlabelled multi-view imaging data to improve model robustness and accuracy. By incorporating a class balancing method, a test-time adaptation technique and a multi-view optimization strategy, we address the critical issue of domain shift that often hampers the performance of machine learning models in real-world applications. Experiments comparing various state-of-the-art domain generalization and test-time optimization methodologies show that our approach consistently outperforms when combined with existing baseline and state-of-the-art methods. We also show our online method improves all existing techniques. Our framework demonstrates improvements in domain generalization capabilities and offers a practical solution for real-world deployment by facilitating online adaptation to new, unseen datasets.

### Getting started

#### Install

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

#### Data preparation

Your dataset should be organized as:

```
.
├── images
│   ├── DATASET1
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── DATASET2
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── DATASET3
│   │    ...
│   ...  ...
│   
├── masks
│   ├── DATASET1
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── DATASET2
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── DATASET3
│   │    ...
│   ...  ...
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

#### Training souce domain

#### Training target domain
