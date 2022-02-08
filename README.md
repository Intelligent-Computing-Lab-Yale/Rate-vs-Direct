# Rate-vs-Direct

This repository contains the source code associated with "RATE CODING OR DIRECT CODING: WHICH ONE IS BETTER FOR ACCURATE, ROBUST, and ENERGY-EFFICIENT SPIKING NEURAL NETWORKS?", accepted to ICASSP2022. (https://arxiv.org/abs/2202.03133)


## Introduction

Spiking Neural Networks (SNNs) have recently emerged as the low-power alternative to Artificial Neural Networks (ANNs), because of their asynchronous, sparse, and binary event-driven processing. Recent SNN works focus on an image classification task, therefore various coding techniques have been proposed to convert an image into temporal binary spikes. Among them, rate coding and direct coding are regarded as  prospective candidates for building a practical SNN system as they show state-of-the-art performance on large-scale datasets. Despite their usage, there is little attention to comparing these two coding schemes in a fair manner. In this paper, we conduct a comprehensive analysis of the two  coding techniques from three perspectives: accuracy, adversarial robustness, and energy-efficiency. 
First, we compare the performance of two coding techniques with three different architectures on various datasets. Then, we attack SNNs with two adversarial attack methods to reveal the adversarial robustness of each coding scheme. Finally, we evaluate the energy-efficiency of two coding schemes on a digital hardware platform. Our results show that direct coding can achieve better accuracy especially for a small number of timesteps. On the other hand, rate coding shows better robustness to adversarial attacks owing to the non-differentiable spike generation process. Rate coding also yields higher energy-efficiency than direct coding which requires multi-bit precision for the first layer. Our study explores the advantages and disadvantages of two codings, which is an important design consideration for building SNNs.



## Prerequisites
* Ubuntu 18.04    
* Python 3.6+    
* PyTorch 1.5+ (recent version is recommended)     
* Torchvision 0.8.0+ (recent version is recommended)     
* NVIDIA GPU (>= 12GB)        


## Training and testing

* ```train.py```: code for training  
* ```model.py```: code for MLP/VGG5/VGG9 Spiking Neural Networks with Rate/Direct coding
* ```util.py```: code for accuracy calculation / learning rate scheduler


* Argparse configuration

```
--dataset [mnist, cifar10, cifar100]
--encode [p, d]
--arch [mlp, vgg5, vgg9]
--T(timestep) = [5, 10, 15, 20, 30]
--leak_mem = [0.5]
--epoch [100]
--lr [1e-3]
```

*  Run the following command for VGG5-SNN-Direct on CIFAR10

```
python train.py --dataset cifar10 --arch vgg5 --epoch 100 --encode d --leak_mem 0.5 --T 10 --lr 1e-3 --batch_size 128
```

*  Run the following command for VGG9-SNN-Poisson on CIFAR100

```
python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.5 --T 20 --lr 1e-3 --batch_size 128
```


 
 

