# Rate-vs-Direct

This repository contains the source code associated with [arXiv preprint arXiv:2010.01729][arXiv preprint arXiv:2010.01729]

[arXiv preprint arXiv:2010.01729]: https://arxiv.org/abs/2010.01729

## Introduction

Spiking Neural Networks (SNNs) have recently emerged as an alternative to deep learning owing to sparse, asynchronous and binary event (or spike) driven processing, that can yield huge energy efficiency benefits on neuromorphic hardware. However, training high-accuracy and low-latency SNNs from scratch suffers from non-differentiable nature of a spiking neuron. To address this training issue in SNNs, we revisit batch normalization and propose a temporal Batch Normalization Through Time (BNTT) technique. Most prior SNN works till now have disregarded batch normalization deeming it ineffective for training temporal SNNs. Different from previous works, our proposed BNTT decouples the parameters in a BNTT layer along the time axis to capture the temporal dynamics of spikes. The temporally evolving learnable parameters in BNTT allow a neuron to control its spike rate through different time-steps, enabling low-latency and low-energy training from scratch. We conduct experiments on CIFAR-10, CIFAR-100, Tiny-ImageNet and event-driven DVS-CIFAR10 datasets. BNTT allows us to train deep SNN architectures from scratch, for the first time, on complex datasets with just few 25-30 time-steps. We also propose an early exit algorithm using the distribution of parameters in BNTT to reduce the latency at inference, that further improves the energy-efficiency.


## Prerequisites
* Ubuntu 18.04    
* Python 3.6+    
* PyTorch 1.5+ (recent version is recommended)     
* Torchvision 0.8.0+ (recent version is recommended)     
* NVIDIA GPU (>= 12GB)        

## Getting Started

### Installation
* Configure virtual (anaconda) environment
```
conda create -n env_name python=3.7
source activate env_name
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```


## Training and testing

* We provide VGG9/VGG11 architectures on CIFAR10/CIAR100 datasets
* ```train.py```: code for training  
* ```model.py```: code for MLP/VGG5/VGG9 Spiking Neural Networks with Rate/Direct coding
* ```util.py```: code for accuracy calculation / learning rate scheduler

*  Run the following command for VGG5-SNN-Direct on CIFAR10

```
python train.py --dataset cifar10 --arch vgg5 --encode d --leak_mem 0.5 --T 10 --lr 1e-3 --batch_size 128
```

*  Run the following command for VGG9-SNN-Poisson on CIFAR100

```
python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.5 --T 20 --lr 1e-3 --batch_size 128
```


## Citation
 
Please consider citing our paper:
 ```
 @article{kim2020revisiting,
  title={Revisiting Batch Normalization for Training Low-latency Deep Spiking Neural Networks from Scratch},
  author={Kim, Youngeun and Panda, Priyadarshini},
  journal={arXiv preprint arXiv:2010.01729},
  year={2020}
}
 ```
 
 

