#!/bin/bash
#SBATCH --partition=pi_panda
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=C100_VGG9_D_t30_lr1e-3_r5
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --error buffer/err/a_C100_VGG9_D_T30_lr1e-3_r5.txt
#SBATCH --output buffer/res/arpaper_C100_VGG9_D_t30_lr1e-3_r5.txt

source activate py37torch170

# dataset / encode / arch / leak_mem / T (timestep) /
# dataset [mnist, cifar10, cifar100]
# encode [p, d]
# arch [mlp, lenet, vgg9]
# T = [8, 10, 15, 20] for d / [15 ,20, 30, 50] for p
# leak_mem = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
# N = [120, 200]
# lr [0.01, 0.001]
# mm = %j


#CIFAR10 VGG9
python train_tau.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.5 --T 30 --lr 1e-3 --batch_size 128 --setting 'mainexp adam tau 1e-3 gain5'

# python train.py --dataset cifar10 --arch vgg9 --encode d --leak_mem 0.99 --T 10 --lr 0.1 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode d --leak_mem 0.50 --T 15 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode d --leak_mem 0.50 --T 20 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode d --leak_mem 0.50 --T 25 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode d --leak_mem 0.50 --T 30 --batch_size 128

# python train.py --dataset cifar10 --arch vgg9 --encode p --leak_mem 0.99 --T 10 --lr 0.1 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode p --leak_mem 0.50 --T 20 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode p --leak_mem 0.50 --T 40 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode p --leak_mem 0.50 --T 50 --batch_size 128





# CIFAR100 VGG9


# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.50 --T 8 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.70 --T 8 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.90 --T 8 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.95 --T 8 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.99 --T 8 --batch_size 128

# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.50 --T 15 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.70 --T 15 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.90 --T 15 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.95 --T 15 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.99 --T 15 --batch_size 128

# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.50 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.70 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.90 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.95 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.99 --T 20 --batch_size 128

# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.99 --T 30 --batch_size 128


# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.50 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.70 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.90 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.95 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.99 --T 20 --batch_size 128

# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.50 --T 50 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.70 --T 50 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.90 --T 50 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.95 --T 50 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.99 --T 50 --batch_size 128

# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.50 --T 70 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.70 --T 70 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.90 --T 70 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.95 --T 70 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.99 --T 70 --batch_size 128



##### MNIST_MLP

# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.50 --T 8 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.70 --T 8 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.90 --T 8 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.95 --T 8 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.99 --T 8 --batch_size 128

# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.50 --T 10 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.70 --T 10 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.90 --T 10 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.95 --T 10 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.99 --T 10 --batch_size 128

# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.50 --T 15 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.70 --T 15 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.90 --T 15 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.95 --T 15 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.99 --T 15 --batch_size 128

# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.50 --T 20 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.70 --T 20 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.90 --T 20 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.95 --T 20 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode d --leak_mem 0.99 --T 20 --batch_size 128

# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.50 --T 10 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.70 --T 10 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.90 --T 10 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.95 --T 10 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.99 --T 10 --batch_size 128

# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.50 --T 15 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.70 --T 15 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.90 --T 15 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.95 --T 15 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.99 --T 15 --batch_size 128

# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.50 --T 20 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.70 --T 20 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.90 --T 20 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.95 --T 20 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.99 --T 20 --batch_size 128

# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset mnist --arch mlp --encode p --leak_mem 0.99 --T 30 --batch_size 128






#CIFAR10 Lenet

# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.50 --T 8 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.70 --T 8 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.90 --T 8 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.95 --T 8 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.99 --T 8 --batch_size 128

# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.50 --T 10 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.70 --T 10 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.90 --T 10 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.95 --T 10 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.99 --T 10 --batch_size 128

# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.50 --T 15 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.70 --T 15 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.90 --T 15 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.95 --T 15 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.99 --T 15 --batch_size 128

# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.50 --T 20 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.70 --T 20 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.90 --T 20 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.95 --T 20 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.99 --T 20 --batch_size 128

# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.50 --T 15 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.70 --T 15 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.90 --T 15 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.95 --T 15 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.99 --T 15 --batch_size 128

# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.50 --T 20 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.70 --T 20 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.90 --T 20 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.95 --T 20 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.99 --T 20 --batch_size 128

# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.50 --T 50 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.70 --T 50 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.90 --T 50 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.95 --T 50 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.99 --T 50 --batch_size 128













###END########



##### MNIST_LeNet

# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.50 --T 8 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.70 --T 8 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.90 --T 8 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.95 --T 8 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.99 --T 8 --batch_size 128

# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.50 --T 10 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.70 --T 10 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.90 --T 10 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.95 --T 10 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.99 --T 10 --batch_size 128

# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.50 --T 15 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.70 --T 15 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.90 --T 15 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.95 --T 15 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.99 --T 15 --batch_size 128

# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.50 --T 20 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.70 --T 20 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.90 --T 20 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.95 --T 20 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode d --leak_mem 0.99 --T 20 --batch_size 128

# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.50 --T 10 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.70 --T 10 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.90 --T 10 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.95 --T 10 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.99 --T 10 --batch_size 128

# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.50 --T 15 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.70 --T 15 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.90 --T 15 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.95 --T 15 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.99 --T 15 --batch_size 128

# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.50 --T 20 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.70 --T 20 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.90 --T 20 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.95 --T 20 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.99 --T 20 --batch_size 128

# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.99 --T 30 --batch_size 128


# CIFAR100 Lenet


# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.50 --T 8 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.70 --T 8 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.90 --T 8 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.95 --T 8 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.99 --T 8 --batch_size 128

# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.50 --T 10 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.70 --T 10 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.90 --T 10 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.95 --T 10 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.99 --T 10 --batch_size 128

# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.50 --T 15 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.70 --T 15 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.90 --T 15 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.95 --T 15 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.99 --T 15 --batch_size 128

# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.50 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.70 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.90 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.95 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.99 --T 20 --batch_size 128

# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.50 --T 15 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.70 --T 15 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.90 --T 15 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.95 --T 15 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.99 --T 15 --batch_size 128

# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.50 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.70 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.90 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.95 --T 20 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.99 --T 20 --batch_size 128

# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.50 --T 50 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.70 --T 50 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.90 --T 50 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.95 --T 50 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.99 --T 50 --batch_size 128


















# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode d --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode d --leak_mem 0.99 --T 30 --batch_size 128


# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset mnist --arch lenet --encode p --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch lenet --encode p --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch lenet --encode p --leak_mem 0.99 --T 30 --batch_size 128




# python train.py --dataset mnist --arch vgg9 --encode d --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset mnist --arch vgg9 --encode d --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset mnist --arch vgg9 --encode d --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset mnist --arch vgg9 --encode d --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset mnist --arch vgg9 --encode d --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar10 --arch vgg9 --encode d --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode d --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode d --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode d --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode d --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset mnist --arch vgg9 --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset mnist --arch vgg9 --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset mnist --arch vgg9 --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset mnist --arch vgg9 --encode p --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset mnist --arch vgg9 --encode p --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar10 --arch vgg9 --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch vgg9 --encode p --leak_mem 0.95 --T 35 --batch_size 128 
# python train.py --dataset cifar10 --arch vgg9 --encode p --leak_mem 0.95 --T 35 --lr 0.002 --batch_size 128 
# python train.py --dataset cifar10 --arch vgg9 --encode p --leak_mem 0.95 --T 35 --lr 0.002 --batch_size 128 

# python train.py --dataset cifar10 --arch vgg9 --encode p --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch vgg9 --encode p --leak_mem 0.99 --T 30 --batch_size 128






# python train.py --dataset mnist --arch cifar10net --encode d --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset mnist --arch cifar10net --encode d --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset mnist --arch cifar10net --encode d --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset mnist --arch cifar10net --encode d --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset mnist --arch cifar10net --encode d --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar10 --arch cifar10net --encode d --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch cifar10net --encode d --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch cifar10net --encode d --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch cifar10net --encode d --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch cifar10net --encode d --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar100 --arch cifar10net --encode d --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch cifar10net --encode d --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch cifar10net --encode d --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch cifar10net --encode d --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch cifar10net --encode d --leak_mem 0.99 --T 30 --batch_size 128


# python train.py --dataset mnist --arch cifar10net --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset mnist --arch cifar10net --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset mnist --arch cifar10net --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset mnist --arch cifar10net --encode p --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset mnist --arch cifar10net --encode p --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar10 --arch cifar10net --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch cifar10net --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch cifar10net --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch cifar10net --encode p --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar10 --arch cifar10net --encode p --leak_mem 0.99 --T 30 --batch_size 128

# python train.py --dataset cifar100 --arch cifar10net --encode p --leak_mem 0.50 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch cifar10net --encode p --leak_mem 0.70 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch cifar10net --encode p --leak_mem 0.90 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch cifar10net --encode p --leak_mem 0.95 --T 30 --batch_size 128
# python train.py --dataset cifar100 --arch cifar10net --encode p --leak_mem 0.99 --T 30 --batch_size 128
