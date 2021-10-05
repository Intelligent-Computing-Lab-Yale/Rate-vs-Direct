# dataset / encode / arch / leak_mem / T (timestep) /
# dataset [mnist, cifar10, cifar100]
# encode [p, d]
# arch [mlp, lenet, vgg9]
# T = [8, 10, 15, 20] for d / [15 ,20, 30, 50] for p
# leak_mem = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
# N = [120, 200]
# lr [0.01, 0.001]


python train_tau.py --dataset cifar100 --arch vgg9 --encode d --leak_mem 0.5 --T 30 --lr 1e-3 --batch_size 128
