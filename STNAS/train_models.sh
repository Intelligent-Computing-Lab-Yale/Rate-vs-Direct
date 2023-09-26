# dataset / encode / arch / leak_mem / T (timestep) /
# dataset [mnist, cifar10, cifar100]
# encode [p, d]
# arch [mlp, lenet, vgg9]
# T = [8, 10, 15, 20] for d / [15 ,20, 30, 50] for p
# leak_mem = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
# N = [120, 200]
# lr [0.01, 0.001]


python train.py --dataset imagenet100 --lr 0.06 --arch SNN_3_1117 --lw_ts True --batch_size 128 --resume False --epoch 200