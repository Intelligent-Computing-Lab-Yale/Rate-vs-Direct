import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# torch.manual_seed(0)


import models.model as model
from util import adjust_learning_rate, accuracy, AverageMeter
import torchvision
from torchvision import transforms
# import compute_score_2 as compute_score
import compute_score
import numpy as np
import os
import sys
import time
import argparse
import vgg 

############## Reproducibility ##############
# seed = 2021
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#############################################

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dump-dir', type=str, default="logdir")
parser.add_argument("--encode", default="d", type=str, help="Encoding [p d]")
parser.add_argument("--arch", default="vgg9", type=str, help="Arch [mlp, lenet, vgg9, cifar10net]")
parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset [mnist, cifar10, cifar100]")
parser.add_argument("--optim", default='adam', type=str, help="Optimizer [adam, sgd]")
parser.add_argument('--leak_mem', default=0.5, type=float)
parser.add_argument('--T', type=int, default=5)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument("--num_workers", default=1, type=int, help="number of workers")
parser.add_argument("--train_display_freq", default=1, type=int, help="display_freq for train")
parser.add_argument("--test_display_freq", default=1, type=int, help="display_freq for test")
parser.add_argument("--setting", type=str, help="display_freq for test")

args = parser.parse_args()


layers = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "AM"] #[64, 64, 'M', 64, 512, 'M', 128, 256, 64, 'M', 64, 128, 512, 'M', 256, 64, 512, 'AM']  #[64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "AM"] # #[256, 64, 'M', 64, 512, 'M', 128, 256, 512, 'M', 256, 512, 512, 'M', 256, 512, 64, 'M'] #[64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"] #[256, 64, 'M', 64, 512, 'M', 128, 256, 512, 'M', 256, 512, 512, 'M', 256, 512, 64, 'M'] #[64, 64, 'M', 128, 64, 'M', 256, 128, 256, 'M', 256, 64, 128, 'M', 64, 128, 256, 'M'] #[64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"] #[64, 64, 'M', 128, 64, 'M', 256, 128, 256, 'M', 256, 64, 128, 'M', 64, 128, 256, 'M'] #[256, 64, 'M', 64, 512, 'M', 128, 256, 512, 'M', 256, 512, 512, 'M', 256, 512, 64, 'M'] #[64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
lw_timesteps = [5]*13 #[2, 5, 6, 6, 5, 2, 3, 3, 3, 2, 7, 6, 3] #[5]*13 #[4, 4, 2, 2, 5, 2, 3, 4, 4, 2, 4, 2, 6] #[5]*13 #
net = vgg._vgg("custom", layers, False, False, False, num_linear_layers=512,
                               total_timestep=max(lw_timesteps), lw_timesteps=lw_timesteps, train_n=False, num_classes=100,
                               dataset='imagenet100')

score = compute_score.SNN_score(net, 'imagenet100')
print(score)



sys.exit(0)

