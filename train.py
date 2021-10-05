
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter


import models.model as model
from util import adjust_learning_rate, accuracy, AverageMeter
import torchvision
from torchvision import transforms


import numpy as np
import os
import sys
import time
import argparse


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
parser.add_argument('--leak_mem',default=0.5, type=float)
parser.add_argument('--T', type=int, default=8)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
parser.add_argument("--train_display_freq", default=10, type=int, help="display_freq for train")
parser.add_argument("--test_display_freq", default=10, type=int, help="display_freq for test")
parser.add_argument("--setting", type=str, help="display_freq for test")




args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

batch_size = args.batch_size
lr = args.lr
leak_mem = args.leak_mem

dataset_dir = '/gpfs/project/panda/shared'
dump_dir = args.dump_dir

arch_prefix = args.dataset +"_" + args.arch + "_" + args.encode
file_prefix = "T" + str(args.T) + "_lr" + str(args.lr) + "_epoch" + str(args.epoch) + "_leak" + str(args.leak_mem)

print('{}'.format(args.setting))

print("arch : {} ".format(arch_prefix))
print("hyperparam : {} ".format(file_prefix))

log_dir = os.path.join(dump_dir, 'logs', arch_prefix, file_prefix)
model_dir = os.path.join(dump_dir, 'models', arch_prefix, file_prefix)

file_prefix = file_prefix + '.pkg'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


T = args.T
N = args.epoch

file_prefix = 'lr-' + np.format_float_scientific(lr, exp_digits=1, trim='-') + f'-b-{batch_size}-T-{T}'

# Data augmentation
img_size = {
    'mnist' : 28,
    'cifar10': 32,
    'cifar100': 32,
}

num_cls = {
    'mnist' : 10,
    'cifar10': 10,
    'cifar100': 100,
}

mean = {
    'mnist' : 0.1307,
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    }

std = {
    'mnist' : 0.3081,
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    }

if args.dataset == 'mnist':
    input_dim = 1
else:
    input_dim = 3

    
img_size = img_size[args.dataset]
num_cls = num_cls[args.dataset]

if args.dataset == 'mnist':
    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),
    ])
    
    train_dataset = torchvision.datasets.MNIST(
            root=dataset_dir,
            train=True,
            transform=transform_train,
            download=True)

    test_dataset = torchvision.datasets.MNIST(
            root=dataset_dir,
            train=False,
            transform=transform_test,
            download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True, 
        num_workers=8, 
        pin_memory=True)
    
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False, 
        num_workers=8, 
        pin_memory=True)

elif args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset])
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        transform=transform_train,
        download=True)
        
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=False,
        transform=transform_test,
        download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True, 
        num_workers=4,
        pin_memory=True)
    
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False, 
        num_workers=4,
        pin_memory=True)


elif args.dataset == 'cifar100':

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset])
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataset_dir,
        train=True,
        transform=transform_train,
        download=True)
        
    test_dataset = torchvision.datasets.CIFAR100(
        root=dataset_dir,
        train=False,
        transform=transform_test,
        download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True, 
        num_workers=4,
        pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False, 
        num_workers=4,
        pin_memory=True)

Encoding = args.encode

if args.encode == 'd':
    if args.arch == 'mlp':
        net = model.MLP_Direct(num_steps=T, leak_mem= leak_mem, img_size = img_size, input_dim = input_dim).cuda()
        print(f'Create new model')
    elif args.arch == 'vgg5':
        net = model.VGG5_Direct(num_steps=T, leak_mem= leak_mem, img_size = img_size, input_dim = input_dim, num_cls = num_cls).cuda()
        print(f'Create new model')
    elif args.arch == 'vgg9':
        net = model.VGG9_Direct(num_steps=T, leak_mem= leak_mem, img_size = img_size, input_dim = input_dim, num_cls = num_cls).cuda()
        print(f'Create new model')
    else:
        print(f'Not implemented Err - Architecture')
        exit()

elif args.encode == 'p':
    if args.arch == 'mlp':
        net = model.MLP_Poisson(num_steps=T, leak_mem= leak_mem, input_dim = input_dim).cuda()
        print(f'Create new model')
    elif args.arch == 'vgg5':
        net = model.VGG5_Poisson(num_steps=T, leak_mem= leak_mem, img_size = img_size, input_dim = input_dim, num_cls = num_cls).cuda()
        print(f'Create new model')
    elif args.arch == 'vgg9':
        net = model.VGG9_Poisson(num_steps=T, leak_mem= leak_mem, input_dim = input_dim, img_size=img_size, num_cls = num_cls).cuda()
        print(f'Create new model')
    else:
        print(f'Not implemented Err - Architecture')
        exit()

else:
        print(f'Not implemented Err - Encoding')
        exit()


# print(net)

max_test_accuracy = 0

# Training Loop
net= net.cuda()

# Configure the loss function and optimizer
criterion = nn.CrossEntropyLoss()
if args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum = 0.9, weight_decay=1e-4)
else:
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
best_acc = 0

# Print the SNN model, optimizer, and simulation parameters
print("********** SNN simulation parameters **********")
print("Simulation # time-step : {}".format(T))
print("Membrane decay rate : {0:.2f}\n".format(args.leak_mem))
print("********** SNN learning parameters **********")
print("Backprop optimizer     : SGD")
print("Batch size (training)  : {}".format(batch_size))
print("Batch size (testing)   : {}".format(batch_size*2))
print("Number of epochs       : {}".format(args.epoch))
print("Learning rate          : {}".format(lr))

# --------------------------------------------------
# Train the SNN using surrogate gradients
# --------------------------------------------------
print("********** SNN training and evaluation **********")
train_loss_list = []
test_acc_list = []
start_epoch = 0


for epoch in range(args.epoch):
    time_start = time.time()

    train_loss = AverageMeter()
    net.train()
    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        output = net(inputs)
        
        loss = criterion(output, labels)
        prec1, prec5 = accuracy(output, labels, topk=(1, 5))
        train_loss.update(loss.item(), labels.size(0))
        loss.backward()
        optimizer.step()

    if (epoch + 1) % args.train_display_freq == 0:
        print(
            "Epoch: {}/{};".format(epoch + 1, args.epoch),
            "########## Training loss: {}".format(train_loss.avg),
        )

    adjust_learning_rate(optimizer, epoch, args.epoch)

    if (epoch + 1) % args.test_display_freq == 0:
        acc_top1, acc_top5 = [], []
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(test_data_loader):
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()

                out = net(images)
                prec1, prec5 = accuracy(out, labels, topk=(1, 5))
                acc_top1.append(float(prec1))
                acc_top5.append(float(prec5))

        test_accuracy = np.mean(acc_top1)

        # Model save
        if best_acc < test_accuracy:
            best_acc = test_accuracy

            net_dict = {
                "global_step": epoch + 1,
                "state_dict": net.state_dict(),
                "optim" : optimizer.state_dict(),
                "accuracy": test_accuracy,
            }

            torch.save(
                net_dict, model_dir + "/" + "_bestmodel.pth.tar"
            )
        print("best_accuracy : {}".format(best_acc))

    time_end = time.time()
print("best accracy in {} is : {}".format(arch_prefix + file_prefix, best_acc))
    # print(f'Elapse: {time_end - time_start:.2f}s')


sys.exit(0)

