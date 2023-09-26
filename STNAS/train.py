import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# torch.manual_seed(0)

# import models.model as model
from util import adjust_learning_rate, accuracy, AverageMeter, progress_bar
import torchvision
# from torchvision import transforms
from spikingjelly.clock_driven.functional import reset_net
from torchvision import datasets
import torchvision.transforms as transforms
import torch.cuda.amp as amp
# import model_imagenet
from torch.cuda.amp import autocast
# from utils import progress_bar

import numpy as np
import os
import sys
import time
import argparse
import vgg

# ############# Reproducibility ##############
# seed = 2021
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# ############################################


## python train.py --dataset imagenet100 --lr 0.0005 --batch_size 128 --num_workers 15 --arch arch_2 --optim adam


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dump-dir', type=str, default="logdir")
parser.add_argument("--encode", default="d", type=str, help="Encoding [p d]")
parser.add_argument("--arch", default="vgg9", type=str, help="Arch [mlp, lenet, vgg9, cifar10net]")
parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset [mnist, cifar10, cifar100]")
parser.add_argument("--optim", default='sgd', type=str, help="Optimizer [adam, sgd]")
parser.add_argument('--leak_mem', default=0.5, type=float)
parser.add_argument('--T', type=int, default=5)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
parser.add_argument("--train_display_freq", default=1, type=int, help="display_freq for train")
parser.add_argument("--test_display_freq", default=1, type=int, help="display_freq for test")
parser.add_argument("--setting", type=str, help="display_freq for test")
parser.add_argument("--resume", default=False, help="resume or not")
parser.add_argument("--lw_ts", default=False, help="resume or not")
parser.add_argument("--distributed", default=0, type=int, help="Distributed or Not")
parser.add_argument("--in_memory", default=1, type=int, help="In-memory")

args = parser.parse_args()
print('hello')

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

batch_size = args.batch_size
lr = args.lr
leak_mem = args.leak_mem

dataset_dir = './data'  # '/gpfs/project/panda/shared'
dump_dir = args.dump_dir

arch_prefix = args.dataset + "_" + args.arch + "_" + args.encode
file_prefix = "T" + str(args.T) + "_lr" + str(args.lr) + "_epoch" + str(args.epoch) + "_leak" + str(args.leak_mem)

print('{}'.format(args.setting))
print("arch : {} ".format(arch_prefix))
print("hyperparam : {} ".format(file_prefix))

log_dir = os.path.join(dump_dir, 'logs', arch_prefix, file_prefix)
model_dir = os.path.join(dump_dir, 'models', arch_prefix, file_prefix)

file_prefix = file_prefix + '.pkg'

if args.lw_ts:
    file_path = os.path.join(log_dir, './log_file_lwts.txt')
else:
    file_path = os.path.join(log_dir, './log_file.txt')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

T = args.T
N = args.epoch

file_prefix = 'lr-' + np.format_float_scientific(lr, exp_digits=1, trim='-') + f'-b-{batch_size}-T-{T}'

# Data augmentation
img_size = {
    'mnist': 28,
    'cifar10': 32,
    'cifar100': 32,
    'imagenet100': 224,
    'imagenet100_raw': 224

}

num_cls = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'imagenet100': 100,
    'imagenet100_raw': 100
}

mean = {
    'mnist': 0.1307,
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'mnist': 0.3081,
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

    n_class = 10


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

    n_class = 10

elif args.dataset == 'imagenet100':

    import load_imagenet100_ffcv as img_ffcv

    batch_size = args.batch_size
    distributed = args.distributed
    in_memory = args.in_memory
    num_workers = args.num_workers

    train_dataset = '/gpfs/gibbs/project/panda/shared/imagenet100_ffcv/train_500_0.50_90.ffcv'
    val_dataset = '/gpfs/gibbs/project/panda/shared/imagenet100_ffcv/val_500_0.50_90.ffcv'
    train_data_loader = img_ffcv.create_train_loader(train_dataset, num_workers, batch_size,
                                                     distributed, in_memory)

    test_data_loader = img_ffcv.create_val_loader(val_dataset, num_workers, batch_size, distributed)
    n_class = 100

elif args.dataset == 'imagenet100_raw':
    # Data loading code
    traindir = os.path.join('/gpfs/gibbs/project/panda/shared/imagenet-100/train')
    valdir = os.path.join('/gpfs/gibbs/project/panda/shared/imagenet-100/val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trainset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]))

    train_data_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    n_class = 100

# python train.py --dataset imagenet100 --lr 0.01 --arch SNN_1436 --batch_size 64

vgg_configs = {'SNN_1446': [64, 64, 'M', 64, 512, 'M', 128, 256, 64, 'M', 64, 128, 512, 'M', 256, 64, 512, 'AM'],
               'SNN_1419': [64, 128, 'M', 256, 512, 'M', 128, 64, 128, 'M', 256, 512, 256, 'M', 256, 64, 128, 'AM'],
               'SNN_1414': [64, 256, 'M', 128, 64, 'M', 256, 64, 512, 'M', 128, 64, 256, 'M', 128, 64, 512, 'AM'],
               'SNN_1465': [64, 128, 'M', 256, 512, 'M', 64, 128, 128, 'M', 64, 128, 64, 'M', 256, 128, 128, 'AM'],
               'SNN_1436': [64, 128, 'M', 256, 512, 'M', 64, 64, 128, 'M', 512, 64, 256, 'M', 512, 64, 256, 'AM'],
               'SNN_VGG16': [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "AM"],
               'XPertNet_50': [64, 128, 'M', 64, 512, 'M', 512, 128 ,128, 'M', 64, 512, 128, 'M', 64, 64, 64, 'AM'],
               'SNN_1452': [128, 256, 'M', 64, 64, 'M', 128, 128, 512, 'M', 128, 512, 512, 'M', 512, 256, 256, 'AM'],
               'SNN_1431': [256, 128, 'M', 64, 256, 'M', 64, 128, 256, 'M', 256, 128, 512, 'M', 512, 512, 256, 'AM'],
               'SNN_1377': [64, 64, 'M', 64, 128, 'M', 256, 512, 128, 'M', 128, 256, 256, 'M', 256, 64, 128, 'AM'],
               'SNN_1415': [64, 64, 'M', 256, 128, 'M', 128, 512, 128, 'M', 512, 64, 512, 'M', 256, 64, 256, 'AM'],
               'SNN_1453': [256, 128, 'M', 512, 64, 'M', 512, 64, 256, 'M', 256, 128, 64, 'M', 64, 64, 128, 'AM'],
               'SNN_1547': [128, 256, 'M', 512, 128, 'M', 256, 256, 128, 'M', 64, 128, 64, 'M', 64, 128, 256, 'AM'],
               'SNN_3_1350': [64, 128, 'M', 64, 512, 'M', 128, 512, 256, 'M', 256, 256, 128, 'M', 256, 128, 128, 'AM'],
               'SNN_3_1331': [64, 64, 'M', 64, 128, 'M', 256, 128, 128, 'M', 128, 64, 512, 'M', 256, 256, 64, 'AM'],
               'SNN_3_1117': [256, 256, 'M', 128, 512, 'M', 64, 256, 512, 'M', 512, 256, 256, 'M', 512, 256, 256, 'AM'],
               'SNN_1400': [64, 128, 'M', 64, 512, 'M', 256, 128, 256, 'M', 64, 512, 64, 'M', 256, 128, 128, 'AM'],
               'SNN_1589': [512, 256, 'M', 128, 64, 'M', 256, 256, 128, 'M', 256, 512, 128, 'M', 64, 256, 128, 'AM'],
               'SNN_1400_mod': [64, 128, 'M', 64, 512, 'M', 256, 128, 256, 'M', 64, 512, 64, 'M', 512, 512, 512, 'AM'],
               'SNN_3_1226': [256, 128, 'M', 64, 64, 'M', 512, 256, 128, 'M', 256, 512, 512, 'M', 512, 64, 128, 'AM'],}
layers = vgg_configs[args.arch] #[64, 64, 'M', 64, 512, 'M', 128, 256, 64, 'M', 64, 128, 512, 'M', 256, 64, 512, 'AM'] #[64, 128, 'M', 64, 512, 'M', 512, 128, 128, 'M', 64, 512, 128, 'M', 64, 64, 64, 'AM']
if args.lw_ts:
    lw_timesteps = [5, 5, 6, 7, 4, 7, 6, 7, 6, 7, 6, 5, 7]
else:
    lw_timesteps = [5]*13
max_ts = max(lw_timesteps)
# lw_timesteps = [max_ts] * 13
# lw_timesteps = [max_ts]*13
net = vgg._vgg("custom", layers, True, False, True, num_linear_layers=layers[len(layers) - 2], total_timestep=max_ts,
                lw_timesteps=lw_timesteps, train_n=True, num_classes=n_class,
                dataset=args.dataset)  # [3,4,5,6,8,6,5,2,3,4,5,6]
net = torch.nn.DataParallel(net)

# net = model_imagenet.VGG16_blockshare(num_classes=100)
# net = torch.nn.DataParallel(net)

# print(net)

max_test_accuracy = 0

# Training Loop
net = net.cuda()

if args.resume == 'True':
    print(args.resume)
    model_file = torch.load(model_dir + "/" + "_bestmodel_lr_"+str(args.lr)+".pth.tar")
    start_epoch = model_file['global_step']
    best_acc = model_file['accuracy']
    print(f'Loading saved model with accuracy {best_acc}')
    # print(model_file['state_dict'].keys())
    net.load_state_dict(model_file['state_dict'])
else:
    start_epoch = 0

# Configure the loss function and optimizer
criterion = nn.CrossEntropyLoss()
if args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
else:
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
best_acc = 0

# Print the SNN model, optimizer, and simulation parameters
print("********** SNN simulation parameters **********")
print("Simulation # time-step : {}".format(T))
print("Membrane decay rate : {0:.2f}\n".format(args.leak_mem))
print("********** SNN learning parameters **********")
print("Backprop optimizer     : " + str(args.optim))
print("Batch size (training)  : {}".format(batch_size))
print("Batch size (testing)   : {}".format(batch_size * 2))
print("Number of epochs       : {}".format(args.epoch))
print("Learning rate          : {}".format(lr))

# --------------------------------------------------
# Train the SNN using surrogate gradients
# --------------------------------------------------
print("********** SNN training and evaluation **********")
train_loss_list = []
test_acc_list = []

# val_meters = {
# 'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=num_cls, compute_on_step=False).cuda(),
# 'top_5': torchmetrics.Accuracy(task='multiclass', num_classes=num_cls, compute_on_step=False, top_k=5).cuda(),
# 'loss': MeanScalarMetric(compute_on_step=False).cuda()
# }
with open(file_path, 'a') as file:
    file.write('{')
for epoch in range(start_epoch, start_epoch + args.epoch):
    time_start = time.time()

    train_loss = AverageMeter()
    # net.train()

    net.train()
    # EPS = 1e-6

    for batch_idx, (imgs, targets) in enumerate(train_data_loader):
        train_loss = 0.0
        # print(f'{batch_idx+1}/{len(train_data_loader)}')
        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda()

        with amp.autocast():
        # imgs = imgs.float()
            output_list = net(imgs)
            for output in output_list:
                train_loss += criterion(output/max_ts, targets) #/ args.T  # max_ts #args.timestep

        train_loss.backward()
        # scaler.scale(train_loss).backward()  # torch.amp
        # train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        # for name, p in model.named_parameters():
        #     if 'weight' in name:
        #         tensor = p.data
        #         if (len(tensor.size())) == 1:
        #             continue
        #         grad_tensor = p.grad
        #         grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
        #         p.grad.data = grad_tensor

        optimizer.step()
        reset_net(net)
        # if batch_idx == 0:
        #     exit(0)

        progress_bar(batch_idx, len(train_data_loader), 'Loss: %.3f'
                     % (train_loss / (batch_idx + 1)))
    scheduler.step()


    # for i, data in enumerate(train_data_loader):
    #     loss = 0
    #     inputs, labels = data
    #     inputs = inputs.cuda()
    #     labels = labels.cuda()
    #
    #     optimizer.zero_grad()
    #     with autocast():
    #         output_list = net(inputs)
    #         # output_list = model(imgs)
    #         for output in output_list:
    #             loss += criterion(output, labels) #/ args.T
    #     # train_loss.backward()
    #     # loss = criterion(output, labels)
    #     # prec1, prec5 = accuracy(output, labels, topk=(1, 5))
    #     # train_loss.update(loss.item(), labels.size(0))
    #     torch.autograd.set_detect_anomaly(True)
    #     loss.backward()
    #
    #     optimizer.step()
    #
    if (epoch + 1) % args.train_display_freq == 0:
        print(
            "Epoch: {}/{};".format(epoch + 1, args.epoch),
            "########## Training loss: {}".format(train_loss.mean()),
        )
    #
    # adjust_learning_rate(optimizer, epoch, args.epoch)

    if (epoch + 1) % args.test_display_freq == 0:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device('cuda')
        net.eval()
        test_loss = 0
        correct = 0
        n_samples = 0
        with torch.no_grad():
            with autocast():
                for data, target in test_data_loader:
                    # data, target = data.cuda(), target.cuda()
                    data = data.float()
                    output = net(data)
                    output = sum(output)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.data.view_as(pred)).sum().item()
                    reset_net(net)
                    n_samples += output.size(0)
                test_loss /= n_samples  # len(test_data_loader*data.size(0))
                accuracy = 100. * correct / n_samples  # len(test_data_loader*data.size(0))
                acc_string = f"'{epoch}': {accuracy}, \n"
                with open(file_path, 'a') as file:
                    # Write the data to the file
                    file.write(acc_string)

        # Model save
        if best_acc < accuracy:
            best_acc = accuracy

            net_dict = {
                "global_step": epoch + 1,
                "state_dict": net.state_dict(),
                "optim": optimizer.state_dict(),
                "accuracy": accuracy,
            }

            torch.save(
                net_dict, model_dir + "/" + "_bestmodel_lr_" + str(args.lr) + ".pth.tar"
            )
            print(model_dir + "/" + "_bestmodel_lr_" + str(args.lr) + ".pth.tar")
        print("best_accuracy : {}".format(best_acc))

    time_end = time.time()

with open(file_path, 'a') as file:
    file.write('}')
print("best accracy in {} is : {}".format(arch_prefix + file_prefix, best_acc))
# print(f'Elapse: {time_end - time_start:.2f}s')


sys.exit(0)


