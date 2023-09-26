import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import copy
import compute_score_2 as compute_score
import os
import time
import numpy as np
from scipy.special import softmax
import argparse
import vgg
import dataset

# python SNN_Search_VGG16.py --dataset imagenet100

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=8, type=float, help='learning rate')
parser.add_argument('--hw_params', default='16,4,274,64', help='n_pe_tile, n_xbar_pe, total_tiles, xbar_size')
parser.add_argument('--t_latency', default=5000., type=float, help='target latency')
parser.add_argument('--t_area', default=80e6, type=float, help='target latency')
parser.add_argument('--factor', default=0.75, type=float, help='factor for PEs')
parser.add_argument('--epochs', default=2000, type=int, help='Epochs')
parser.add_argument('--wt_prec', default=8, type=int, help='Epochs')
parser.add_argument('--cellbit', default=4, type=int, help='Epochs')
parser.add_argument('--area_tolerance', default=2, type=int, help='Epochs')
parser.add_argument('--dataset', default='imagenet100', help='Dataset')
args = parser.parse_args()
class masking(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a_param):

        m = nn.Softmax()
        p_vals = m(a_param)

        gates = torch.FloatTensor([1]).cuda() * (1. *(p_vals == torch.max(p_vals))).float()

        return gates

    @staticmethod
    def backward(ctx, grad_output):

        grad_return = grad_output
        return grad_return, None

class compute_latency(nn.Module):

    def forward(self, layer_latency_nodes, layer_arch_params):
        softmax = nn.Softmax()
        prob = softmax(layer_arch_params)
        lat = torch.sum(layer_latency_nodes * prob)

        return lat

k_space = [3]
# ch_space = [16, 32, 64, 128, 256, 512]
ch_space = [64, 128, 256, 512]
f_space = [112] #, 16, 8, 4]
p_space = [1, 2, 4, 8, 32, 64]
T_space = [3,4,5,6,7]
# T_space = [3]
adc_type_space = [1]
## Hw config
n_pe_tile, n_xbar_pe, total_tiles, xbar_size = map(float, args.hw_params.split(',')) #torch.tensor([8]).cuda()

network_latency_nodes = []
network_arch = []
total_layers = 13

for layers in range(total_layers):
    layer_arch = []
    layer_latency_nodes = []
    for adc in adc_type_space:
        for i in ch_space:
            for j in k_space:
                for k in f_space:
                    for l in p_space:
                        for m in T_space:
                            if layers < 2:
                                layer_arch.append((adc, i, j, k, l, m))

                            elif layers >= 2 and layers < 4:
                                layer_arch.append((adc, i, j, k/2., l, m))

                            elif layers >= 4 and layers < 7:
                                layer_arch.append((adc, i, j, k/4., l, m))

                            elif layers >= 7 and layers < 10:
                                layer_arch.append((adc, i, j, k/8., l, m))

                            elif layers >= 10 and layers < 13:
                                layer_arch.append((adc, i, j, k / 16., l, m))


    network_latency_nodes.append(layer_latency_nodes)
    network_arch.append(layer_arch)

def LUT_area(tiles, mux, adc_type):
    if adc_type == 1:
        a,b,c,d,e = 2557.166495434807, 1325750339.0150185, 0.0037802706563706337, -1325744426.1570153, 355177.96577219985
    if adc_type == 2:
        a, b, c, d, e = 1662.1420546233026, 1647434588.0067654, 0.0011610296696243961, -1647433897.2349775, 328615.0508042259

    area = (a * mux + e) * tiles + b * torch.exp(c * tiles / mux) + d

    return area


def LUT_latency(tiles, mux, adc_type, speedup, feature_size, T_steps):

    if adc_type == 1:
        a, b, c, d, e = 0.08836970306542265, 8.466777271635184e-05, 1.0, 0.10469741426869324, 0.12532326984887984

    if adc_type == 2:
        a, b, c, d, e = 0.24051026469311507, 0.000432503086158724, 1.0, 0.4050948974138533, 0.16033299026875378


    latency = a * tiles * e + b * mux * mux + d
    latency = latency * (feature_size ** 2) * T_steps / speedup
    return latency


network_arch = np.array(network_arch)
network_arch = torch.from_numpy(network_arch)
network_arch = network_arch.cuda()
print(network_arch.size())
# print(f' len {network_arch[0])

th0 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th1 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th2 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th3 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th4 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th5 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th6 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th7 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th8 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th9 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th10 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th11 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th12 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)

arch_params = [th0, th1, th2, th3, th4, th5, th6, th7, th8, th9, th10, th11, th12] #, th12]

optimizer = optim.SGD(arch_params, lr=args.lr, momentum=0.99, weight_decay=0.00001)

loss_fn = torch.nn.MSELoss()
s = nn.Softmax()

target_latency = torch.FloatTensor([args.t_latency])
target_latency = target_latency.cuda()
target_latency = Variable(target_latency, requires_grad = True)

target_util = torch.FloatTensor([1])
target_util = target_util.cuda()
target_util = Variable(target_util, requires_grad = True)
comp_lat = compute_latency()

latency_list = []
arch_param_epoch = []
lat_err_list, r_loss_list, err_list, tile_list = [], [], [], []
best_pe = 0
best_diff = 50e6
best_area = 500e6
best_latency = 5000000.
best_score = 0

pool_layers = [1, 3, 6, 9]

class custom_model(nn.Module):
    def __init__(self, features, classifier):
        super(custom_model, self).__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

best_network = nn.Sequential(*([torch.nn.BatchNorm2d(3)]))
prev_list = [3]
for epochs in range(args.epochs):
    print(epochs)
    total_latency = 0
    total_pe = 0
    if epochs % 100 == 0:
        a_param = copy.deepcopy(arch_params)
        arch_param_epoch.append(a_param)

    parallel = []
    mux = []
    adc_arch = []
    if epochs % 1 == 0:
        in_ch = 3
        total_pe = 0
        tile_area = 0
        tile_latency = 0
        t_speedup = 0
        act_tile_area = 0
        act_tile_latency = 0
        expected_util = 0
        abs_tile = 0
        sup, t_use, util_list = [], [], []
        layers = []
        o_list = []
        for i in range(total_layers):
            prob = s(arch_params[i])
            index = torch.argmax(prob)
            a = network_arch[i][index]

            gate = masking.apply(prob)
            adc_type = a[0]
            o_ch = a[1]
            k = a[2]
            f = a[3]
            t_steps = a[5]
            xbar_s = torch.FloatTensor([xbar_size]).cuda()
            total_area = torch.FloatTensor([args.t_area]).cuda()
            layers.append(int(o_ch.item()))
            mux.append(int(t_steps.item()))
            o_list.append(int(o_ch.item()))
            if i in pool_layers:
                layers.append('M')

            if i < total_layers-1:

                n_cells = int(args.wt_prec / args.cellbit)
                n_xbars = torch.ceil(in_ch*k*k / xbar_s) * torch.ceil(o_ch*n_cells / xbar_s)
                n_tiles = torch.ceil(n_xbars/(n_pe_tile*n_xbar_pe))
                if (o_ch * n_cells / xbar_s) >= 1:
                    speedup = (n_tiles * n_xbar_pe * n_pe_tile / n_xbars)
                else:
                    speedup = (n_tiles * n_xbar_pe * n_pe_tile / n_xbars) * (xbar_s / (o_ch * n_cells))


                act_tile_area += LUT_area(n_tiles, 8, adc_type)
                act_tile_latency += LUT_latency(n_tiles, 8, adc_type, speedup, f, t_steps)


                tile_area += gate.sum() * LUT_area(n_tiles, m, adc_type) #(668468 * (0.9981761863406945 * n_tiles - 0.041827731986675554))
                tile_latency += gate.sum() * LUT_latency(n_tiles, m, adc_type, speedup, f, t_steps)


                t_use.append(n_tiles.item())



            if i == total_layers-1:
                feature_size = int(32 / (2**len(pool_layers)))
                n_cells = int(args.wt_prec / args.cellbit)
                fc_xbars = torch.ceil(o_ch / xbar_s) * torch.ceil(10*n_cells / xbar_s)
                n_tiles = torch.ceil(fc_xbars / 64.)

                act_tile_area += LUT_area(n_tiles, 8, adc_type)
                tile_area += gate.sum() * LUT_area(n_tiles, 8, adc_type)



                linear_layer_size = int(o_ch.item())

                if args.dataset == 'imagenet100':
                    num_class = 100
                layers = layers + ['AM']
                lw_timesteps = [min(mux)]*13

                max_ts = max(lw_timesteps)
                network = vgg._vgg("custom", layers, False, False, False, num_linear_layers=linear_layer_size,
                               total_timestep=max_ts, lw_timesteps=lw_timesteps, train_n=False, num_classes=num_class,
                               dataset=args.dataset)  # [3,4,5,6,8,6,5,2,3,4,5,6]
                print(mux, lw_timesteps, layers)

            in_ch = int(o_ch.item())

        print(f' %%%%%% act tile speedup {act_tile_latency.item(), tile_latency.item()} best_latency {best_latency} act_area {act_tile_area.item()} Time_steps {np.mean(mux)}')
        factor = args.factor
        target_area = torch.FloatTensor([total_area]).cuda()

        error = 0.0005 * tile_latency + 0.1 * 1e-6 * (loss_fn(tile_area, total_area))

        err_list.append(error.item())

        if o_list != prev_list and (act_tile_area-total_area).abs().item() < (args.area_tolerance*0.01*total_area).item(): #act_tile_latency.item() < best_latency:

            print('enters here')

            prev_list = o_list
            try:
                score = compute_score.SNN_score(network, args.dataset, lw_timesteps)
                print(score)
            except:
                print('$$$$$$$$$$$$$$$ Model Too large for GPU $$$$$$$$$$$$$$$$')
                continue


            if layers != best_network and score >= best_score:

                print('hello')
                best_diff = act_tile_latency.item()
                best_area = act_tile_area.detach().cpu().numpy()
                best_latency = act_tile_latency.detach().cpu().numpy()
                best_mux = mux
                best_network = layers
                best_par = parallel
                best_tile_use = t_use
                best_score = score
                best_adc_arch = adc_arch
                best_adc_mean = np.mean(np.array(best_adc_arch))
                best_mux_mean = np.mean(np.array(best_mux))
                best_tile_sum = np.sum(np.array(best_tile_use))

                best_arch = copy.deepcopy(arch_params)

    flag=4
    optimizer.zero_grad()
    error.backward(retain_graph=True)
    optimizer.step()
arch = []
for i in range(total_layers):
    prob = s(best_arch[i])
    index = torch.argmax(prob)
    a = network_arch[i][index]
    print(a)
    arch.append(a)

arch2 = []
arch3 = []
csv = []
in_ch = 3
l_c = 0
for i in arch:
    if i[2] == 3:
        pad = 1
    if i[2] == 5:
        pad = 2
    if i[2] == 7:
        pad = 3
    if int(i[3].item()) == 4 or int(i[3].item()) == 2:
        f_size = 6
    else:
        f_size = int(i[3].item())

    csv.append((f_size+2, f_size+2, int(in_ch),  int(i[2].item()),
                int(i[2].item()), int(i[1].item()), 0, 1, int(i[5]), int(i[0]) ))

    if int(in_ch) == 64 and int(i[1].item()) == 512:
        csv.append((int(i[3].item())+2, int(i[3].item())+2,512,3,3,64,0,1,int(i[5]), int(i[0])))
    else:
        csv.append((int(i[3].item())+2, int(i[3].item())+2,64,3,3,512,0,1,int(i[5]), int(i[0])))

    arch3.append(('C', int(i[1].item()), int(i[2].item()), 'same', int(8)))
    if l_c == 1 or l_c == 3 or l_c == 6 or l_c == 9:
        arch3.append(('M',2,2))


    arch2.append(('C', int(in_ch), int(i[1].item()), int(i[2].item()), 'same', int(8)))
    if int(in_ch) == 64 and int(i[1].item()) == 512:
        arch2.append(('C', 512, 64, 3, 'same', int(8)))
    else:
        arch2.append(('C',64, 512, 3, 'same', int(8)))
    if l_c == 1 or l_c == 3 or l_c == 6 or l_c == 9:
        arch2.append(('M',2,2))
    l_c += 1
    in_ch = int(i[1].item())

print(f'latency {best_latency.item()},\nbest_adc_arch {best_adc_arch}, \nbest_ADC_area {best_area}, \nbest_network {best_network}, \nlw_timesteps {best_mux} \nbest_t_use {best_tile_use} \nbest_score {best_score} \n adc_mean {best_adc_mean} mux_mean {best_mux_mean} tile_sum {best_tile_sum}')

for i in range(len(csv)):
    print(str(csv[i])[1:-1])

print(arch3)
print(arch2)
print('\n')