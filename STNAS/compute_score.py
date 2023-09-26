import argparse
import random
import numpy as np
import torch
import os
# from scores import get_score_func
from scipy import stats
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from spikingjelly.clock_driven.functional import reset_net
import copy
from torch.cuda.amp import autocast
from torchvision import datasets

from datetime import datetime

def logdet(K):
    s, ld = np.linalg.slogdet(K)
    return  ld

def det(K):
    ld = np.linalg.det(K)
    # print(ld)
    return  ld

def SNN_score(SNN_network, dataset, lw_ts):

    search_batchsize = 128


    # dataset = 'cifar10'
    # train_data = train_loader # Get the trainloader here

    if dataset == 'imagenet100':
        # batch_size = 128
        # distributed = 0
        # in_memory = 1
        # num_workers = 4
        repeat = 2
        torch.manual_seed(0)
        traindir = os.path.join('/gpfs/gibbs/project/panda/shared/imagenet-100/train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize]))

        train_data = torch.utils.data.DataLoader(
            trainset, batch_size=search_batchsize, shuffle=False,
            num_workers=4, pin_memory=True)

    # if dataset == 'imagenet100':
    #     import load_imagenet100_ffcv as img_ffcv
    #
    #     batch_size = 128
    #     distributed = 0
    #     in_memory = 1
    #     num_workers = 4
    #     repeat = 2
    #     train_dataset = '/gpfs/gibbs/project/panda/shared/imagenet100_ffcv/train_500_0.50_90.ffcv'
    #     val_dataset = '/gpfs/gibbs/project/panda/shared/imagenet100_ffcv/val_500_0.50_90.ffcv'
    #     train_data = img_ffcv.create_train_loader(train_dataset, num_workers, batch_size,
    #                                                      distributed, in_memory, score_mode= 1)

    else:
        import dataset
        repeat = 5
        print('DATASET IS CIFAR10 !!')
        training_set = dataset.get_dataset('cifar10')[0]
        train_data = torch.utils.data.DataLoader(training_set, batch_size=search_batchsize,
                                               shuffle=False, pin_memory=True, num_workers=4) ## Check This
    scores = []
    history = []
    neuron_type = 'LIFNode'
    # print(SNN_network)

    with torch.no_grad():
        # for i in range(args.num_search):
        #     while (1):
        #         # con_mat =connection_matrix_gen(args, num_node=4, num_options=5)
        #         #
        #         # # sanity check on connection matrix
        #         # neigh2_cnts = torch.mm(con_mat, con_mat)
        #         # neigh3_cnts = torch.mm(neigh2_cnts, con_mat)
        #         # neigh4_cnts = torch.mm(neigh3_cnts, con_mat)
        #         # connection_graph = con_mat + neigh2_cnts + neigh3_cnts + neigh4_cnts
        #
        #         for node in range(3):
        #             if connection_graph[node,3] ==0: # if any node doesnt send information to the last layer, remove it
        #                 con_mat[:, node] = 0
        #                 con_mat[node,:] = 0
        #         for node in range(3):
        #             if connection_graph[0,node+1] ==0: # if any node doesnt get information from the input layer, remove it
        #                 con_mat[:, node+1] = 0
        #                 con_mat[node+1,:] = 0
        #
        #         if con_mat[0, 3] != 0: # ensure direct connection between input=>output for fast information propagation
        #             break


        searchnet = SNN_network # SNASNet(args, con_mat)
        searchnet.cuda()
        searchnet.eval()
        searchnet.K = np.zeros((search_batchsize, search_batchsize))
        searchnet.num_actfun = 0

        num_LIF_layers = 0

        dummy_tensor = torch.tensor([]).cuda()
        for name, module in searchnet.named_modules():
            # print(str(type(module)))
            if neuron_type in str(type(module)):
                num_LIF_layers += 1
                # searchnet.spike_map_list.append(torch.tensor([]).cuda())
        def computing_K_eachtime(module, inp, out):
            if isinstance(out, tuple):
                # print('########################################')
                out = out[0]
                # out = out.contiguous()
            # print(out.size())
            out = out.contiguous().view(out.size(0), -1)

            # print (out.sum()/torch.numel(out))

            # print(f' layer {int(searchnet.count % num_LIF_layers)} TS {int(searchnet.count/num_LIF_layers)}')

            if searchnet.count >= 0:
                time = searchnet.time #int(searchnet.count / (num_LIF_layers-1))
                layer = searchnet.layer_no #int(searchnet.count % (num_LIF_layers-1))
            # print(f'layer {layer}')

            # searchnet.spike_map_list.append(x)

            batch_num , neuron_num = out.size()
            x = (out > 0).float()

            # sp_map_temp = copy.deepcopy(out)
            # print(f'size {searchnet.spike_map_list[layer].size()} time {time} layer {layer}')
            # print(f'layer {layer}')

            # print(f'searchcount {searchnet.count}')
            if searchnet.count == -1:
                dup_x = x.repeat(1, 5)
                searchnet.spike_map_list.append(dup_x)

            else:
                if time == 0:
                    # print(f'size time zero {searchnet.count} {x.size()} {time} {layer+1}')
                    searchnet.spike_map_list.append(x)

                elif time > 0 and layer < 12: # and time < lw_ts[layer]:
                    # print('not time 0')
                    # print(f'size {searchnet.count} {x.size()} {time} {layer+1}')
                    searchnet.spike_map_list[layer + 1] = torch.cat((searchnet.spike_map_list[layer + 1], x), 1)

            # print(f'x.size() {x.size()} searchcount {searchnet.count}')
            searchnet.count += 1


            # if time == 0:
            #     # deepcopy the spike map
            #     # searchnet.layer1_spike_map = copy.deepcopy(x)
            #     # print(f'time 0')
            #     print(f'size {searchnet.count} {out.size()} {time} {layer}')
            #     searchnet.spike_map_list.append(x)
            #     # searchnet.spike_map_list[layer] = torch.cat((searchnet.spike_map_list[layer], x), 0)
            #     # print(f' time {time}, layer {layer}, tensor_size {searchnet.spike_map_list[layer].size()}')
            #
            # elif time > 0 and layer < 12:
            #     # print('not time 0')
            #     print(f'size {searchnet.count} {x.size()} {time} {layer}')
            #     searchnet.spike_map_list[layer+1] = torch.cat((searchnet.spike_map_list[layer+1], x), 1)


            searchnet.num_actfun = num_LIF_layers

            full_matrix = torch.ones((search_batchsize, search_batchsize)).cuda() * neuron_num
            sparsity = (x.sum(1)/neuron_num).unsqueeze(1)
            norm_K = ((sparsity @ (1-sparsity.t())) + ((1-sparsity) @ sparsity.t())) * neuron_num
            # rescale_factor = torch.div(0.5* torch.ones((search_batchsize, search_batchsize)).cuda(), norm_K+1e-3)

        # for i in searchnet.spike_map_list:
        #     K = x @ x.t()
        #     K2 = (1. - x) @ (1. - x.t())
        #     searchnet.K = searchnet.K + K.cpu().numpy() + K2.cpu().numpy()
        #
        #     # K1_0 = (x @ (1 - x.t()))
        #     # K0_1 = ((1-x) @ x.t())
        #     # K_total = (full_matrix - (K0_1 + K1_0))
        #     # # print(torch.unique(K_total))
        #     # searchnet.K = searchnet.K + (K_total.cpu().numpy())
        #     searchnet.num_actfun += 1


        s = []
        for name, module in searchnet.named_modules():
            # print(str(type(module)))
            if neuron_type in str(type(module)):
                module.register_forward_hook(computing_K_eachtime)

        for j in range(repeat):
            searchnet.count = -1
            searchnet.spike_map_list = []
            # for name, module in searchnet.named_modules():
                # if neuron_type in str(type(module)):
                #     searchnet.spike_map_list.append(torch.tensor([]).cuda())
            #
            with torch.no_grad():
                searchnet.K = np.zeros((search_batchsize, search_batchsize))
                searchnet.num_actfun = 0
                data_iterator = iter(train_data)
                inputs, targets = next(data_iterator)
                # inputs, targets = inputs.cuda(), targets.cuda()
                # print(targets[:40])
                with autocast():
                    # print(f'spike_list {(searchnet.spike_map_list[0].size())}, act {searchnet.num_actfun}')
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = searchnet(inputs)
                    # for idx, elem in enumerate(searchnet.spike_map_list):
                    #     # searchnet.spike_map_list[idx] = elem[:lw_ts[idx]]
                    #     # print(f'layer {idx} {(searchnet.spike_map_list[idx].size())}')
                    #     print(f'layer {idx} {(searchnet.spike_map_list[idx]).mean()}')
                    # print(f' repeat {j} {inputs[0,0,0,0]}')
                # for idx, i in enumerate(searchnet.spike_map_list):
                    # print(f'spike_list {idx} {(i.size())}')
                for i in searchnet.spike_map_list:
                    K = i @ i.t()
                    K2 = (1. - i) @ (1. - i.t())
                    searchnet.K = searchnet.K + K.cpu().numpy() + K2.cpu().numpy()
                    # print(i.size())

                # print(outputs)
                # print(searchnet.K)
                # print(searchnet.num_actfun)
                if det(searchnet.K/(searchnet.num_actfun)) != 0:
                    s.append(logdet(searchnet.K/(searchnet.num_actfun)))
                reset_net(searchnet)

        # scores.append(np.mean(s))
        # sum, len = 0, 0
        # for i in s:
        #     if i != 'nan' or i != '-inf':
        #         sum += i
        #         len += 1

        # print(sum/len)
        # print(s)
        score = np.mean(s)
        # history.append(con_mat)

        # print ("mean / var:", np.mean(scores), np.var(scores))
        # print ("max score:", max(scores))
        # best_idx = (np.argsort(scores))[-1]
        # best_policy = history[best_idx]
    return score

# import vgg
# import model_dict
#
# best_score = 0
# models = model_dict.models
# for key in models.keys():
#     layers, lw_ts = models[key][0], models[key][1]
# # layers = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "AM"] #[64, 128, 'M', 64, 512, 'M', 256, 128, 256, 'M', 64, 512, 64, 'M', 256, 128, 128, 'AM'] #[64, 128, 'M', 64, 512, 'M', 256, 128, 256, 'M', 64, 512, 64, 'M', 512, 512, 512, 'AM'] #[256, 256, 'M', 256, 256, 'M', 64, 64, 64, 'M', 64, 256, 128, 'M', 256, 64, 256, 'AM']  #[256, 256, 'M', 128, 64, 'M', 64, 64, 64, 'M', 512, 128, 256, 'M', 64, 64, 512, 'AM'] #[64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "AM"] #[64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "AM"] #[128, 256, 'M', 64, 64, 'M', 128, 128, 512, 'M', 128, 512, 512, 'M', 512, 256, 256, 'AM'] #[256, 128, 'M', 64, 256, 'M', 64, 128, 256, 'M', 256, 128, 512, 'M', 512, 512, 256, 'AM'] #[64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "AM"]
#
# # torch.manual_seed(1234)
# # lw_ts = [3]*13 #, 6, 3, 6, 5, 5, 5, 3, 7, 4, 3, 4, 3] # 6, 3, 6, 5, 5, 5, 3, 5, 4, 4, 4, 4] #[7, 4, 6, 5, 3, 3, 4, 4, 4, 6, 6, 4, 6] #[5, 5, 6, 7, 5, 7, 7, 5, 7, 6, 5, 7, 7] #[5, 4, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3]
#     network = vgg._vgg("custom", layers, False, False, False, num_linear_layers=layers[len(layers)-2],
#                                total_timestep=max(lw_ts), lw_timesteps=lw_ts, train_n=False, num_classes=100,
#                                dataset='imagenet100')
# # print(network)
#     score = SNN_score(network, 'imagenet100', lw_ts)
#     print(score)
#     if score > best_score:
#         best_score = score
# print(score)