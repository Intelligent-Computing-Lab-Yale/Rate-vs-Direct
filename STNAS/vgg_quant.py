from typing import Union, List, Dict, Any, cast

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
import copy
import numpy as np

class QConvBN2dLIF(nn.Module):
    """ integerate the conv2d, BN, and LIF in the inference"""

    def __init__(self, conv_module, bn_module, lif_module, num_bits_w=4, num_bits_b=4, num_bits_u=4):
        super(QConvBN2dLIF, self).__init__()

        self.conv_module = conv_module
        self.lif_module = lif_module
        self.bn_module = bn_module

        self.num_bits_w = num_bits_w
        self.num_bits_b = num_bits_b
        self.num_bits_u = num_bits_u

        initial_beta = torch.Tensor(conv_module.weight.abs().mean() * 2 / math.sqrt((2 ** (self.num_bits_w - 1) - 1)))
        self.beta = nn.ParameterList([nn.Parameter(initial_beta) for i in range(1)]).cuda()


    def forward(self, x):
        # if self.training:
        if args.wq:
            if args.share:
                qweight, beta = w_q(self.conv_module.weight, self.num_bits_w, self.beta[0])
            else:
                qweight = b_q(self.conv_module.weight, self.num_bits_w)
        else:
            qweight = self.conv_module.weight
        x = F.conv2d(x, qweight, self.conv_module.bias, stride=self.conv_module.stride,
                     padding=self.conv_module.padding,
                     dilation=self.conv_module.dilation,
                     groups=self.conv_module.groups)
        x = self.bn_module(x)

        if args.share:
            s = self.lif_module(x, args.share, beta, bias=0)
        else:
            s = self.lif_module(x, args.share, 0, bias=0)


        return s


__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 10, init_weights: bool = True, dropout: float = 0.5, total_timestep: int = 6,
        n_linear: int = 512, train_n: bool = False, lw_timesteps: list = []) -> None:
        super().__init__()
        # torch.manual_seed(0)
        # self.total_timestep = total_timestep
        self.features = features
        self.total_timestep = max(lw_timesteps)
        self.classifier = nn.Linear(n_linear, num_classes)

        self._initialize_weights()
        # print(self.features)
        self.lw_timesteps = lw_timesteps
        self.layer_no = 0
        self.time = 0
        # print(self.lw_timesteps)
        self.integer = int(np.random.randint(low=0, high=10000, size=(1,)))
        self.train_n = train_n
        if self.train_n == True:
            print(self.features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # acc_voltage =
        output_list = []

        if self.train_n == True:
            # static_x = self.features[:3](x)
            # conv_list = [(1,3,7), (2,7,10), (3,10,14), (4,14,17), (5,17,20), (6,20,24), (7,24,27), (8,27,30), (9,30,34), (10,34,37), (11,37,40), (12,40,44)]
            # static_x = self.features[:4](x)
            static_x = self.features[:4](x)
            conv_list = [(1, 4, 8), (2, 8, 11), (3, 11, 15), (4, 15, 18), (5, 18, 21), (6, 21, 25), (7, 25, 28),
                         (8, 28, 31), (9, 31, 35), (10, 35, 38), (11, 38, 41), (12, 41, 45)]
            # conv_list = [(1, 1, 5), (2, 5, 9), (3, 9, 12), (4, 12, 16), (5, 16, 19), (6, 19, 22), (7, 22, 26),
            #              (8, 26, 29), (9, 29, 32), (10, 32, 36), (11, 36, 39), (12, 39, 42)]

        else:
            static_x = self.features[:3](x)
            conv_list = [(1, 3, 6), (2, 6, 8), (3, 8, 11), (4, 11, 13), (5, 13, 15), (6, 15, 18), (7, 18, 20),
                         (8, 20, 22), (9, 22, 25), (10, 25, 27), (11, 27, 29), (12, 29, 32)]

        input_list = [torch.tensor([0])] * 13

        for t in range(self.total_timestep):
            x = static_x
            self.time = t
            for index, layer_config in enumerate(conv_list):
                # print(layer_config)
                l_idx = layer_config[0]
                l_strt = layer_config[1]
                l_end = layer_config[2]

                # print(f'time {t} layer {index}')

                # print(f' timestep {t} layer {l_idx} layer_ts {self.lw_timesteps[l_idx]}')


                if t >= self.lw_timesteps[l_idx] and t < self.total_timestep:
                    # x = self.features[28:31](input_list[layer-1])
                    # print('reuse done')

                    x = input_list[l_idx]
                    # print(f' Replicating layer {l_idx} t {t} output {x[0,0,0,0].item(), x[0,1,0,0].item(), x[0,2,0,0].item()}')

                    # print(f'\t completed reuse branch')

                elif t < self.lw_timesteps[l_idx]:
                    # print(f' layer {layer} time {t}')
                    self.layer_no = index
                    x = self.features[l_strt:l_end](x)

                    # print(f' Original layer {l_idx} t {t} output {x[0,0,0,0].item(), x[0,1,0,0].item(), x[0,2,0,0].item()}')

                    # print(f'\t completed general branch')

                if t == self.lw_timesteps[l_idx] - 1:
                    # torch.save(x, './out_prev1')
                    # sp_inp = torch.load('./out_prev1')  # copy.deepcopy(out_prev)

                    # torch.save(x, './out_prev1'+str(self.integer))
                    # sp_inp = torch.load('./out_prev1'+str(self.integer))  # copy.deepcopy(out_prev)
                    input_list[l_idx] = x
                    # print(f' Saving layer {l_idx} t {t} output {x[0,0,0,0].item(), x[0,1,0,0].item(), x[0,2,0,0].item()}')
                    # input_list[l_idx] = copy.deepcopy(x)
                    # print(f'\t completed saving branch')

                # print(f'layer {l_idx}  sum {x.sum().item()}')
            # print(f'########## T {t} ###############')
            # for ip in input_list:
            #     print(ip.size())
            #     print(f'layer {l_idx} t {t} output {x[0,0,0,0].item(), x[0,1,0,0].item(), x[0,2,0,0].item()}')
            x = torch.flatten(x, 1)

            prob = self.classifier(x)
            # acc_voltage = acc_voltage + x
            output_list.append(prob)

            # acc_voltage = acc_voltage / self.total_timestep
            # print(f' output_list {sum(output_list)}')
        return output_list

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, 0, 0.06)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.normal_(m.weight, 0, 0.06)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, dataset = 'cifar10') -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'AM':
            # self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
            layers += [nn.AdaptiveAvgPool2d((1, 1))]
        else:
            v = cast(int, v)
            # print(in_channels, v)
            if in_channels == 3 and dataset == 'imagenet100':
                conv2d = nn.Conv2d(in_channels, v, kernel_size=7, padding=3, stride=2)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                if in_channels == 3 and dataset == 'imagenet100':
                    # maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding= 1)
                    layers += [conv2d, nn.BatchNorm2d(v), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                               # neuron.ParametricLIFNode(v_threshold=1.0, v_reset=0.0, init_tau=2.,
                               #                          surrogate_function=surrogate.ATan(),
                               #                          detach_reset=True)
                               neuron.LIFNode(v_threshold=0.5, v_reset=0.0, tau=1.2,  # 4/3.,
                                              surrogate_function=surrogate.ATan(), decay_input=False,
                                              detach_reset=True)
                               ]
                else:
                    layers += [conv2d, nn.BatchNorm2d(v),
                               # neuron.ParametricLIFNode(v_threshold=1.0, v_reset=0.0, init_tau=2.,
                               #                          surrogate_function=surrogate.ATan(),
                               #                          detach_reset=True)
                               neuron.LIFNode(v_threshold=0.5, v_reset=0.0, tau= 1.2, #4/3.,
                               surrogate_function=surrogate.ATan(), decay_input=False,
                               detach_reset=True)
                               ]
            else:
                if in_channels == 3 and dataset == 'imagenet100':
                    layers += [conv2d, nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                               # neuron.ParametricLIFNode(v_threshold=1.0, v_reset=0.0, init_tau=2.,
                               #                          surrogate_function=surrogate.ATan(),
                               #                          detach_reset=True)
                               neuron.LIFNode(v_threshold=0.5, v_reset=0.0, tau= 1.2, #4/3.,
                               surrogate_function=surrogate.ATan(), decay_input= False,
                               detach_reset=True)
                               ]

                else:
                    layers += [conv2d,
                               # neuron.ParametricLIFNode(v_threshold=1.0, v_reset=0.0, init_tau=2.,
                               #                          surrogate_function=surrogate.ATan(),
                               #                          detach_reset=True)
                               neuron.LIFNode(v_threshold=0.5, v_reset=0.0, tau= 1.2, #4/3.,
                               surrogate_function=surrogate.ATan(), decay_input=False,
                               detach_reset=True)
                               ]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch: str, cfgs: list, batch_norm: bool, pretrained: bool, progress: bool, num_linear_layers = 512, total_timestep=  5, lw_timesteps= [], train_n = False, num_classes = 10, dataset= '', **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs, batch_norm=batch_norm, dataset = dataset), num_classes = num_classes, total_timestep = max(lw_timesteps), n_linear= num_linear_layers, lw_timesteps = lw_timesteps, train_n = train_n, **kwargs) #
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13", "B", False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, **kwargs)