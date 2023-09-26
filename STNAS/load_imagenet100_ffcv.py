from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from pathlib import Path
import numpy as np
import torch

def create_train_loader(train_dataset, num_workers, batch_size,
                        distributed, in_memory, score_mode = 0):
    # this_device = f'cuda:{gpu}'
    train_path = Path(train_dataset)
    assert train_path.is_file()

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    DEFAULT_CROP_RATIO = 224 / 256

    # res = get_resolution(epoch=0)
    # decoder = RandomResizedCropRGBImageDecoder((res, res))
    image_pipeline: List[Operation] = [
        RandomResizedCropRGBImageDecoder((224, 224)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device('cuda'), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device('cuda'), non_blocking=True)
    ]

    torch.torch.manual_seed(0)
    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    if score_mode == 1:
        order = OrderOption.SEQUENTIAL
        print('sequential')
    loader = Loader(train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=True,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed)

    return loader


# @param('data.val_dataset')
# @param('data.num_workers')
# @param('validation.batch_size')
# @param('validation.resolution')
# @param('training.distributed')
def create_val_loader(val_dataset, num_workers, batch_size,
                     distributed):
    # this_device = f'cuda:{torch.device('cuda')'}')
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    DEFAULT_CROP_RATIO = 224 / 256

    val_path = Path(val_dataset)
    assert val_path.is_file()
    res_tuple = (224, 224)
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(torch.device('cuda'), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device('cuda'),
                 non_blocking=True)
    ]

    loader = Loader(val_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.SEQUENTIAL,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed)
    return loader