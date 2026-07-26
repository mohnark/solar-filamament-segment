import albumentations as A
from albumentations.pytorch import ToTensorV2
 
IMG_SIZE = 512
 
# Single-channel, but the resnet34 encoder is ImageNet-pretrained, and smp
# adapts it to in_channels=1 by summing the RGB conv weights — i.e. the
# encoder still expects ImageNet-normalized input. These are the ImageNet
# RGB stats collapsed to luminance (0.299R + 0.587G + 0.114B), so the input
# distribution matches what the pretrained weights were trained on. Plain
# 0.5/0.5 shifted and scaled it away from that.
GRAYSCALE_MEAN = (0.449,)
GRAYSCALE_STD = (0.226,)
 
 
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD),
        ToTensorV2(),
    ])
 
 
def get_val_transform():
    return A.Compose([
        A.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD),
        ToTensorV2(),
    ])
 