import albumentations as A
from albumentations.pytorch import ToTensorV2
 
IMG_SIZE = 512
 
# hardcoded for now, single-channel grayscale
GRAYSCALE_MEAN = (0.5,)
GRAYSCALE_STD = (0.5,)
 
 
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
 