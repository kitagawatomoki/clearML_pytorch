import torchvision.transforms as transforms
import torchvision
import torch
import os
import cv2
import albumentations as A
import random
import numpy as np

class Dataset(torch.utils.data.Dataset):

    def __init__(self, config, phase):
        super().__init__()

        self.phase = phase
        self.img_size = (config["size"], config["size"])
        self.aug = config["aug"]

        if phase == "train":
            self.indices = torchvision.datasets.CIFAR10(root='/Corpus3/b3study', train=True, download=False, transform=None)
        else:
            self.indices = torchvision.datasets.CIFAR10(root='/Corpus3/b3study', train=False, download=False, transform=None)

        self.len = len(self.indices)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["mean"],
                std=config["std"]
            )
        ])

        Blur = A.Blur(always_apply=False, p=0.25, blur_limit=(3, 15))
        CLAHE = A.CLAHE(always_apply=False, p=0.25, clip_limit=(1, 4), tile_grid_size=(8, 8))
        Downscale = A.Downscale(always_apply=False, p=0.25, scale_min=0.25, scale_max=0.25, interpolation=0)
        Rotate = A.Rotate(always_apply=False, p=0.25, limit=(-15, 15), interpolation=1, border_mode=3, value=(0, 0, 0), mask_value=None, method='largest_box', crop_border=False)
        ElasticTransform = A.ElasticTransform(alpha=1, 
                                            sigma=10, 
                                            alpha_affine=20, 
                                            interpolation=1, 
                                            border_mode=3, 
                                            value=None, 
                                            mask_value=None, 
                                            always_apply=False, 
                                            approximate=False, 
                                            same_dxdy=True, 
                                            p=0.25)
        HorizontalFlip = A.HorizontalFlip(always_apply=False, p=0.25)
        GaussNoise = A.GaussNoise(always_apply=False, p=0.25, var_limit=(10.0, 50.0), per_channel=True, mean=0.0)
        RandomBrightnessContrast = A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True)
        # ToSepia = A.ToSepia(always_apply=False, p=0.25)
        RandomSizedCrop = A.RandomSizedCrop(always_apply=False, p=1.0, min_max_height=(200, 224), height=224, width=224, w2h_ratio=1.0, interpolation=1)

        # aug_list = [Blur, CLAHE,Cutout,Downscale,Rotate,ElasticTransform,Flip,GaussNoise,RandomBrightnessContrast,ToSepia,RandomSizedCrop]
        self.use_list = [Downscale,Rotate,HorizontalFlip,RandomBrightnessContrast,RandomSizedCrop]
        self.select_list1 = [Blur, CLAHE, GaussNoise]
        self.select_list2 = [ElasticTransform, Rotate]

    def __getitem__(self, index):
        img, label = self.indices[index]

        img = np.array(img, dtype=np.uint8)

        img = cv2.resize(img, self.img_size)

        if self.phase =="train" and self.aug:
            aug_list =  [random.choice(self.select_list1)] + [random.choice(self.select_list2)] + self.use_list
            compose = A.Compose(aug_list)
            transformed = compose(image=img)
            img = transformed["image"]
            img = cv2.resize(img, self.img_size)

        img = self.transforms(img)
        label = torch.tensor(label)

        return img, label

    def __len__(self):
        return self.len