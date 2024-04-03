import glob
import random
import numpy as np
from PIL import Image
from skimage.io import imread

from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as TF


class CVCDataset(data.Dataset):
    def __init__(
        self,
        img_path,
        depth_path,
        img_height=256,
        img_width=256,
        transform_input=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
                ),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        transform_target=transforms.Compose(
            [transforms.ToTensor()]
        ),
        hflip=True,
        vflip=True,
        affine=True,
    ):
        input_paths = np.array(sorted(glob.glob(img_path)))
        target_paths = np.array(sorted(glob.glob(depth_path)))

        self.img_height = img_height
        self.img_width = img_width
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]
        target_ID = self.target_paths[index]

        x, y_ = imread(input_ID), imread(target_ID)

        y = np.zeros((self.img_height, self.img_width), dtype=np.bool_)
        pillow_x = Image.fromarray(x)
        pillow_x = pillow_x.resize((self.img_height, self.img_width))
        pillow_y = Image.fromarray(y_)
        pillow_y = pillow_y.resize((self.img_height, self.img_width), resample=Image.LANCZOS)
        y_ = np.array(pillow_y)
        for i in range(self.img_height):
            for j in range(self.img_width):
                if y_[i,j] >= 127:
                    y[i,j] = 1

        x = self.transform_input(pillow_x)
        y = self.transform_target(y)

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)

        if self.affine:
            angle = random.uniform(-180.0, 180.0)
            h_trans = random.uniform(-self.img_height / 8, self.img_width / 8)
            v_trans = random.uniform(-self.img_height / 8, self.img_width / 8)
            scale = random.uniform(0.5, 1.5)
            shear = random.uniform(-22.5, 22.5)
            x = TF.affine(x, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            y = TF.affine(y, angle, (h_trans, v_trans), scale, shear, fill=0.0)
        return (x.float(), (y>0).float())
