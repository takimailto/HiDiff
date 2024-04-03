import glob
import random
import numpy as np
from skimage.io import imread

from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as TF


def random_crop(input_img, label_img, size):
    """
    Crop random section from image
    size: int or list of int
        when it's a list, it should include x, y values
    Use for training
    """
    if isinstance(size, int):
        size = [size]*2
    assert len(size) == 2
    """draw x,y,z coords
    """
    coords = [0]*2
    for i in range(2):
        coords[i] = np.random.choice(label_img.shape[i] - size[i])
    x, y = coords
    ex = input_img[x:x+size[0], y:y+size[1],:]

    label = label_img[x:x+size[0], y:y+size[1]]
    return ex, label


class DriveDataset(data.Dataset):
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

        x, y = imread(input_ID), imread(target_ID)[0]
        x, y = random_crop(x, y, [self.img_height, self.img_width])

        x = self.transform_input(x)
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
