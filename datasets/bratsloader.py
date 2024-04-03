import torch
import torch.nn
import numpy as np
import os
import os.path
from scipy.ndimage.interpolation import zoom


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, list_dir, transform=None, test_flag=True):
        super().__init__()
        self.transform = transform
        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = np.load(list_dir, allow_pickle=True)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = np.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, path)
        else:
            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label[label==4] = 3
            
            image_zoom = np.zeros((4, 256, 256))
            for i in range(4):
                image_zoom[i] = zoom(image[i], (256 / 224, 256 / 224), order=3)
            image = image_zoom
            label = zoom(label[0], (256 / 224, 256 / 224), order=0)[None]
            
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            label = torch.nn.functional.one_hot(torch.from_numpy(label).long(), num_classes=4)[0].permute(2,0,1)
            return (torch.from_numpy(image).float(), label.float())

    def __len__(self):
        return len(self.database)
