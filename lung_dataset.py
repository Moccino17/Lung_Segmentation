import torch
from PIL import Image
from os.path import join
import numpy as np
import cv2


class LungDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.length = 16932
        self.directory = directory

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = join(self.directory, 'train', 'image', 'cxrimage_' + str(index) + '.png')
        mask = join(self.directory, 'train', 'mask', 'cxrmask_' + str(index) + '.jpeg')
        image = np.array(Image.open(image).convert('L'))
        image = image.reshape(1, *image.shape)
        _x = np.zeros((1, 256, 256), dtype=np.uint8)
        image = np.concatenate((image, _x, _x), axis=0)

        mask = np.array(Image.open(mask).convert('L'))
        mask = mask.reshape(1, *mask.shape)
        mask = cv2.GaussianBlur(mask, (3, 3), cv2.BORDER_DEFAULT) > 128

        return image, mask
