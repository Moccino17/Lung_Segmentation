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
        mask = np.array(Image.open(mask).convert('L'))
        mask = cv2.GaussianBlur(mask,(3,3), cv2.BORDER_DEFAULT) > 128
        return np.array(Image.open(image).convert('L')), mask