import torch
from loss import *
from lung_dataset import LungDataset
from networks import *
from os.path import join
from train import train_network


data_directory = '/home/mohamed/Desktop/Lung_Segmentation/ChestXray-256'

if __name__ == '__main__':
    _network = TestNet(n=32).cuda()
    _network_load_name = '/home/mohamed/Desktop/Lung_Segmentation/saved_networks/testNet.pth'
    _dataset = LungDataset(data_directory)
    _data_loader = torch.utils.data.DataLoader(_dataset, batch_size=4, drop_last=True)
    train_network(_network,
                  _data_loader,
                  DiceLoss(),
                  join(data_directory, 'val'),
                  validation_lossFn=DiceLoss(),
                  validation_period=4233)