import torch
from loss import *
from lung_dataset import LungDataset
from baseline_network import *
from os.path import join
from train import train_network


data_directory = '/home/dell/Desktop/Lung_Segmentation/ChestXray-256'

if __name__ == '__main__':
    _network = BaselineNetwork().cuda()
    _network_load_name = '/home/dell/Desktop/Lung_Segmentation/saved_networks/baseline.pth'
    _network.load_state_dict(torch.load(_network_load_name))
    _dataset = LungDataset(data_directory)
    _data_loader = torch.utils.data.DataLoader(_dataset, batch_size=4, drop_last=True)
    train_network(_network,
                  _data_loader,
                  DiceLoss(),
                  join(data_directory, 'val'),
                  validation_lossFn=DiceLoss(),
                  validation_period=200)