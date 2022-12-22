import torch
from tqdm import tqdm
from PIL import Image
from loss import *
from os.path import join
import numpy as np
from copy import deepcopy
from skimage.io import imshow_collection, imshow
import matplotlib.pyplot as plt
import sys
import cv2


def validate_network(network, validation_data_path, errFn=DiceLoss()):
    loss = 0
    # num_images = os.listdir(join(validation_data_path, 'image')).__len__()
    num_images = 10
    start_image = 19048
    for img_number in range(start_image, start_image + num_images + 1):
        image_path = join(validation_data_path, 'image', 'cxrimage_' + str(img_number) + '.png')
        mask_path = join(validation_data_path, 'mask', 'cxrmask_' + str(img_number) + '.jpeg')

        image = np.array(Image.open(image_path).convert('L'))
        image = image.reshape(1, *image.shape)
        x = np.zeros((1, 256, 256), dtype=np.uint8)
        image = np.concatenate((image, x, x), axis=0)
        image = torch.Tensor(image).view(1, *image.shape).float().cuda()

        mask = np.array(Image.open(mask_path).convert('L'))
        mask = cv2.GaussianBlur(mask,(3,3), cv2.BORDER_DEFAULT) > 128
        mask = torch.Tensor(mask).view(1, 1, *mask.shape).float().cuda()

        loss += errFn(mask, network(image)).data.item()

    return loss / num_images


def train_network(network,
                  data_loader,
                  loss_function,
                  validation_data_path,
                  learning_rate=1e-3,
                  number_epochs=50,
                  validation_period=None,
                  validation_lossFn=torch.nn.MSELoss()):

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    count = 0
    sum_loss = 0
    for epoch in range(number_epochs):
        for images, masks in tqdm(data_loader):
            images = images.float().cuda()
            masks = masks.float().cuda()
            # =================== Forward =====================
            output = network(images)
            loss = loss_function(output, masks)
            sum_loss += loss.data.item()
            # =================== Backward ====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ========================== DEBUG ==========================
            count += 1
            if count == validation_period:
                train_loss = sum_loss / (validation_period * images.shape[1])
                val_loss = validate_network(network, validation_data_path, errFn=validation_lossFn)
                print('\ntraining loss = {:.2f}'.format(train_loss))
                print('validation loss = {:.2f}'.format(val_loss))
                count = 0
                sum_loss = 0
                # first_slice = lambda x : x.detach().cpu().numpy()[0, 0, :, :]
                # imshow_collection([first_slice(masks),
                #                    first_slice(output),
                #                    first_slice(images * (1-masks))])
                # plt.show()
                torch.save(network.state_dict(), 'saved_networks/baseline.pth', _use_new_zipfile_serialization=False)
