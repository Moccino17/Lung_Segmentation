import sys

import torch
import shutil
from loss import *
import os
from PIL import Image
import numpy as np
import cv2
from random import randint
from networks import TestNet
from tqdm import tqdm

test_data_path = '/home/mohamed/Desktop/Lung_Segmentation/ChestXray-256/test'


def load_image(image_number):
    _image_name = 'cxrimage_' + str(image_number) + '.png'
    image_path = os.path.join(test_data_path, 'image', _image_name)
    image = np.array(Image.open(image_path).convert('L'))
    return torch.FloatTensor(image).view(1, 1, image.shape[0], image.shape[1]).cuda()


def load_mask(mask_number):
    _mask_name = 'cxrmask_' + str(mask_number) + '.jpeg'
    mask_path = os.path.join(test_data_path, 'mask', _mask_name)
    mask = np.array(Image.open(mask_path).convert('L'))
    mask = cv2.GaussianBlur(mask, (3, 3), cv2.BORDER_DEFAULT) > 128
    return torch.FloatTensor(mask).view(1, 1, mask.shape[0], mask.shape[1]).cuda()


def save_outputs_to_report(report_dir, image_number, image, pred, mask):
    sample_dir = os.path.join(report_dir, 'samples', str(image_number))
    os.mkdir(sample_dir)

    convert = lambda x: x.detach().cpu().view(x.shape[2], x.shape[3]).numpy()
    Image.fromarray(convert(image)).convert('RGB')\
        .save(os.path.join(sample_dir, 'image_' + str(image_number) + '.png'))
    Image.fromarray(convert(mask * 255)).convert('RGB')\
        .save(os.path.join(sample_dir, 'mask_' + str(image_number) + '.png'))
    Image.fromarray(convert(pred * 255)).convert('RGB')\
        .save(os.path.join(sample_dir, 'prediction_' + str(image_number) + '.png'))


def print_metrics_to_file(report_dir, acc, sens, spec, dice, jaccard):
    with open(os.path.join(report_dir, 'metrics.txt'), 'w') as metrics_file:
        metrics_file.write('\nAccuracy = ' + str(acc))
        metrics_file.write('\nSensitivity = ' + str(sens))
        metrics_file.write('\nSpecificity = ' + str(spec))
        metrics_file.write('\nDice score = ' + str(dice))
        metrics_file.write('\nJaccard index = ' + str(jaccard))


def print_network_test_results(network, report_name):
    start_img = 16932
    num_images = 2116
    dice = accuracy = specificity = sensitivity = jaccard = 0

    report_dir = './test_results/' + str(report_name)
    if os.path.exists(report_dir):
        shutil.rmtree(report_dir, ignore_errors=False, onerror=None)
    os.mkdir(report_dir)
    os.mkdir(os.path.join(report_dir, 'samples'))

    for num in tqdm(range(start_img, num_images + start_img - 1)):
        image = load_image(num)
        mask = load_mask(num)
        with torch.no_grad():
            pred = network(image)

        accuracy += accuracy_score(pred, mask)
        sensitivity += sensitivity_score(pred, mask)
        specificity += specificity_score(pred, mask)
        dice += dice_score(pred, mask)
        jaccard += jaccard_index(pred, mask)

        if randint(1, 100) == 1:
            save_outputs_to_report(report_dir, num, image, pred, mask)

    accuracy /= num_images
    sensitivity /= num_images
    specificity /= num_images
    dice /= num_images
    jaccard /= num_images

    print_metrics_to_file(report_dir, accuracy, sensitivity, specificity, dice, jaccard)
    print('Accuracy = ' + str(accuracy))
    print('\nSensitivity = ' + str(sensitivity))
    print('\nSpecificity = ' + str(specificity))
    print('\nDice score = ' + str(dice))
    print('\nJaccard index = ' + str(jaccard))


if __name__ == '__main__':
    network_class = TestNet
    _network_path = './saved_networks/testNet.pth'
    report_name = 'TestNet1'

    network = network_class().cuda()
    network.load_state_dict(torch.load(_network_path))
    print_network_test_results(network, report_name)