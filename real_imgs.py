from __future__ import division
from __future__ import print_function
from os import X_OK

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid, save_image

from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy


from utils import *
from models import *
from fid_score import *
from inception_score import *

import random


"""
ラベル「0」： airplane（飛行機）
ラベル「1」： automobile（自動車）
ラベル「2」： bird（鳥）
ラベル「3」： cat（猫）
ラベル「4」： deer（鹿）
ラベル「5」： dog（犬）
ラベル「6」： frog（カエル）
ラベル「7」： horse（馬）
ラベル「8」： ship（船）
ラベル「9」： truck（トラック）
"""


def test_real_imgs_row_data(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck):
    '''
    CIFAR10の中から、好きな種類の画像のみを、1つのリストにしてくれます。
    
    関数利用時は、

    airplane = 0
    automobile = 0
    bird = 1
    cat = 0
    deer = 0
    dog = 0
    frog = 1
    horse = 0
    ship = 0
    truck = 0

    のように、予め配列化したい種類の画像に１のフラグを立てて置く必要があります。
    '''
    img_size = 32
    transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_image_set = []

    for i in range(len(train_set)):
        if airplane == 1:
            if train_set[i][1] == 0:
                cifar_image_set.append(train_set[i][0])
        elif automobile == 1:
            if train_set[i][1] == 1:
                cifar_image_set.append(train_set[i][0])
        elif bird == 1:
            if train_set[i][1] == 2:
                cifar_image_set.append(train_set[i][0])
        elif cat == 1:
            if train_set[i][1] == 3:
                cifar_image_set.append(train_set[i][0])
        elif deer == 1:
            if train_set[i][1] == 4:
                cifar_image_set.append(train_set[i][0])
        elif dog == 1:
            if train_set[i][1] == 5:
                cifar_image_set.append(train_set[i][0])
        elif frog == 1:
            if train_set[i][1] == 6:
                cifar_image_set.append(train_set[i][0])
        elif horse == 1:
            if train_set[i][1] == 7:
                cifar_image_set.append(train_set[i][0])
        elif ship == 1:
            if train_set[i][1] == 8:
                cifar_image_set.append(train_set[i][0])
        elif truck == 1:
            if train_set[i][1] == 9:
                cifar_image_set.append(train_set[i][0])

    print(f'(before) type(cifar_image_set): {type(cifar_image_set)}')  # List
    print(f'len(cifar_image_set): {len(cifar_image_set)}')
    cifar_image_set = torch.stack(cifar_image_set, 0)
    print(f'(after) type(cifar_image_set): {type(cifar_image_set)}')  # List
    print(f'cifar_image_set.size(): {cifar_image_set.size()}')
    print("complete create cifar_image_set")
    return cifar_image_set


def test_a_real_img(list):
    """
    指定された種類数*`2500枚の中から、1枚ランダムで取り出してくれる関数
    """
    a_real_imgs = list[random.randrange(1,len(list))]
    return a_real_imgs


