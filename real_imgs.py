from __future__ import division
from __future__ import print_function


import torch
from torch._C import FloatStorageBase
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

class RealImage():

    def __init__(self) -> None:
        img_size = 32
        self.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)

    def real_imgs_row_data(self, label_list):
        # label_list は例えば ['airpalane', 'car'] とか ['frog']. つまり、ほしいラベルの文字列のリスト
        map_label_to_num = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9,
        }

        image_dict = {i: [] for i in range(10)}
        for i in range(len(self.train_set)):
            img, label = self.train_set[i][0], self.train_set[i][1]
            image_dict[label].append(img)

        result_list = []
        for label in label_list: #label_list = ['frog','bird']
            for i in range(len(image_dict[map_label_to_num[label]])):
                result_list.append(image_dict[map_label_to_num[label]][i])

        result_list = torch.stack(result_list, 0)

        return result_list

# 使い方の雰囲気
if __name__ == '__main__':
    r = RealImage()  # ここで一度だけダウンロードと transform が行われる。
    real_images = r.real_imgs_row_data()
    print(real_images.size())