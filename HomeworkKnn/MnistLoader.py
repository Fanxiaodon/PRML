import torch
import numpy as np
import struct
import os
from torch.utils.data import Dataset, DataLoader


class MyDataSet(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.train = train
        self.root_dir = root_dir
        self.transform = transform

        self.data = []
        self.target = []
        # 加载训练集
        if self.train:
            self._load_mnist(kind='train')
        else:
            self._load_mnist(kind='t10k')

    def _load_mnist(self, kind='train'):
        images_path = os.path.join(self.root_dir, '%s-images.idx3-ubyte' % kind)
        labels_path = os.path.join(self.root_dir, '%s-labels.idx1-ubyte' % kind)

        with open(images_path, 'rb') as img_path:
            magic, num, rows, cols = struct.unpack('>IIII', img_path.read(16))
            self.data = np.fromfile(img_path, dtype=np.uint8).reshape(-1, 28, 28)

        with open(labels_path, 'rb') as lb_path:
            magic, n = struct.unpack('>II', lb_path.read(8))
            self.target = np.fromfile(lb_path, dtype=np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        实现迭代方法
        :param index: index
        :return: sample 一个[img, label]列表，img.size = 1*28*28;label.size = 10
        """
        img = self.data[index]
        # 归一化
        img = img/255.0
        img = img[np.newaxis, :]
        label = self.target[index]
        # print(img.shape)
        # print(label.shape)
        if self.transform:
            [img, label] = self.transform([img, label])
        return [img, label]