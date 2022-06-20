import json
import os
from scipy.fftpack import fft
from torch.utils.data import Dataset
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np


def loadfile(filepath, known_classes):
    path_list = os.listdir(filepath)

    data_list = []
    label_list = []
    label_dict = {"0": 9, "1": 9, "2": 9, "3": 9, "4": 9, "5": 9, "6": 9, "7": 9, "8": 9, "9": 9}
    for i, c in enumerate(known_classes):
        label_dict[c] = i

    for index, filename in enumerate(path_list):
        f = open(os.path.join(filepath, filename), 'r', encoding="gbk")
        test_dict = json.load(f)
        data = test_dict["Data"]
        data_list.append(data)
        label_list.append(label_dict[test_dict["Result"]])
        f.close()

    index = [i for i in range(len(data_list))]
    np.random.shuffle(index)
    datas = np.array(data_list)[index]
    labels = np.array(label_list)[index]
    return datas, torch.LongTensor(labels)


def convert(data):
    # CWRU
    res = np.array(data).astype('float32')
    res.resize(1, 32, 32)
    return res


class MyDataset(Dataset):
    def __init__(self, datapath, known_classes, transform=None, target_transform=None):
        train_X, train_y = loadfile(datapath, known_classes)
        self.x_data_train = train_X
        self.y_data_train = train_y
        self.len = len(self.x_data_train)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        rawdata = np.array(self.x_data_train[index]).astype('float32')

        fft_y = fft(rawdata)
        fft_y[0] = 0
        abs_y = np.abs(fft_y)

        channel1 = convert(rawdata)
        channel2 = convert(abs_y)
        data = torch.cat([torch.from_numpy(channel1), torch.from_numpy(channel2), torch.from_numpy(channel1),
                          torch.from_numpy(channel2)], dim=0)

        return data, self.y_data_train[index]

    def __len__(self):
        return self.len


class CWRU:
    def __init__(self, is_gpu, args):
        self.known_classes = args.known_classes
        self.num_classes = len(self.known_classes)
        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)
        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.Resize(size=(patch_size, patch_size)),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(size=(patch_size, patch_size)),
        ])

        return train_transforms, val_transforms

    def get_dataset(self):
        test_path = 'datasets/known/' + self.__class__.__name__ + self.known_classes + '/test'
        train_path = 'datasets/known/' + self.__class__.__name__ + self.known_classes + '/train'
        trainset = MyDataset(train_path, self.known_classes, transform=self.train_transforms, target_transform=None)
        valset = MyDataset(test_path, self.known_classes, transform=self.val_transforms, target_transform=None)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class OpenSetDataset:
    def __init__(self, is_gpu, args, openset_dataset):
        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)
        self.trainset, self.valset = self.get_dataset(openset_dataset, args.known_classes)
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.Resize(size=(patch_size, patch_size)),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(size=(patch_size, patch_size)),
        ])

        return train_transforms, val_transforms

    def get_dataset(self, name, known_classes):
        valset = MyDataset('datasets/openset/' + name, known_classes, transform=self.val_transforms, target_transform=None)
        return None, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)
        return None, val_loader
