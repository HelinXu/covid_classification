import os
import random
import pickle
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    filename='./main.log',
    filemode='a',
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def split_dataset(dataset_path='./data/', train_ratio=0.8):
    covid_fn = os.listdir(os.path.join(dataset_path, 'COVID'))
    noncovid_fn = os.listdir(os.path.join(dataset_path, 'NonCOVID'))

    random.shuffle(covid_fn)
    random.shuffle(noncovid_fn)
    train_covid_fn = covid_fn[:int(train_ratio*len(covid_fn))]
    test_covid_fn = covid_fn[int(train_ratio*len(covid_fn)):]
    train_noncovid_fn = noncovid_fn[:int(train_ratio*len(noncovid_fn))]
    test_noncovid_fn = noncovid_fn[int(train_ratio*len(noncovid_fn)):]

    train_data = []
    test_data = []
    for fn in train_covid_fn:
        train_data.append((fn, 1))
    for fn in train_noncovid_fn:
        train_data.append((fn, 0))
    for fn in test_covid_fn:
        test_data.append((fn, 1))
    for fn in test_noncovid_fn:
        test_data.append((fn, 0))

    random.shuffle(train_data)
    random.shuffle(test_data)
    pickle.dump(train_data, open(
        os.path.join(dataset_path, 'train.pkl'), 'wb'))
    pickle.dump(test_data, open(os.path.join(dataset_path, 'test.pkl'), 'wb'))
    logging.info(
        f'Done split dataset. train {len(train_data)} = {len(train_covid_fn)} + {len(train_noncovid_fn)}; test {len(test_data)} = {len(test_covid_fn)} + {len(test_noncovid_fn)}')


class CovidDataset(Dataset):
    def __init__(self, opt, dataset_fn='train.pkl', dataset_path='./data/', shuffle=True):
        self.imgs = pickle.load(
            open(os.path.join(dataset_path, dataset_fn), 'rb'))
        self.dataset_path = dataset_path
        self.train = opt.train
        self.aug = opt.aug
        if shuffle:
            random.shuffle(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.imgs[idx][1]:
            img_path = os.path.join(
                self.dataset_path, 'COVID', self.imgs[idx][0])
        else:
            img_path = os.path.join(
                self.dataset_path, 'NonCOVID', self.imgs[idx][0])
        image = Image.open(img_path).convert('RGB')
        if self.train:
            if self.aug:
                transforms = T.Compose([
                    T.Resize(256),  # Resize(128)
                    T.RandomCrop(240),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                ])
            else:
                transforms = T.Compose([
                    T.Resize(256),  # Resize(128)
                    T.CenterCrop(240),
                    T.ToTensor(),
                ])
        else:
            transforms = T.Compose([
                T.Resize(256),  # Resize(128)
                T.CenterCrop(240),
                T.ToTensor(),
            ])
        image = transforms(image)
        return image, self.imgs[idx][1]


def dataloader(opt, dataset_fn='train.pkl', dataset_path='./data/', batch_size=64, num_workers=4, shuffle=True, drop_last=True):
    dataset = CovidDataset(opt=opt, dataset_fn=dataset_fn,
                           dataset_path=dataset_path)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            drop_last=drop_last)
    return dataloader


if __name__ == '__main__':
    # matplotlib.use('Agg')
    training_data = CovidDataset(dataset_fn='train.pkl')
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        name = {0: 'NonCOVID', 1: 'COVID'}
        plt.title(name[label])
        plt.axis("off")
        plt.imshow(np.asarray(img).transpose((1, 2, 0)), cmap="gray")
    plt.savefig('visualization.png')
