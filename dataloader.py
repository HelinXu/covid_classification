import os
import random
import pickle
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image

logging.basicConfig(
                level=logging.DEBUG,
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
    pickle.dump(train_data, open(os.path.join(dataset_path, 'train.pkl'), 'wb'))
    pickle.dump(test_data, open(os.path.join(dataset_path, 'test.pkl'), 'wb'))
    logging.info(f'Done split dataset. train {len(train_data)} = {len(train_covid_fn)} + {len(train_noncovid_fn)}; test {len(test_data)} = {len(test_covid_fn)} + {len(test_noncovid_fn)}')


class CovidDataset(Dataset):
    def __init__(self, dataset_fn='train.pkl', dataset_path='./data/', shuffle=True):
        self.imgs = pickle.load(open(os.path.join(dataset_path, dataset_fn), 'rb'))
        self.dataset_path = dataset_path
        if shuffle: random.shuffle(self.imgs)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        if self.imgs[idx][1]: img_path = os.path.join(self.dataset_path, 'COVID', self.imgs[idx][0])
        else: img_path = os.path.join(self.dataset_path, 'NonCOVID', self.imgs[idx][0])
        image = read_image(img_path)
        # TODO: transform
        return image, self.imgs[idx][1]

        
def dataloader(dataset_fn='train.pkl', dataset_path='./data/', batch_size=8, num_workers=4, shuffle=True, drop_last=True):
    dataset = CovidDataset(dataset_fn=dataset_fn, dataset_path=dataset_path)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            drop_last=drop_last)
    return dataloader

# print(CovidDataset().__getitem__(35))