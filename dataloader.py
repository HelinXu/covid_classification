import os
import random
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


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
                    T.RandomRotation(degrees=(-10, 10)),
                    T.RandomCrop(240),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
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
