import argparse
import logging
import os
import random
import pickle
from dataloader import CovidDataset
import matplotlib.pyplot as plt
import torch
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    filename='./main.log',
    filemode='a',
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def split_dataset(dataset_path='./data/', train_ratio=0.8, val_ratio=0.1):
    covid_fn = os.listdir(os.path.join(dataset_path, 'COVID'))
    noncovid_fn = os.listdir(os.path.join(dataset_path, 'NonCOVID'))

    random.shuffle(covid_fn)
    random.shuffle(noncovid_fn)
    train_covid_fn = covid_fn[:int(train_ratio*len(covid_fn))]
    val_covid_fn = covid_fn[int(train_ratio*len(covid_fn)):int((val_ratio+train_ratio)*len(covid_fn))]
    test_covid_fn = covid_fn[int((val_ratio+train_ratio)*len(covid_fn)):]
    train_noncovid_fn = noncovid_fn[:int(train_ratio*len(noncovid_fn))]
    val_noncovid_fn = noncovid_fn[int(train_ratio*len(noncovid_fn)):int((val_ratio+train_ratio)*len(noncovid_fn))]
    test_noncovid_fn = noncovid_fn[int((val_ratio+train_ratio)*len(noncovid_fn)):]

    train_data = []
    val_data = []
    test_data = []
    for fn in train_covid_fn:
        train_data.append((fn, 1))
    for fn in train_noncovid_fn:
        train_data.append((fn, 0))
    for fn in val_covid_fn:
        val_data.append((fn, 1))
    for fn in val_noncovid_fn:
        val_data.append((fn, 0))
    for fn in test_covid_fn:
        test_data.append((fn, 1))
    for fn in test_noncovid_fn:
        test_data.append((fn, 0))

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    pickle.dump(train_data, open(
        os.path.join(dataset_path, 'train.pkl'), 'wb'))
    pickle.dump(test_data, open(os.path.join(dataset_path, 'test.pkl'), 'wb'))
    pickle.dump(val_data, open(os.path.join(dataset_path, 'val.pkl'), 'wb'))
    logging.info(
        f'Done split dataset. train {len(train_data)} = {len(train_covid_fn)} + {len(train_noncovid_fn)}; val {len(val_data)} = {len(val_covid_fn)} + {len(val_noncovid_fn)}; test {len(test_data)} = {len(test_covid_fn)} + {len(test_noncovid_fn)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', action='store_true',
                        help='Turn on training mode. Default mode is test.')
    parser.add_argument('--use_data_augment', dest='aug',
                        action='store_true', help='Use data augmentation')
    opt = parser.parse_args()

    training_data = CovidDataset(opt, dataset_fn='train.pkl')
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
    plt.savefig('visualization_no_aug.png')