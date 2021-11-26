import os
import random
import pickle
import logging

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
