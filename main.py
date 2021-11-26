from dataloader import dataloader


if __name__ == '__main__':
    train_loader = dataloader(dataset_fn='train.pkl')
    test_loader = dataloader(dataset_fn='test.pkl')

    # solver.train