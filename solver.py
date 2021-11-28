from pickle import load
from model import XuNet
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch import optim
import torch
from dataloader import dataloader
import torchmetrics
import os
import warnings

class Solver(object):
    def __init__(self):
        self.model = XuNet(in_ch=3, out_ch=2)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = 1e-4
        self.optim = optim.Adam(self.model.parameters(),
                                lr=self.lr,
                                weight_decay=5e-3)
        self.epochs = 100
        
        # Dataloaders
        self.train_loader = dataloader(dataset_fn='train.pkl', dataset_path='./data/', batch_size=8, num_workers=4, shuffle=True, drop_last=True)
        self.test_loader = dataloader(dataset_fn='test.pkl', dataset_path='./data/', batch_size=8, num_workers=4, shuffle=True, drop_last=True)
        # Devices
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Single GPU Mode, In: {}".format(self.device))
        self.model.to(self.device)

        self.check_path = './checkpoint/checkpoint.pth'
        self.writer_path = './tensorboard/'
        self.acc = torchmetrics.Accuracy().to(self.device)


    def train(self):
        writer = SummaryWriter(self.writer_path)
        for epoch in tqdm(range(self.epochs), desc="Training Epoch:"):
            for i, (img, gt) in enumerate(self.train_loader):
                img = img.to(self.device)
                gt = gt.to(self.device)
                # print(img.size())
                pred = self.model.forward(img)
                self.acc(pred.softmax(dim=1), gt.reshape(-1))
                loss = self.criterion(pred,
                                      torch.tensor(gt.view(-1, 1).squeeze(), dtype=torch.int64))
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

            if epoch % 10 == 9:
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optim.state_dict(),
                            'loss': loss}, self.check_path)
            train_acc = self.acc.compute()
            self.acc.reset()
            writer.add_scalars('Loss', {'train': loss},
                               global_step=epoch)
            writer.add_scalars('mAcc', {'train': train_acc.cpu()},
                               global_step=epoch)


    def test(self, load_path=None):
        if load_path != None:
            self.load_path = load_path
        if os.path.isfile(self.load_path):
            print("Load Network from {}".format(self.load_path))
            self.model.load_state_dict(torch.load(self.load_path)['model_state_dict'])
        else:
            warnings.warn("NO Existing Model!")
            return
        # for each mini-batch
        for (img, gt) in tqdm(self.test_loader, leave=False):
            batchsize = img.shape[0]
            # pic_cnt += batchsize
            img = img.to(self.device)
            gt = gt.to(self.device)
            pred = self.model.forward(img)
            loss = self.criterion(pred,
                                  torch.tensor(gt.view(-1, 1).squeeze(), dtype=torch.int64))
            # test_loss += loss.item() * batchsize
            self.acc(pred.softmax(dim=1), gt.reshape(-1))
            # cov += torch.sum(gt).cpu()
            # noncov += torch.sum(gt == 0).cpu()
            # save checkpoint
        # test_loss = test_loss / pic_cnt
        test_acc = self.acc.compute()
        self.acc.reset()
        print('accuracy:')
        print(test_acc)
        # Print the log info
        # outstr = r'Test: loss: %.6f, test acc: %.6f, cov/noncov: %d/%d' \
        #     % (test_loss*1.0, test_acc.cpu(), cov, noncov)

    def print_network(self):
        net = self.model
        # name = self.model_name
        # num_argss = 0
        # for argsmeter in net.argsmeters():
        #     num_argss += argsmeter.numel()
        print(net)
        # print(name)
        # print("The number of argsmeters is {}".format(num_argss))

if __name__ == '__main__':
    solver = Solver()
    solver.print_network()
    # solver.train()
    solver.test(load_path='./checkpoint/checkpoint.pth')