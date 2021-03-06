from model import XuNet
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch import optim
import torch
from dataloader import dataloader
import torchmetrics
import os
import warnings
import hiddenlayer


class Solver(object):
    def __init__(self, opt):
        self.model = XuNet(in_ch=3, out_ch=2)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = opt.lr
        self.optim = optim.Adam(self.model.parameters(),
                                lr=self.lr,
                                weight_decay=5e-3)
        self.epochs = opt.epoch

        # Dataloaders
        self.train_loader = dataloader(opt=opt, dataset_fn='train.pkl', dataset_path='./data/',
                                       batch_size=64, num_workers=4, shuffle=True, drop_last=True)
        self.test_loader = dataloader(opt=opt, dataset_fn='test.pkl', dataset_path='./data/',
                                      batch_size=64, num_workers=0, shuffle=True, drop_last=False)
        self.val_loader = dataloader(opt=opt, dataset_fn='val.pkl', dataset_path='./data/',
                                      batch_size=64, num_workers=0, shuffle=True, drop_last=False)
        # Devices
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print("Single GPU Mode, In: {}".format(self.device))
        self.model.to(self.device)

        self.check_path = opt.check_path
        self.writer_path = opt.writer_path

        # Metrics
        self.acc = torchmetrics.Accuracy().to(self.device)
        self.auc = torchmetrics.AUC(reorder=True).to(self.device)
        self.auroc = torchmetrics.AUROC(num_classes=2).to(self.device)
        self.ap = torchmetrics.AveragePrecision(num_classes=2).to(self.device)
        self.f1 = torchmetrics.F1(num_classes=2).to(self.device)
        self.writer = SummaryWriter(self.writer_path)

    def train(self):
        writer = self.writer
        for epoch in tqdm(range(self.epochs), desc="Training Epoch:"):
            for i, (img, gt) in enumerate(self.train_loader):
                img = img.to(self.device)
                gt = gt.to(self.device)
                # print(img.size())
                pred = self.model.forward(img)
                # Metrics
                self.acc(pred.softmax(dim=1), gt.reshape(-1))
                self.auroc(pred.softmax(dim=1), gt.reshape(-1))
                self.ap(pred.softmax(dim=1), gt.reshape(-1))
                self.f1(pred.softmax(dim=1), gt.reshape(-1))
                self.auc(pred.max(1, keepdim=True)[1], gt.reshape(-1))

                loss = self.criterion(pred,
                                      torch.tensor(gt.view(-1, 1).squeeze(), dtype=torch.int64))
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

            train_acc = self.acc.compute().cpu()
            self.acc.reset()
            train_auc = self.auc.compute().cpu()
            self.auc.reset()
            train_auroc = self.auroc.compute().cpu()
            self.auroc.reset()
            train_ap = self.ap.compute().cpu()
            self.ap.reset()
            train_f1 = self.f1.compute().cpu()
            self.f1.reset()
            writer.add_scalars('Loss', {'train': loss},
                               global_step=epoch)
            writer.add_scalars('mAcc', {'train': train_acc}, global_step=epoch)
            writer.add_scalars('AUC', {'train': train_auc}, global_step=epoch)
            writer.add_scalars(
                'AUROC', {'train': train_auroc}, global_step=epoch)
            writer.add_scalars('AP', {'train': train_ap}, global_step=epoch)
            writer.add_scalars('F1', {'train': train_f1}, global_step=epoch)
            if epoch % 10 == 9:
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optim.state_dict(),
                            'loss': loss}, self.check_path)
                self.test(epoch=epoch, val=True)

    def test(self, load_path=None, epoch=0, val=False):
        self.model.eval()
        writer = self.writer
        if load_path != None:
            self.load_path = load_path
            if os.path.isfile(self.load_path):
                print("Load Network from {}".format(self.load_path))
                self.model.load_state_dict(torch.load(
                    self.load_path)['model_state_dict'])
            else:
                warnings.warn("NO Existing Model!")
                return
        # for each mini-batch
        if val: loader = self.val_loader
        else: loader = self.test_loader
        for (img, gt) in tqdm(loader, leave=False):
            batchsize = img.shape[0]
            # pic_cnt += batchsize
            img = img.to(self.device)
            gt = gt.to(self.device)
            pred = self.model.forward(img)
            loss = self.criterion(pred,
                                  torch.tensor(gt.view(-1, 1).squeeze(), dtype=torch.int64))

            # Metrics
            self.acc(pred.softmax(dim=1), gt.reshape(-1))
            self.auroc(pred.softmax(dim=1), gt.reshape(-1))
            self.ap(pred.softmax(dim=1), gt.reshape(-1))
            self.f1(pred.softmax(dim=1), gt.reshape(-1))
            self.auc(pred.max(1, keepdim=True)[1], gt.reshape(-1))
        result = {}

        test_acc = self.acc.compute().cpu()
        result['acc'] = test_acc
        self.acc.reset()
        test_auc = self.auc.compute().cpu()
        result['auc'] = test_auc
        self.auc.reset()
        test_auroc = self.auroc.compute().cpu()
        result['auroc'] = test_auroc
        self.auroc.reset()
        test_ap = self.ap.compute().cpu()
        result['ap'] = test_ap
        self.ap.reset()
        test_f1 = self.f1.compute().cpu()
        result['f1'] = test_f1
        self.f1.reset()
        if val:
            writer.add_scalars('Loss', {'test': loss},
                            global_step=epoch)
            writer.add_scalars('mAcc', {'test': test_acc},
                            global_step=epoch)
            writer.add_scalars('AUC', {'test': test_auc},
                            global_step=epoch)
            writer.add_scalars('AUROC', {'test': test_auroc},
                            global_step=epoch)
            writer.add_scalars('AP', {'test': test_ap},
                            global_step=epoch)
            writer.add_scalars('F1', {'test': test_f1},
                            global_step=epoch)
        print(result)

    def print_network(self):
        net = self.model
        print(net)

    def draw_model(self):
        g = hiddenlayer.build_graph(self.model.cuda().eval(), torch.zeros([1, 3, 240, 240]).cuda())
        g.save('model', format='png')
