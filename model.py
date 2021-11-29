import torch.nn as nn


class conv_block(nn.Module): # 0.5*0.5 size
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, img):
        img = self.conv(img)
        # print(img.size())
        return img


class XuNet(nn.Module):
    def __init__(self, in_ch, out_ch=2):
        super(XuNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            conv_block(in_ch=in_ch, out_ch=16), # 240-120
            conv_block(in_ch=16, out_ch=32), # -60
            conv_block(in_ch=32, out_ch=64), # -30
            conv_block(in_ch=64, out_ch=128), # N*128*15*15
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True), # N*16*15*15
            nn.BatchNorm2d(16), # N*16*15*15
            nn.ReLU(inplace=True) # N*16*15*15
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(16*15*15, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_ch, bias=True)
        )

    def forward(self, img):
        # print(f'input shape: {img.size()}')
        x = self.cnn_layers.forward(img)
        # print(f'before reshape: {x.size()}')
        x = x.reshape(-1, 16*15*15)
        # print(f'after reshape: {x.size()}')
        x = self.fc_layers.forward(x)
        # print(f'return: {x.size()}')
        return x
