# COVID classification

## Requirements

- pytorch 1.10
- python 3.7
- cuda 10.2

```
.
├── checkpoint                        # 实验结果 checkpoints
│   ├── model_aug.pth
│   └── model_no_aug.pth
├── data                              # 数据集
│   ├── COVID
│   ├── NonCOVID
│   ├── test.pkl
│   ├── train.pkl
│   └── val.pkl
├── data_processing.py                # 数据集可视化、数据集划分用代码
├── dataloader.py                     # dataset 和 dataloader，包含数据增强
├── img                               # 可视化结果
│   ├── visualization_aug.png
│   ├── visualization_aug_final.png
│   ├── visualization_no_aug.png
│   └── visualization_test.png
├── main.py                           # 主函数：传递参数，设置训练/测试模式
├── model.py                          # CNN model
├── readme.md
├── solver.py                         # solver类，包含 train/test loop 实现
├── tensorboard                       # 实验 curves
│   ├── aug                           # 使用数据增强的实验结果
│   │   ├── AP
│   │   ├── AUC
│   │   ├── AUROC
│   │   ├── F1
│   │   ├── Loss
│   │   └── mAcc
│   ├── events.out.tfevents.1638791387.seeta-SYS-020
│   └── no_aug                        # 不用数据增强的结果
│       ├── AP
│       ├── AUC
│       ├── AUROC
│       ├── F1
│       ├── Loss
│       └── mAcc
├── train_aug.sh                      # 训练脚本
└── train_no_aug.sh                   # 训练脚本
```

## Usage

Training:

```shell
python main.py --train \
        --use_data_augment \
        --pretrain ./checkpoint/model_aug.pth \
        --check_path ./checkpoint/model_aug.pth \
        --epoch 500 \
        --lr 1e-4 \
        --writer_path ./tensorboard/aug
```

or:

```shell
zsh train_aug.sh
```

Inference:

```shell
python main.py --pretrain ./checkpoint/model_aug.pth
```