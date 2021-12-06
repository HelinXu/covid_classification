python main.py --train \
        --pretrain ./checkpoint/model_no_aug.pth \
        --check_path ./checkpoint/model_no_aug.pth \
        --epoch 500 \
        --lr 1e-5 \
        --writer_path ./tensorboard/no_aug
            