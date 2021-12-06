python main.py --train \
        --use_data_augment \
        --pretrain ./checkpoint/model_aug.pth \
        --check_path ./checkpoint/model_aug.pth \
        --epoch 1000 \
        --lr 3e-6 \
        --writer_path ./tensorboard/aug
            