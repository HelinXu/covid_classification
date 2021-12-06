python main.py --train \
        --use_data_augment \
        --pretrain ./checkpoint/model_aug.pth \
        --check_path ./checkpoint/model_aug.pth \
        --epoch 500 \
        --lr 1e-4 \
        --writer_path ./tensorboard/aug
            