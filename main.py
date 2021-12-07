import argparse
from solver import Solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=500,
                        type=int, help='epochs to train.')
    parser.add_argument('--lr', default=1e-5,
                        type=float, help='epochs to train.')
    parser.add_argument('--pretrain', type=str,
                        default='./checkpoint/model.pth', help='path to pretrain model')
    parser.add_argument('--check_path', default='./checkpoint/model.pth',
                        type=str, help='path to save checkpoint')
    parser.add_argument('--writer_path', default='./tensorboard/', type=str,
                        help='path to save tensorboard outputs')
    parser.add_argument('--train', dest='train', action='store_true',
                        help='Turn on training mode. Default mode is test.')
    parser.add_argument('--use_data_augment', dest='aug',
                        action='store_true', help='Use data augmentation')
    opt = parser.parse_args()
    print(f'Train augmentation: {opt.aug}\n\
Train mode: {opt.train}\n\
Pretrain model: {opt.pretrain}\n\
Train epochs: {opt.epoch}')

    solver = Solver(opt=opt)
    solver.print_network()
    solver.draw_model()
    if opt.train:
        print('Starting to train...')
        solver.train()

    print('Starting to test...')
    solver.test(load_path=opt.pretrain)
