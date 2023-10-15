import argparse
import os
import torch.backends.cudnn as cudnn
from data.get_data import get_data_set
from termcolor import colored
from utils.engine import train_model, evaluation
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DistributedSampler
import utils.misc as misc
from configs import get_args_parser
import time
import datetime
from model.get_model import get_model
from utils.misc import NativeScalerWithGradNormCount as NativeScaler


def main():
    # distribution
    misc.init_distributed_mode(args)#分布式模式的初始化
    device = torch.device(args.device)#初始化设备
    #  cuDNN 是英伟达专门为深度神经网络所开发出来的 GPU 加速库，针对卷积、池化等等常见操作做了非常多的底层优化，比一般的 GPU 程序要快很多。
    cudnn.benchmark = True
    best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'dsc': 0, 'sp': 0, 'se': 0, 'iou': 0, 'hd': 0, 'asd': 0}

    if args.backbone_pretrain:
        load = False
    else:
        load = True

    model = get_model(args, load=load)
    model.to(device)
    #DistributedDataParallel（DDP）是一个支持多机多卡、分布式训练的深度学习工程方法。
    model_without_ddp = model
    train_set, val_set = get_data_set(args)
    #判断是否是分布式
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
        sampler_train = DistributedSampler(train_set)
        sampler_val = DistributedSampler(val_set, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=sampler_val, num_workers=args.num_workers,
                            shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    param_groups = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    print(optimizer)
    loss_scaler = NativeScaler()

    print(f"Start training for {args.num_epoch} epochs")
    start_time = time.time()

    for epoch in range(args.num_epoch):
        if epoch == 25:
            break
        if args.distributed:
            sampler_train.set_epoch(epoch)

        print(colored('Epoch %d/%d' % (epoch + 1, args.num_epoch), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        print("start training: ")
        train_model(args, epoch, model, train_loader, criterion, optimizer, loss_scaler, device)

        print("start evaluation: ")
        evaluation(args, best_record, epoch, model, model_without_ddp, val_loader, criterion, device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs('save_model/', exist_ok=True)
    main()