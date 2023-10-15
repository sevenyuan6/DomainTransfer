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
    device = torch.device(args.device)
    cudnn.benchmark = True
    best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'dsc': 0, 'sp': 0, 'se': 0, 'iou': 0, 'hd': 0, 'asd': 0}

    model = get_model(args)
    model.to(device)
    model_without_ddp = model

    save_name = args.output_dir + f"{args.dataset}_{args.model_name}_DT{args.test_domain_index}_{args.prediction_mode}" \
                + f"{'_pre' if args.backbone_pretrain else '_'}" + ".pt"
    checkpoint = torch.load(save_name, map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    print("load model finished")
    model.eval()

    train_set, val_set = get_data_set(args)
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=sampler_val, num_workers=args.num_workers,
                            shuffle=False)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    evaluation(args, best_record, 0, model, model_without_ddp, val_loader, criterion, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main()