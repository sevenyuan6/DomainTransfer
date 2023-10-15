import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="PSP-Net Network", add_help=False)

    # train settings
    parser.add_argument("--mode_eval", type=str, default="no_beta", help="this will implement cross domain validation.")
    parser.add_argument("--dataset", type=str, default="brain")
    parser.add_argument("--test_domain_index", type=int, default=3)
    parser.add_argument("--backbone_pretrain", type=str, default=True)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_domains", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=320)
    parser.add_argument("--prediction_mode", type=str, default="cup")
    parser.add_argument("--model_name", type=str, default="deeplabv3plus_mobilenet")
    parser.add_argument("--output_stride", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--setting_size", type=int, default=[256, 256], help="input size of data set image.")
    parser.add_argument("--data_root", type=str, default="/home/wangbowen/DATA/Domain", help="Path to the directory containing images.")
    parser.add_argument("--num_epoch", type=int, default=40, help="Number of training steps.")

    # optimizer settings
    parser.add_argument("--lr", type=float, default=0.0001, help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="learning rate decay.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay.")

    # other settings
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument("--print_freq", type=str, default=5, help="print frequency.")
    parser.add_argument('--output_dir', default='save_model/', help='path where to save, empty for no saving')

    # training parameters
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument("--device", type=str, default='cuda:0', help="choose gpu device.")
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser