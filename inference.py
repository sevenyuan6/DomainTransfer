import torch
import argparse
import torch.backends.cudnn as cudnn
from configs import get_args_parser
from data.get_data import get_data_set
from model.get_model import get_model
from utils.vis_tools import show_single


def inference():
    device = torch.device(args.device)
    cudnn.benchmark = True
    model = get_model(args)
    model.to(device)

    save_name = args.output_dir + f"{args.dataset}_{args.model_name}_DT{args.test_domain_index}_{args.prediction_mode}" \
                + f"{'_pre' if args.backbone_pretrain else '_'}" + ".pt"
    checkpoint = torch.load(save_name, map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    print("load model finished")
    model.eval()

    train_set, val_set = get_data_set(args)
    loader = torch.utils.data.DataLoader(val_set, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    data = iter(loader).next()
    img = data["images"].cuda()
    mask = data["masks"].detach().numpy()[select]
    pred, f = model(img)
    _, pred = torch.max(pred, 1)
    pred = pred.detach().cpu().numpy()[select]

    mask[mask == 1] = 255
    pred[pred == 1] = 255
    show_single(mask)
    show_single(pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.batch_size = 2
    select = 0
    inference()