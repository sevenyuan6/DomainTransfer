import torch
import argparse
import torch.backends.cudnn as cudnn
from configs import get_args_parser
from PIL import Image
from model.get_model import get_model
from data.brain.brain_datas import Brain_Tumor
from utils.vis_tools import show_single
from data.data_utils import get_standard_transformations, get_joint_transformations_val
from data.data_utils import img_mask_crop_binary
import numpy as np
import cv2


def draw_counter(img, mask, type):
    contours_cup, _ = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours_cup, -1, (255, 0, 0), 2)
    show_single(img, type)


def get_data():
    train, test = Brain_Tumor(args).get_data()
    select_ed = train[select_img]

    t1 = select_ed["t1"][select_slice][0]
    t2 = select_ed["t2"][select_slice][0]
    t1ce = select_ed["t1ce"][select_slice][0]
    flair = select_ed["flair"][select_slice][0]
    seg = select_ed["flair"][select_slice][1]

    t1 = Image.open(t1).convert('RGB')
    t2 = Image.open(t2).convert('RGB')
    t1ce = Image.open(t1ce).convert('RGB')
    flair = Image.open(flair).convert('RGB')
    seg = Image.open(seg).convert('L')

    return t1, t2, t1ce, flair, seg


def inference(img, mask, type):
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

    joint_trans = get_joint_transformations_val(args)
    standard_transformations = get_standard_transformations()
    img, mask = joint_trans(img, mask)
    img = standard_transformations(img)
    img = img.cuda().unsqueeze(0)
    pred, f = model(img)
    _, pred = torch.max(pred, 1)
    pred = pred.detach().cpu().numpy()[0]
    pred[pred == 1] = 255
    pred = np.array(pred).astype(np.uint8)
    show_single(pred, type + "_" + "pred")


def main():
    t1, t2, t1ce, flair, seg = get_data()

    inference(t1, seg, "t1")
    inference(t2, seg, "t2")
    inference(t1ce, seg, "t1ce")
    inference(flair, seg, "flair")

    seg = np.array(seg)
    seg[seg > 0] = 255
    show_single(seg, "truth")

    draw_counter(np.array(t1), seg, "t1")
    draw_counter(np.array(t2), seg, "t2")
    draw_counter(np.array(t1ce), seg, "t1ce")
    draw_counter(np.array(flair), seg, "flair")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    select_img = 2
    select_slice = 5
    main()