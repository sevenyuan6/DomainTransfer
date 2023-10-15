import torch
import argparse
import torch.backends.cudnn as cudnn
from configs import get_args_parser
from PIL import Image
from model.get_model import get_model
from data.retina.retina_datas import *
from utils.vis_tools import show_single
from data.data_utils import get_standard_transformations, get_joint_transformations_val
from data.data_utils import img_mask_crop_binary
import numpy as np
import cv2


def draw_counter(mask_cup, mask_disc, pred_cup, pred_disc, img_draw):
    img_draw = np.array(img_draw)[150:650, 150:650]
    mask_cup = np.array(mask_cup)[150:650, 150:650]
    mask_disc = np.array(mask_disc)[150:650, 150:650]
    pred_cup = pred_cup[150:650, 150:650]
    pred_disc = pred_disc[150:650, 150:650]

    # img_draw = np.array(img_draw)[130:630, 50:550]
    # mask_cup = np.array(mask_cup)[130:630, 50:550]
    # mask_disc = np.array(mask_disc)[130:630, 50:550]
    # pred_cup = pred_cup[130:630, 50:550]
    # pred_disc = pred_disc[130:630, 50:550]

    # img_draw = np.array(img_draw)[170:590, 190:610]
    # mask_cup = np.array(mask_cup)[170:590, 190:610]
    # mask_disc = np.array(mask_disc)[170:590, 190:610]
    # pred_cup = pred_cup[170:590, 190:610]
    # pred_disc = pred_disc[170:590, 190:610]

    img_draw = np.array(img_draw)
    show_single(img_draw, "crop_img")

    contours_cup, _ = cv2.findContours(np.array(mask_cup), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_disk, _ = cv2.findContours(np.array(mask_disc), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pcup, _ = cv2.findContours(pred_cup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pdisc, _ = cv2.findContours(pred_disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_draw, contours_cup, -1, (255, 0, 0), 2)
    cv2.drawContours(img_draw, contours_disk, -1, (0, 255, 0), 2)
    cv2.drawContours(img_draw, contours_pcup, -1, (0, 0, 255), 2)
    cv2.drawContours(img_draw, contours_pdisc, -1, (160, 32, 240), 2)
    show_single(img_draw, "merge")


def get_data():
    all_domain = [Drishti_GS(args).get_data()["Training"], RIM_ONE_r3(args).get_data()["Training"], REFUGE(args, "train").get_data()["Training"], REFUGE(args, "test").get_data()["Training"]]
    selected = all_domain[select]
    img_dir, mask_cup, mask_disc, domain_name = selected[select_img]
    img = Image.open(img_dir).convert('RGB')
    show_single(np.array(img), "origin")
    mask_cup = Image.open(mask_cup)
    mask_disc = Image.open(mask_disc)

    img, mask_cup, mask_disc = img_mask_crop_binary(img, mask_cup, mask_disc, domain_name)
    return img, mask_cup, mask_disc


def inference(img, mask, w, h, type_="cup"):
    device = torch.device(args.device)
    cudnn.benchmark = True
    model = get_model(args)
    model.to(device)
    args.prediction_mode = type_

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
    pred = cv2.resize(pred, (w, h))
    return pred


def main():
    img, mask_cup, mask_disc = get_data()
    w, h = img.size

    pred_cup = inference(img, mask_cup, w, h, "cup")
    pred_disc = inference(img, mask_disc, w, h, "disc")

    draw_counter(mask_cup, mask_disc, pred_cup, pred_disc, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    select = 0
    select_img = 0
    main()