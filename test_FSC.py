import cv2
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists
import os
import torch
from fast_slic import Slic
from segment_anything import sam_model_registry
from automatic_mask_generator import SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
# import clip
from PIL import Image
from preprocess import _transform, _transform2
import torch.nn.functional as F
from torchvision import transforms as pth_transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
preprocess_my = _transform(518)
preprocess_my2 = _transform2(518)


# ////////////////////////////////DINO////////////////////////////////////
# model_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model_dino.eval()

model_dino.to(device)

parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp", "--data_path", type=str, default='cac_data/', help="Path to the FSC147 dataset")
parser.add_argument("-ts", "--test_split", type=str, default='test', choices=["val_PartA","val_PartB","test_PartA","test_PartB","test", "val"], help="what data split to evaluate on")
parser.add_argument("-mt", "--model_type", type=str, default="vit_h", help="model type")

# ablation study
parser.add_argument("-sp", "--super_pixel", type=bool, default=True, help="wether to use superpixel")
parser.add_argument("-tpu", "--trans_update", type=bool, default=True, help="wether to use tpu")
parser.add_argument("-ms", "--multiple_scale", type=bool, default=True, help="wether to use multiple scale")


# parser.add_argument("-mt", "--model_type", type=str, default="vit_b", help="model type")

parser.add_argument("-mp",  "--model_path", type=str, default='sam_vit_h_4b8939.pth', help="path to trained model")

# parser.add_argument("-mp",  "--model_path", type=str, default='sam_vit_b_01ec64.pth', help="path to trained model")

parser.add_argument("-v",  "--viz", type=bool, default=True, help="wether to visualize")
parser.add_argument("-d",   "--device", default='0', help='assign device')
parser.add_argument("-th",   "--threshold", type=float,  default=4.0, help='threholds')
args = parser.parse_args()

data_path = args.data_path
anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'Train_Test_Val_FSC_147.json'

im_dir = data_path + 'images_384_VarV2'


if not exists(anno_file) or not exists(im_dir):
    print("Make sure you set up the --data-path correctly.")
    print("Current setting is {}, but the image dir and annotation file do not exist.".format(args.data_path))
    print("Aborting the evaluation")
    exit(-1)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        x0, y0, w, h = ann['bbox']
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=1))
        ax.scatter([x0+w//2], [y0+h//2], color='green', marker='.', s=1, edgecolor='white', linewidth=1)


def get_patch_fea2(image,good,all):
    image = preprocess_my(Image.open(image).convert('RGB')).unsqueeze(0).to(device)
    image_features = model_dino.forward_features(image)["x_norm_patchtokens"]
    image_features = image_features.transpose(1,2)

    train_masks = []

    for m in range(len(good)):
        mask = good[m]["segmentation"]
        train_masks.append(preprocess_my2((mask*1.0).astype('uint8')).unsqueeze(0))



    low_res_masks = torch.cat(train_masks).to(device)
    low_res_masks = F.interpolate(low_res_masks, [37,37], mode='bilinear', align_corners=False)
    low_res_masks = low_res_masks.flatten(2, 3)


    topk_idx = torch.topk(low_res_masks, 1)[1]
    low_res_masks.scatter_(2, topk_idx, 1.0)
    train_fea = (image_features * low_res_masks).sum(dim=2) / low_res_masks.sum(dim=2)
    train_fea = F.normalize(train_fea, dim=1)

    masks = []
    all_index=[]
    ind = 0
    for m in range(len(all)):
        mask = all[m]["segmentation"]
        masks.append(preprocess_my2((mask*1.0).astype('uint8')).unsqueeze(0))

        all_index.append(ind)
        ind = ind + 1

    low_res_masks = torch.cat(masks).to(device)
    low_res_masks = F.interpolate(low_res_masks, [37,37], mode='bilinear', align_corners=False)
    low_res_masks = low_res_masks.flatten(2, 3)


    topk_idx = torch.topk(low_res_masks, 1)[1]
    low_res_masks.scatter_(2, topk_idx, 1.0)
    all_fea = (image_features * low_res_masks).sum(dim=2) / low_res_masks.sum(dim=2)
    all_fea = F.normalize(all_fea, dim=1)


    return train_fea, all_fea, all_index

debug = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
device = 'cuda'
sam = sam_model_registry[args.model_type](checkpoint=args.model_path)
sam.to(device=device)


mask_generator = SamAutomaticMaskGenerator(
                                model=sam,
                                min_mask_region_area=1
                                )

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)


cnt = 0
SAE = 0  # sum of absolute errors
SSE = 0  # sum of square errors
NAE = 0
SRE = 0
print("Evaluation on {} data".format(args.test_split))
im_ids = data_split[args.test_split]
print(args.super_pixel)
pbar = tqdm(im_ids)
errs = []
for im_id in pbar:
    anno = annotations[im_id]
    bboxes = anno['box_examples_coordinates']
    dots = np.array(anno['points'])

    image = cv2.imread('{}/{}'.format(im_dir, im_id))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_boxes = list()
    box_area = []

    for bbox in bboxes:
        x1, y1 = bbox[0][0], bbox[0][1]
        x2, y2 = bbox[2][0], bbox[2][1]
        input_boxes.append([x1, y1, x2, y2])
        box_area.append((x2-x1)*(y2-y1))

    box_area.sort()


    mask_generator.crop_n_layers = 0
    if args.multiple_scale:
        if (box_area[0]+box_area[-1])/2<30:
            mask_generator.crop_n_layers = 3
        
        elif (box_area[0]+box_area[-1])/2<150:
            mask_generator.crop_n_layers = 1
        
        else:
            mask_generator.crop_n_layers = 0


    
    th1 = 0.5
    th2 = 0.4
    # 0.5 0.5 13.29, RMSE: 58.34
    # 0.5 0.3 12.82, RMSE: 55.88



    slic = Slic(num_components= min(64000,int(image.shape[0]*image.shape[1]*9/box_area[0])), compactness=1000)

    assignment = slic.iterate(image) 


    pos, all = mask_generator.generate(image, input_boxes,slic.slic_model.clusters, args.super_pixel)


    good_fea, all_fea, all_ind = get_patch_fea2('{}/{}'.format(im_dir, im_id),pos,all)

    good_fea = torch.mean(good_fea,0,keepdim=True)

    cos_dis = all_fea @ good_fea.t()

    pred = (cos_dis > th1).squeeze()
    if args.trans_update:
        good_fea2 = all_fea[pred]
        if 1.0*pred.sum() == 0:
            good_fea2 = good_fea
            th2 = 0.5
        else:
            good_fea2 = all_fea[pred]

        good_fea2 = torch.mean(good_fea2,0,keepdim=True)
        cos_dis = all_fea @ good_fea2.t()
        pred = (cos_dis>th2).squeeze()
    
    img_area = image.shape[0]*image.shape[1]
    new_masks = []
    for m in range(len(all_ind)):
        ma = all[all_ind[m]]
        if pred[m]:
            if 0.2*box_area[0]< ma['area'] < 4*box_area[-1]:
                new_masks.append(ma)


    pred_cnt = len(new_masks)


    if args.viz:
        if not exists('viz'):
            os.mkdir('viz')
        plt.figure()
        plt.imshow(image)
        show_anns(all)

        plt.axis('off')
        plt.savefig('viz/{}'.format(im_id),bbox_inches='tight', pad_inches=0)
        plt.close()

    gt_cnt = dots.shape[0]
    
    cnt = cnt + 1
    err = abs(gt_cnt - pred_cnt)
    SAE += err
    SSE += err**2
    NAE = NAE+err/gt_cnt
    SRE = SRE+err**2/gt_cnt
    errs.append(err)


  #  pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f},NAE: {:5.2f},SRE: {:5.2f}'.\
  #                       format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5,NAE/len(im_ids), (SRE/len(im_ids))**0.5))
    
    print(im_id, gt_cnt, pred_cnt,abs(pred_cnt - gt_cnt))



print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f},NAE: {:6.2f},SRE: {:6.2f}'.format(args.test_split, SAE/cnt, (SSE/cnt)**0.5, NAE/len(im_ids), (SRE/len(im_ids))**0.5))

# median
# errs.sort()
# l =len(errs)
# mid = int( l/2 )
# print(errs[mid])
