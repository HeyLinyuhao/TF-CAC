import os
import torch
import torchvision
import argparse
import json
import numpy as np
import os
import copy
from tqdm import tqdm
from os.path import exists,join
import pickle
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from shi_segment_anything import sam_model_registry, SamPredictor
from shi_segment_anything.automatic_mask_generator_carpk2 import SamAutomaticMaskGenerator
# from model import  Resnet50FPN
from utils import *
from fast_slic import Slic
from preprocess import _transform, _transform2

preprocess_my = _transform(518)
preprocess_my2 = _transform2(518)



parser = argparse.ArgumentParser(description="Counting with SAM")
parser.add_argument("-dp", "--data_path", type=str, default='/home/yuhao/Desktop/cac_data/datasets/CARPK_devkit/data/', help="Path to the FSC147 dataset")
parser.add_argument("-o", "--output_dir", type=str,default="./logsSave/CARPK", help="/Path/to/output/logs/")
parser.add_argument("-ts", "--test-split", type=str, default='test', choices=["train", "test"], help="what data split to evaluate on on")
parser.add_argument("-pt", "--prompt-type", type=str, default='box', choices=["box", "point"], help="what type of information to prompt")
parser.add_argument("-d", "--device", type=str,default='cuda:0', help="device")
# ablation study
parser.add_argument("-sp", "--super_pixel", type=bool, default=True, help="wether to use superpixel")
parser.add_argument("-tpu", "--trans_update", type=bool, default=True, help="wether to use tpu")
parser.add_argument("-ms", "--multiple_scale", type=bool, default=True, help="wether to use multiple scale")
args = parser.parse_args()


def get_patch_fea2(image,all):
    image = preprocess_my(Image.open(image).convert('RGB')).unsqueeze(0).to(device=args.device)
    image_features = model_dino.forward_features(image)["x_norm_patchtokens"]
    image_features = image_features.transpose(1,2)

    masks = []
    all_index=[]
    ind = 0
    for m in range(len(all)):
        mask = all[m]["segmentation"]
        masks.append(preprocess_my2((mask*1.0).astype('uint8')).unsqueeze(0))

        all_index.append(ind)
        ind = ind + 1

    low_res_masks = torch.cat(masks).to(device=args.device)
    low_res_masks = F.interpolate(low_res_masks, [37,37], mode='bilinear', align_corners=False)
    low_res_masks = low_res_masks.flatten(2, 3)

    topk_idx = torch.topk(low_res_masks, 1)[1]
    low_res_masks.scatter_(2, topk_idx, 1.0)
    all_fea = (image_features * low_res_masks).sum(dim=2) / low_res_masks.sum(dim=2)
    all_fea = F.normalize(all_fea, dim=1)


    return all_fea, all_index
     
if __name__=="__main__": 

    data_path = args.data_path
    anno_file = data_path + 'Annotations'
    data_split_file = data_path + 'ImageSets'
    im_dir = data_path + 'Images'

    if not exists(args.output_dir):
        os.mkdir(args.output_dir)
        os.mkdir(args.output_dir+'/logs')
    
    if not exists(args.output_dir+'/%s'%args.test_split):
        os.mkdir(args.output_dir+'/%s'%args.test_split)

    if not exists(args.output_dir+'/%s/%s'%(args.test_split,args.prompt_type)):
        os.mkdir(args.output_dir+'/%s/%s'%(args.test_split,args.prompt_type))
    
    log_file = open(args.output_dir+'/logs/log-%s-%s.txt'%(args.test_split,args.prompt_type), "w") 

    sam_checkpoint = "./pretrain/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=args.device)
    
    mask_generator = SamAutomaticMaskGenerator(model=sam)
    
    ref_info = generate_ref_info(args)

    model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model_dino.eval()

    model_dino.to(device=args.device)

    MAE = 0
    RMSE = 0
    NAE = 0
    SRE = 0

    # ===================================================================================================
    exam_dir = 'exemplar.txt'
    proto = []
    exem_file = data_path + 'exemplar.txt'

    with open(exem_file) as f:
        exa_ids = f.readlines()
    for i,exa_id in enumerate(exa_ids):
        strings = exa_id.strip().split(':')
        im_id = strings[0]
        ref_bbox = strings[1][1:-1].split(', ')
        ref_bbox = [int(box) for box in ref_bbox]
        x0, y0, x1,y1 = ref_bbox
        
        image  = '{}/{}'.format(im_dir, im_id)
        image = Image.open(image).convert('RGB')
        width, height = image.size

        mask = np.zeros(( height, width))
        mask[x0:x1,y0:y1] = 1

        image = preprocess_my(image).unsqueeze(0).to(device=args.device)
        
        image_features = model_dino.forward_features(image)["x_norm_patchtokens"]
        image_features = image_features.transpose(1,2)
        mask = preprocess_my2((mask*1.0).astype('uint8')).unsqueeze(0).to(device=args.device)

        low_res_masks = F.interpolate(mask, [37,37], mode='bilinear', align_corners=False)
        low_res_masks = low_res_masks.flatten(2, 3)

        topk_idx = torch.topk(low_res_masks, 1)[1]
        low_res_masks.scatter_(2, topk_idx, 1.0)
        all_fea = (image_features * low_res_masks).sum(dim=2) / low_res_masks.sum(dim=2)
        all_fea = F.normalize(all_fea, dim=1)
        proto.append(all_fea)

    proto = torch.cat(proto).to(device=args.device)
# ===================================================================================================

    with open(data_split_file+'/%s.txt'%args.test_split) as f:
        im_ids = f.readlines()
    
    for i,im_id in tqdm(enumerate(im_ids)):
        im_id = im_id.strip()
        image = cv2.imread('{}/{}.png'.format(im_dir, im_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

       # slic = Slic(num_components= 1600, compactness=0.01)
        # slic = Slic(num_components= 4096, compactness=0.01)
        # slic = Slic(num_components= 2500, compactness=0.01)
#        slic = Slic(num_components= 3025, compactness=0.1)




        # slic = Slic(num_components= 4096, compactness=1000)
        slic = Slic(num_components= 1600, compactness=1000)

        assignment = slic.iterate(image) 

        masks = mask_generator.generate(image, ref_info, slic.slic_model.clusters, args.super_pixel)



        all_fea, all_ind = get_patch_fea2('{}/{}.png'.format(im_dir, im_id),masks)
        # no mean, 0.75,0.6

        proto = torch.mean(proto,0,keepdim=True)

        cos_dis = all_fea @ proto.t()
        cos_dis = cos_dis.squeeze().cpu()


        pred = cos_dis > 0.4
        if args.trans_update:

            th = 0.5
            if 1.0*pred.sum() == 0:
            # print(111)
                proto2 = proto
                th = 0.4
            else:
                proto2 = all_fea[pred]

            proto2 = torch.mean(proto2,0,keepdim=True)  
            cos_dis = all_fea @ proto2.t()
            cos_dis = cos_dis.squeeze().cpu()
            pred = cos_dis>th

        new_masks = []
        for m in range(len(all_ind)):
            ma = masks[all_ind[m]]
            if pred[m]:    
            # if 0.2*box_area[0]< ma['area'] < 4*box_area[-1]:
                new_masks.append(ma)


        with open(anno_file+'/%s.txt'%im_id) as f:
                    box_lines = f.readlines()

        gt_cnt = len(box_lines)
        pred_cnt = len(new_masks)

        print(pred_cnt, gt_cnt, abs(pred_cnt-gt_cnt))
        log_file.write("%d: %d,%d,%d\n"%(i, pred_cnt, gt_cnt,abs(pred_cnt-gt_cnt)))
        log_file.flush()

        err = abs(gt_cnt - pred_cnt)
        MAE = MAE + err
        RMSE = RMSE + err**2
        NAE = NAE+err/gt_cnt
        SRE = SRE+err**2/gt_cnt

        # """
        # fig = plt.figure()
        # plt.axis('off')
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        # ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        # plt.imshow(image)
        # show_anns(new_masks, plt.gca())
        # plt.savefig('%s/%s/%03d_mask.png'%(args.output_dir,args.test_split,i), bbox_inches='tight', pad_inches=0)
        # plt.close()#"""

    MAE = MAE/len(im_ids)
    RMSE = math.sqrt(RMSE/len(im_ids))
    NAE = NAE/len(im_ids)
    SRE = math.sqrt(SRE/len(im_ids))

    print("MAE:%0.2f,RMSE:%0.2f,NAE:%0.2f,SRE:%0.2f"%(MAE,RMSE,NAE,SRE))
    log_file.write("MAE:%0.2f,RMSE:%0.2f,NAE:%0.2f,SRE:%0.2f"%(MAE,RMSE,NAE,SRE))
    log_file.close()

        
