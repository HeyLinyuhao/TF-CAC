import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
import matplotlib.pyplot as plt
import math
from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from collections import defaultdict
from skimage.transform import resize
from segment_anything.modeling import Sam
from segment_anything.predictor import SamPredictor
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
    points2box
)
import copy

def compare(pixels, color_list,val):
    for i in color_list:
        if (np.square(np.array(pixels) - np.array(i))).sum()<val:
            return True

def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset

def pre_process_ref_box(ref_box, crop_box, layer_idx):
    if layer_idx == 0:
        return ref_box
    else:
        new_bbox = []
        x0, y0, x1, y1 = crop_box
        for ref in ref_box:
            x0_r, y0_r, x1_r, y1_r = ref
            area = (y1_r - y0_r) * (x1_r - x0_r)
            x_0_new = max(x0, x0_r)
            y_0_new = max(y0, y0_r)
            x_1_new = min(x1, x1_r)
            y_1_new = min(y1, y1_r)
            crop_area = (y_1_new - y_0_new) * (x_1_new - x_0_new)
            if crop_area / area > 0.7:
                new_bbox.append([x_0_new, y_0_new, x_1_new, y_1_new])

        return new_bbox

def show_anns(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for pi in anns['points']:
        x0, y0 = pi
        # ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.scatter(x0,y0, color='green', marker='*', s=10, edgecolor='white', linewidth=1.25)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_mask2(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 100,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.3,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.3,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.3,
        crop_overlap_ratio: float = 10 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crops_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crops_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")


        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

        self.prototype = defaultdict(list)


    @torch.no_grad()
    def generate(self, image: np.ndarray, ref_bbox,SLIC_centers,super_pixel) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.


        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data, mask_data2 = self._generate_masks(image, ref_bbox,SLIC_centers,super_pixel) 
       
        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
            mask_data2["segmentations"] = [coco_encode_rle(rle) for rle in mask_data2["rles"]]
            

        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
            mask_data2["segmentations"] = [rle_to_mask(rle) for rle in mask_data2["rles"]]


        else:
            mask_data["segmentations"] = mask_data["rles"]
            mask_data2["segmentations"] = mask_data2["rles"]


        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "label":1
            }
            curr_anns.append(ann)


        curr_anns2 = []
        for idx in range(len(mask_data2["segmentations"])):
            ann = {
                "segmentation": mask_data2["segmentations"][idx],
                "area": area_from_rle(mask_data2["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data2["boxes"][idx]).tolist(),
                "predicted_iou": mask_data2["iou_preds"][idx].item(),
                "point_coords": [mask_data2["points"][idx].tolist()],
                "stability_score": mask_data2["stability_score"][idx].item(),
                "label":0

            }
            curr_anns2.append(ann)

        
        return curr_anns,curr_anns2
    

            

    def _generate_masks(self, image: np.ndarray, ref_box,SLIC_centers,super_pixel) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        proto_data = MaskData()
        all_data = MaskData()
        self.predictor.set_image(image)  

        ref_box = torch.tensor(ref_box).to(self.predictor.device)

        transformed_boxes = self.predictor.transform.apply_boxes_torch(ref_box, orig_size)

        masks, iou_preds, low_res_masks = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )

        x = ref_box[:, 0] + (ref_box[:, 2] - ref_box[:, 0]) / 2
        y = ref_box[:, 1] + (ref_box[:, 3] - ref_box[:, 1]) / 2
        points = torch.stack([x, y], dim=1)
        proto_data = MaskData(
                masks=masks.flatten(0, 1),
                iou_preds= torch.ones_like(iou_preds.flatten(0, 1)),
                points=points.cpu(),
                stability_score = torch.ones_like(iou_preds.flatten(0, 1)),
            )
            
        proto_data["boxes"] = batched_mask_to_box(proto_data["masks"])
        proto_data["rles"] = mask_to_rle_pytorch(proto_data["masks"])
        del proto_data["masks"]

        self.predictor.reset_image()


        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            # print(layer_idx)
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size, ref_box,SLIC_centers,super_pixel)
            all_data.cat(crop_data)



        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(all_data ["crop_boxes"])
            scores = scores.to(all_data ["boxes"].device)
            keep_by_nms = batched_nms(
                all_data ["boxes"].float(),
                scores,
                torch.zeros(len(all_data ["boxes"]), device="cuda"),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            all_data .filter(keep_by_nms)

        proto_data.to_numpy()
        all_data .to_numpy()


        return proto_data,all_data
    
    

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
        ref_box,
        SLIC_centers,
        super_pixel
    ) -> MaskData:

        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]


        self.predictor.set_image(cropped_im)
        data = MaskData()
        
        if super_pixel:
            if crop_layer_idx == 0:
                points_for_image = []
                for i in SLIC_centers:
                    points_for_image.append([int(i['yx'][1]),int(i['yx'][0])])
            
            
                # points_for_image = np.array(SLIC_centers)
                points_for_image = np.array(points_for_image)  
                #   
            else:
                points_scale = np.array(cropped_im_size)[None, ::-1]
                points_for_image = self.point_grids[0] * points_scale
                points_for_image = np.array(points_for_image)  
        else:

            points_scale = np.array(cropped_im_size)[None, ::-1]
            points_for_image = self.point_grids[0] * points_scale
            points_for_image = np.array(points_for_image)  
            # Generate masks for this crop in batches
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(image, points, cropped_im_size, 
                                             crop_box, orig_size)
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros(len(data["boxes"]), device="cuda"),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])



        return data

    def _process_batch(
        self,
        image: np.ndarray,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...]
    ) -> MaskData:
        orig_h, orig_w = orig_size
        
        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size).astype(int)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)


        masks, iou_preds, low_res_masks = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            # boxes=boxes,
            multimask_output=False,
            return_logits=False,
        )

        # masks = masks[:,1,:,:]
        # masks = masks.unsqueeze(1)
        # iou_preds = iou_preds[:,1]
        # iou_preds = iou_preds.unsqueeze(1)
        # print(iou_preds.shape)

        # print(masks.shape)

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks


        # # Filter by predicted IoU
        # if self.pred_iou_thresh > 0.0:
        #     keep_mask = data["iou_preds"] > self.pred_iou_thresh
            # data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        # if self.stability_score_thresh > 0.0:
        #     keep_mask = data["stability_score"] >= self.stability_score_thresh
        #     data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        # data["masks"] = data["masks"]
        data["boxes"] = batched_mask_to_box(data["masks"])


        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros(len(boxes)),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data
