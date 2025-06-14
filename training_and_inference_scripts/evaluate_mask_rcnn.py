# Step 4: Evaluation
# This piece of code does the following:
# a) In this setup:
# Train & val json have 0 for red, 1 for green, i.e., "categories": [{"id": 0, "name": "Red"}, {"id": 1, "name": "Green"}]. 
# Model was trained with 3 classes, background + 1 for red, 2 for green.
# Function 'update_category_ids' ensures that during evaluation, train & val json assign 1 for red, 2 for green to match with labels in COCOMaskRCNNDataset.
# b) Loads the pre-trained weights with model architecture.
# c) Evaluates on 'val_dataset' via pycocotools  

# During training: GT masks are binary (from annToMask) 
# During inference: Predicted masks are soft (masks = output["masks"].cpu().numpy())
# Threshold the soft masks to convert them into binary during evaluation for comparison (e.g., > 0.5) (mask = masks[i, 0] > 0.5)

# Problem: engine.py does not support segmentation masks, it only evaluates bounding boxes. 
# Mask R-CNN gives soft masks and COCO evaluation need binary masks (thresholding the soft mask and converting to RLE)
# Solution: use pycocotools

import json
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from pycocotools import mask as mask_utils
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np

# COCOeval
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Import COCOMaskRCNNDataset from dataset/coco_maskrcnn_dataset for evaluation on val_dataset
from dataset.coco_maskrcnn_dataset import COCOMaskRCNNDataset

# Paths
val_json_path = "/content/drive/MyDrive/MASK-RCNN_TrafficLight/annotation_val.json"

val_dataset = COCOMaskRCNNDataset(
    images_dir='/content/drive/MyDrive/MASK-RCNN_TrafficLight/images/val',
    annotation_json='/content/drive/MyDrive/MASK-RCNN_TrafficLight/annotation_val.json',
    transforms=T.ToTensor()
)

coco_predictions = []

# Function to update category IDs to ensure class mapping
def update_category_ids(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Update categories
    data["categories"] = [
        {"id": 1, "name": "Red"},
        {"id": 2, "name": "Green"}
    ]

    # Map old ID (0 -> 1, 1 -> 2)
    id_map = {0: 1, 1: 2}
    for ann in data["annotations"]:
        ann["category_id"] = id_map[ann["category_id"]]

    # Save updated JSON
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

# Apply to val
update_category_ids(val_json_path)

" Updated annotation_train.json and annotation_val.json with Red=1, Green=2."

# Load the pre-trained weights with model architecture.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model with same num_classes
model = maskrcnn_resnet50_fpn(num_classes=3)                                                             # 1 bg + 2 classes
model.load_state_dict(torch.load("best_maskrcnn.pth")["model_state_dict"])
model.to(device)
model.eval()

# Evaluate on val_dataset via pycocotools
for idx in tqdm(range(len(val_dataset))):
    img, target = val_dataset[idx]
    image_id = int(target["image_id"].item())                                                            # tensor([image_id]) is converted to plain int for pycocotools

    with torch.no_grad():
        output = model([img.to(device)])[0]

    masks = output["masks"].cpu().numpy()                                                                #  GPU to NumPy for COCO evaluation,  numpy() works only on CPU tensors.
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    for i in range(len(masks)):
        if scores[i] < 0.5:
            continue

        mask = masks[i, 0] > 0.5             
        # Shape (H, W) of ith instance: [num_instances, 1, H, W] where 1 is channel. Channel is not used as Mask R-CNN always predicts 1-channel mask. 
        # Threshold the soft mask to binary using 0.25 (floating-point probability mask to binary mask)
        # Changing 0.5 affects which pixels count as foreground -> affects predicted mask shape -> affects IoU -> affects COCO metrics.

        # Converting the binary mask (after thresholding) into RLE format for COCO evaluator
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) 
        rle["counts"] = rle["counts"].decode("utf-8")

        coco_predictions.append({
            "image_id": image_id,
            "category_id": int(labels[i]),
            "segmentation": rle,
            "score": float(scores[i]),
        })

with open("maskrcnn_preds.json", "w") as f:
    json.dump(coco_predictions, f)

coco_gt = COCO("/content/drive/MyDrive/MASK-RCNN_TrafficLight/annotation_val.json")                      # For ground truth
coco_dt = coco_gt.loadRes("maskrcnn_preds.json") # Predicted by the trained model                        # COCO object (coco_gt) holds image and category metadata.
coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
