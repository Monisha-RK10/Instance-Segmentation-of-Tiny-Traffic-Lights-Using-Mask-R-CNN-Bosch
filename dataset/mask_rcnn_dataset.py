import os
import torch
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json
import os
import random
import numpy as np
import torch

# Step 1: Create a custom Mask R-CNN class
# This piece of code does the following:
# COCOMaskRCNNDataset class takes image dir, loads annotation JSON via COCO, gets a list of image IDs.
# Loop through each image ID, get file name, open it in RGB format, & apply transform.
# Get all annotations for that image, and load them to extract each annotation's box (xmin/max, ymin/max), label (class starts from 1 (red), 2(green)), mask.
# If annotation empty, append '[]' & convert target to tensor.
# Return image & target.

class COCOMaskRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotation_json, transforms=None):
        self.images_dir = images_dir
        self.coco = COCO(annotation_json)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []

        for ann in annotations:
            xmin, ymin, width, height = ann['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann['category_id']) + 1) # Shift labels from 0/1 to 1/2 (be careful, check if json has red=1, green=2. If classes startes from 1, then dont add 1)
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks, # segmentation masks are binary (0 or 1).
            "image_id": torch.tensor([image_id]),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_ids)
