# dataset/

This folder contains the dataset loader for training Mask R-CNN on the Bosch Small Traffic Light Dataset. It is designed to parse COCO-style annotations and generate input data for PyTorch's detection models.

---

## coco_maskrcnn_dataset.py
- **'Purpose'**: Defines the COCOMaskRCNNDataset class, a custom PyTorch Dataset that:
Loads images and COCO-format annotations.
Extracts bounding boxes, labels, and binary masks for instance segmentation.
Returns each image and its corresponding annotations in the format expected by Mask R-CNN.

---

Key Features:
Image Loading: Reads RGB images from the specified directory.
Annotation Parsing: Uses the pycocotools API to extract bounding boxes (bbox), class labels (category_id), and segmentation masks.
Empty Annotations Handling: If an image has no annotations, returns empty tensors for boxes, labels, and masks.
Transform Support: Supports image augmentations via optional transforms.

Class Label Mapping:
1 → Red Traffic Light
2 → Green Traffic Light



