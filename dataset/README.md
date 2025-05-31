# dataset/

This folder contains the dataset loader for training Mask R-CNN on the Bosch Small Traffic Light Dataset. It is designed to parse COCO-style annotations and generate input data for PyTorch's detection models.

---

## coco_maskrcnn_dataset.py
**'Purpose'**:
Defines the COCOMaskRCNNDataset class, a custom PyTorch Dataset that:
- Loads images from a directory and annotations from a COCO-style JSON file.
- Extracts bounding boxes, labels, and binary masks for instance segmentation.
- Returns each image and its corresponding annotation dictionary (target) in the format expected by Mask R-CNN.

**'Key Features'**:
- Image Loading: Opens RGB images using PIL from the provided directory.
- Annotation Parsing: Uses pycocotools to extract:
  -- Bounding boxes (bbox)
  -- Class labels (category_id)
  -- Segmentation masks
- Empty Annotation Handling: If an image has no annotations, returns empty tensors for boxes, labels, and masks.
- Transform Support: Accepts optional torchvision transforms.

**'Class Label Mapping'**:
- 1 → Red Traffic Light
- 2 → Green Traffic Light



