# input/

This folder contains the COCO-format annotation files used for training and validating the Mask R-CNN model on the Bosch Small Traffic Light Dataset.

---

## Files

- **`annotation_train.json`**  
  Contains the training set annotations in COCO format. These were created by splitting the full dataset using `split_coco_annotations.py` (located in `utils/`).

- **`annotation_val.json`**  
  Contains the validation set annotations in COCO format.

---

Each file includes:
- `images`: List of image metadata (filename, dimensions, etc.)
- `annotations`: Bounding boxes, segmentation masks, and category IDs for each instance.
- `categories`: Class mappings (1 → Red, 2 → Green).

