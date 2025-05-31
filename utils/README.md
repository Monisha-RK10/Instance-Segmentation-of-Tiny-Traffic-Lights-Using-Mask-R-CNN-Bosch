# utils/

This folder contains utility scripts to support dataset preparation and reproducibility for Mask R-CNN training on the Bosch Small Traffic Light Dataset.

---

## seed_utils.py
**`Purpose`**:

Ensures reproducibility across runs by setting seeds for common libraries.

**`Features`**

Sets seeds for:
- random (Python)
- numpy
- torch
- torch.backends.cudnn
- Enables deterministic operations in PyTorch (by setting cudnn.deterministic = True and cudnn.benchmark = False)

## split_coco_annotations.py
**`Purpose`**

Splits a single COCO-style annotation JSON (containing all images) into two files: one for training and one for validation.

**`Functionality`**

Input:
- Original COCO JSON (with all image and annotation entries)
- Folder names or lists specifying which images go to train and val

Process:
- Iterates over all images
  - If an image belongs to the training or validation set:
  - Assigns a new image ID
  - Maps old ID → new ID (image_id_map)
- Updates each annotation:
  - Keeps only relevant annotations
  - Assigns new image_id and unique id for annotation

Output:
- annotation_train.json
- annotation_val.json


