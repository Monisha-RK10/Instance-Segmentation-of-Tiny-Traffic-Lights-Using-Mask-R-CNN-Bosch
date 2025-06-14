# training_and_inference_scripts/

This folder contains the main scripts for training and evaluating the Mask R-CNN model on the Bosch Small Traffic Light Dataset.

---

## Files

### `train_mask_rcnn.py`
- **Purpose**: Trains a Mask R-CNN model for instance segmentation of red and green traffic lights.
- **Key Steps**:
  - Loads a pretrained Mask R-CNN model from torchvision.
  - Replaces the classification and mask heads with task-specific heads for 3 classes (background + 2 traffic light classes).
  - Uses the custom `COCOMaskRCNNDataset` from the `dataset/` folder.
  - Configures optimizer, learning rate scheduler, and early stopping.
  - Tracks training and validation losses across epochs.
  - Saves the best-performing model based on validation loss.
  - Generates and saves loss plots for analysis and reporting.

---

### `evaluate_mask_rcnn.py`
- **Purpose**: Evaluates the trained model on the validation dataset using COCO metrics.
- **Key Steps**:
  - Normalizes label mapping: Ensures `annotation_train.json` and `annotation_val.json` use the correct class IDs (1 → Red, 2 → Green) to match model expectations.
  - Loads the trained model weights and sets the model to evaluation mode.
  - Converts the output of MASK R-CNN from soft mask to binary mask by setting different thresholds (0.25, 0.50, 0.75).
  - Binary masks from earlier are then converted to RLE and GT masks are sent for evaluation.
  - Uses pycocotools to compute standard COCO evaluation metrics (AP, AR, etc.) on the validation dataset.

---

These scripts together form the training and evaluation pipeline for reproducible and scalable instance segmentation experiments on small traffic lights.

