# Step 3: Train the model
# This piece of code does the following:
# Load pretrained weights
# Replace heads for the custom task (classifier + mask) with number of classes
# Custom COCO dataset loader, optimizer, scheduler
# Train-val loss tracking
# Early stopping
# Save best model/early stopping model
# Loss plots for reporting

import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Paths
save_path = "best_maskrcnn.pth"

# Import COCOMaskRCNNDataset from dataset/coco_maskrcnn_dataset 
from dataset.coco_maskrcnn_dataset import COCOMaskRCNNDataset

def get_mask_rcnn_model(num_classes):
    # Load a pre-trained model for COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the classifier head
    in_features_cls = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features_cls, num_classes                                                                # Box classification is scalar -> linear layer
    )

    # Replace the mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes                                                 # Mask prediction is spatial -> CNN + upsampling → need hidden layers.
    )

    return model

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_mask_rcnn_model(num_classes=3)  # Background, Red, Green
model.to(device)

# Optimizer & LR scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)        # Reduces LR only when val loss stagnates, mode='max' for metrics like accuracy, where higher is better.

# Create dataset & dataloaders
train_dataset = COCOMaskRCNNDataset(
    images_dir='/content/drive/MyDrive/MASK-RCNN_TrafficLight/images/train',
    annotation_json='/content/drive/MyDrive/MASK-RCNN_TrafficLight/annotation_train.json',
    transforms=T.ToTensor()  # Or custom Compose
)
val_dataset = COCOMaskRCNNDataset(
    images_dir='/content/drive/MyDrive/MASK-RCNN_TrafficLight/images/val',
    annotation_json='/content/drive/MyDrive/MASK-RCNN_TrafficLight/annotation_val.json',
    transforms=T.ToTensor()
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

num_epochs = 40
train_losses, val_losses = [], []
best_val_loss = float('inf')
early_stop_patience = 10
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0

    for images, targets in tqdm(train_loader,  desc=f"Epoch {epoch+1} [Train]"):
        images = list(img.to(device) for img in images)  # Send all images to GPU/CPU
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]                       # Send all annotation values to device. v is a tensor, k is a string (no .to())

        loss_dict = model(images, targets)                                                         # Forward pass -> returns a dict of losses
        loss = sum(loss for loss in loss_dict.values())                                            # Loss is a combination of: classification loss, box regression loss, mask segmentation loss, objectness loss, RPN loss

        optimizer.zero_grad()                                                                      # Clear gradients from previous step
        loss.backward()                                                                            # Backpropagation: compute gradients w.r.t model params
        optimizer.step()                                                                           # Optimizer updates weights using gradients

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation step
    model.eval()  # Proper eval mode to disable dropout/batchnorm.
    epoch_val_loss = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()  # TEMPORARILY switch for loss computation
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            model.eval()  # Switch back to eval to keep behavior consistent

            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Scheduler step, adjusts the learning rate dynamically based on validation loss.
    scheduler.step(avg_val_loss)                                                                   # If val_loss doesn’t improve for 20 epochs (patience=20), reduce the LR by half (factor=0.5).

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({                                                                               # Checkpoint dictionary
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),                                                # Contains only the weights, not architecture (reconstruct the model architecture and then load the weights).
            'optimizer_state_dict': optimizer.state_dict(),                                        # To resume training with the same momentum,
            'val_loss': avg_val_loss
        }, save_path)
        print(f" Saved Best Model (Epoch {epoch+1})")
    else:
        epochs_no_improve += 1

    if epochs_no_improve == early_stop_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs+1), val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()
