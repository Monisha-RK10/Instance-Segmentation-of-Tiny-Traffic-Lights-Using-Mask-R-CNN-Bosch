# Mask Evaluation: Mask R-CNN vs. SAM vs. YOLO seg
This section clarifies the key differences in prediction format, thresholding, and evaluation pipeline when working with:

- Mask R-CNN (soft masks, COCOeval)
- SAM (binary masks, manual IoU)

| Aspect                    | **SAM (Segment Anything Model)**                               | **Mask R-CNN**                                                |
| ------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------- |
| **Segmentation Type**     | Class-agnostic / Point/Box-guided                              | Category-aware instance segmentation                          |
| **Prediction per Object** | One binary mask from SAM for each box or point                 | One soft mask per detected instance (then thresholded)        |
| **Category Info**         | Not included by default                                        | Included (`category_id`, `score`)                             |
| **Mask Format**           | Binary numpy mask (bool or 0/1)                                | Soft mask (0–1 float), thresholded & converted to **RLE**     |
| **Evaluation**            | Manual IoU with GT masks (from polygons → merged RLE → binary) | Standard **COCOeval** using `pycocotools`                     |
| **GT Format**             | COCO polygons → merge → binary mask                            | COCO polygons → RLE (per instance)                            |
| **Usage Goal**            | Evaluate SAM's mask quality vs GT                              | Full instance segmentation performance (mask + class + score) |

>Note: This project is an extension to 'Hybrid Detection and Segmentation of Small Traffic Lights using YOLOv8 and SAM', hence, the comparison for better understanding.



#  Prediction Format

| Model          | Output Shape               | Output Type               | Thresholding Needed?              |
| -------------- | -------------------------- | ------------------------- | --------------------------------- |
| **Mask R-CNN** | `[num_instances, 1, H, W]` | Soft masks ∈ `[0.0, 1.0]` | Yes (e.g., `> 0.25` or `> 0.5`) |
| **SAM**        | `[H, W]` (per mask)        | Binary masks (0 or 1)     | No (already thresholded)        |

# Ground Truth Format

Both models share the same GT annotation format, a JSON exported from Makesense.ai in COCO format (polygons). However, GT masks must be converted to binary masks before IoU comparison:

## Convert polygon → RLE → binary mask

> rle = mask_utils.frPyObjects(segmentation_poly, height, width)
> 
> rle = mask_utils.merge(rle)
> 
> gt_mask = mask_utils.decode(rle)  # binary mask shape (H, W)

# Evaluation Flow

| Step                  | **Mask R-CNN**                                   | **SAM**                                      |
| --------------------- | ------------------------------------------------ | -------------------------------------------- |
| Prediction format     | Soft mask per instance                           | Binary mask per prompt                       |
| Thresholding          | Required: `mask = mask > 0.25`                   | Already binary                             |
| Format for evaluation | Convert to RLE using `pycocotools.mask.encode()` | Used directly as binary mask                 |
| Evaluation method     | `pycocotools.COCOeval` (mAP, AP\@IoU thresholds) | Manual IoU between `pred_mask` and `gt_mask` |

**Notes**
> Mask R-CNN soft masks must be thresholded before converting to RLE. Common threshold = 0.5, but 0.25 gives higher recall (more lenient). Threshold affects predicted mask shape → IoU → mAP.
> 
> SAM masks are already binary (bool arrays), no need for thresholding. You can optionally convert SAM predictions to RLE + build a COCO-format prediction JSON for use with COCOeval, but this is not strictly necessary.

##  Code Snippet: Mask R-CNN

> soft_mask = masks[i, 0]  # shape: (H, W)
> 
> binary_mask = soft_mask > 0.25
> 
> rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
> 
> rle["counts"] = rle["counts"].decode("utf-8")

##  Code Snippet: SAM

> masks, scores, _ = predictor.predict(...)
> 
> pred_mask = masks[0]  # already binary
> 
> iou = compute_iou(pred_mask, gt_mask)

If you want to extend SAM’s evaluation to full COCO metrics, you can:

- Convert SAM's pred_mask to RLE.
- Build a prediction JSON.
- Use pycocotools.COCOeval.

## Comparing Mask R-CNN with SAM and YOLO seg

| Model      | Pred Format              | GT Format           | Eval Style         | Special Handling                |
| ---------- | ------------------------ | ------------------- | ------------------ | ------------------------------- |
| YOLO Seg   | Polygon `.txt`           | Polygon `.txt`      | Auto (Ultralytics) | No pycocotools needed           |
| SAM        | Binary Mask              | Polygon → Binary    | Manual IoU         | GT needs decode                 |
| Mask R-CNN | Soft Mask → Binary (RLE) | Polygon (COCO JSON) | `COCOeval`         | Threshold + RLE needed for pred |

