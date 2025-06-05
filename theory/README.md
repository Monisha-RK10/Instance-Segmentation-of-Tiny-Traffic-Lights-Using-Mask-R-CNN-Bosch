## SAM vs. Mask R-CNN: Evaluation Philosophy

| Aspect                    | **SAM (Segment Anything Model)**                               | **Mask R-CNN**                                                |
| ------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------- |
| **Segmentation Type**     | Class-agnostic / Point/Box-guided                              | Category-aware instance segmentation                          |
| **Prediction per Object** | One binary mask from SAM for each box or point                 | One soft mask per detected instance (then thresholded)        |
| **Category Info**         | Not included by default                                        | Included (`category_id`, `score`)                             |
| **Mask Format**           | Binary numpy mask (bool or 0/1)                                | Soft mask (0–1 float), thresholded & converted to **RLE**     |
| **Evaluation**            | Manual IoU with GT masks (from polygons → merged RLE → binary) | Standard **COCOeval** using `pycocotools`                     |
| **GT Format**             | COCO polygons → merge → binary mask                            | COCO polygons → RLE (per instance)                            |
| **Usage Goal**            | Evaluate SAM's mask quality vs GT                              | Full instance segmentation performance (mask + class + score) |

Note: This project is an extension to 'Hybrid Detection and Segmentation of Small Traffic Lights using YOLOv8 and SAM', hence, the comparison for better understanding.
