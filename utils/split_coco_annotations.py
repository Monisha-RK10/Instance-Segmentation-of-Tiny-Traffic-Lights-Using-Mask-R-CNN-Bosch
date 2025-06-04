# Step 2: COCO JSON Splitter Script
# Splits a single COCO annotation file (with all 100 images) into separate train.json and val.json.
# For each image: Checks if its file_name is in the train or val image folder. If yes, updates its ID (new_id), stores in image_id_map.
# For annotations: Keeps only annotations of the selected images. Updates their image_id using image_id_map and assigns new unique ann_id.
# For categories: Keeps the original categories as-is.

# Paths
full_json_path = "/content/drive/MyDrive/MASK-RCNN_TrafficLight/annotation.json"
train_img_dir = "/content/drive/MyDrive/MASK-RCNN_TrafficLight/images/train"
val_img_dir = "/content/drive/MyDrive/MASK-RCNN_TrafficLight/images/val"
output_train_json = "/content/drive/MyDrive/MASK-RCNN_TrafficLight/annotation_train.json"
output_val_json = "/content/drive/MyDrive/MASK-RCNN_TrafficLight/annotation_val.json"

# Load full JSON
with open(full_json_path) as f:
    full_data = json.load(f)

# Get image names in each split
train_imgs = set(os.listdir(train_img_dir))
val_imgs = set(os.listdir(val_img_dir))

def split_coco_json(full_data, image_names):
    new_images = []
    new_annotations = []
    image_id_map = {}  # old_id -> new_id
    new_id = 0
    ann_id = 0

    for img in full_data['images']:
        if img['file_name'] in image_names: # Keeps only images whose filenames are in image_names.
            image_id_map[img['id']] = new_id  # old_id -> new_id
            img_copy = img.copy() # Don't mutate original
            img_copy['id'] = new_id # Assign new image ID, Remaps their ids to a clean 0-based index.
            new_images.append(img_copy)  # Save it, Stores them in new_images.
            new_id += 1

    for ann in full_data['annotations']:
        old_id = ann['image_id']
        if old_id in image_id_map: # Keeps annotations linked to the kept images only.
            ann_copy = ann.copy()
            ann_copy['image_id'] = image_id_map[old_id] # Remaps image_id to the new ID (from step 1).
            ann_copy['id'] = ann_id # Gives each annotation a new unique id.
            new_annotations.append(ann_copy)
            ann_id += 1

    return {
        "images": new_images,
        "annotations": new_annotations,
        "categories": full_data["categories"]
    }

# Create train.json and val.json
train_json = split_coco_json(full_data, train_imgs)
val_json = split_coco_json(full_data, val_imgs)

# Save them
with open(output_train_json, "w") as f:
    json.dump(train_json, f)
with open(output_val_json, "w") as f:
    json.dump(val_json, f)

print("COCO JSONs split into train and val.")
