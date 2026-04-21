import os, random, shutil
import xml.etree.ElementTree as ET
from pathlib import Path

base = Path("datasets/road_signs")
img_dir = base / "images"
ann_dir = base / "annotations"

# 1. Create the required YOLOv8 directory structure
for split in ['train', 'val']:
    (base / 'images' / split).mkdir(parents=True, exist_ok=True)
    (base / 'labels' / split).mkdir(parents=True, exist_ok=True)

# The actual classes in the Kaggle dataset
classes = {'trafficlight': 0, 'speedlimit': 1, 'crosswalk': 2, 'stop': 3}

xml_files = list(ann_dir.glob("*.xml"))
random.shuffle(xml_files)
split_idx = int(len(xml_files) * 0.8) # 80/20 split

# 2. Convert XML to TXT and move files
for i, xml_file in enumerate(xml_files):
    split = 'train' if i < split_idx else 'val'
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    img_filename = root.find('filename').text
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    
    txt_path = base / 'labels' / split / f"{xml_file.stem}.txt"
    
    with open(txt_path, 'w') as f:
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in classes: continue
            cls_id = classes[cls_name]
            
            box = obj.find('bndbox')
            xmin, ymin = int(box.find('xmin').text), int(box.find('ymin').text)
            xmax, ymax = int(box.find('xmax').text), int(box.find('ymax').text)
            
            # Convert to YOLO format (normalized center x, center y, width, height)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
    
    # Move the corresponding image
    src_img = img_dir / img_filename
    if src_img.exists():
        shutil.move(str(src_img), str(base / 'images' / split / img_filename))

print("Data formatting complete.")