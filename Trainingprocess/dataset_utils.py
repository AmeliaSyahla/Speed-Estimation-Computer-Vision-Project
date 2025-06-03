import os
import torch
import torch.utils.data
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import transforms
from tqdm import tqdm 

class PascalVOCDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        
        self.valid_image_files = []
        self.annotation_map = {} 
        
        all_image_candidates = []
        all_xml_candidates = {}

        print(f"Scanning directory: {self.data_dir}...")
        for f_name in os.listdir(self.data_dir):
            if f_name.endswith('.jpg'):
                all_image_candidates.append(f_name)
            elif f_name.endswith('.xml'):
                base_name_xml = f_name.split('.rf.')[0] + '.jpg'
                all_xml_candidates[base_name_xml] = f_name
        print(f"Found {len(all_image_candidates)} JPGs and {len(all_xml_candidates)} XMLs.")

        self.class_to_id = {'car': 1, 'bus': 2, 'truck': 3} 
        
        print("Filtering images with no objects...")
        for img_full_name in tqdm(sorted(all_image_candidates)):
            base_img_name_for_xml = img_full_name.split('.rf.')[0] + '.jpg'
            
            if base_img_name_for_xml not in all_xml_candidates:
                continue

            annotation_full_name = all_xml_candidates[base_img_name_for_xml]
            annotation_path = os.path.join(self.data_dir, annotation_full_name)

            try:
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                
                has_objects = False
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in self.class_to_id:
                        has_objects = True
                        break

                if has_objects:
                    self.valid_image_files.append(img_full_name)
                    self.annotation_map[img_full_name] = annotation_full_name
            except ET.ParseError:
                print(f"Warning: Could not parse XML file {annotation_full_name}. Skipping.")
                continue
            except FileNotFoundError:
                print(f"Warning: XML file not found {annotation_full_name}. Skipping.")
                continue
            except Exception as e:
                print(f"Error processing {annotation_full_name}: {e}. Skipping.")
                continue
        
        self.valid_image_files = sorted(self.valid_image_files)
        print(f"Dataset initialized with {len(self.valid_image_files)} images (after filtering).")
        print(f"Original images found: {len(all_image_candidates)}. Filtered {len(all_image_candidates) - len(self.valid_image_files)} images.")

    def __getitem__(self, idx):
        img_full_name = self.valid_image_files[idx]
        img_path = os.path.join(self.data_dir, img_full_name)
        
        annotation_full_name = self.annotation_map[img_full_name]
        annotation_path = os.path.join(self.data_dir, annotation_full_name)

        img = Image.open(img_path).convert("RGB")
        
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in self.class_to_id:
                xmin = float(obj.find('bndbox/xmin').text)
                ymin = float(obj.find('bndbox/ymin').text)
                xmax = float(obj.find('bndbox/xmax').text)
                ymax = float(obj.find('bndbox/ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_id[name])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx]) 
        
        if boxes.numel() == 0: 
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        else:
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img) 

        return img, target

    def __len__(self):
        return len(self.valid_image_files)


def get_transform(train):
    transforms_list = []
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)

def collate_fn(batch):
    return tuple(zip(*batch))