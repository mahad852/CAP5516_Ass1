from torch.utils.data import Dataset
from PIL import Image
import os

class ChestXrayDataset(Dataset):
    def __init__(self, root: str, x_transform = None):
        self.class_names_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}
        self.root = root
        self.x_transform = x_transform
        self.images, self.labels = self.load_images_and_labels()

    def __len__(self):
        return len(self.images)
    
    def load_images_and_labels(self):
        images = []
        labels = []

        for class_folder in os.listdir(self.root):
            if class_folder not in self.class_names_to_idx.keys():
                continue

            label = self.class_names_to_idx[class_folder]
            
            class_path = os.path.join(self.root, class_folder)
            
            for img_fname in os.listdir(class_path):
                ext = img_fname.split(".")[-1]
                if ext not in ["png", "jpg", "jpeg"]:
                    continue
                    
                labels.append(label)
                images.append(os.path.join(class_path, img_fname))
        
        return images, labels
    
    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        img = Image.open(img_path).convert("RGB")

        if self.x_transform:
            img = self.x_transform(img)
        
        return img, label, img_path
                

