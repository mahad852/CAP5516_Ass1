import argparse
import json
from models.ResNet18 import ResNet18
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from dataset.ChestXrayDataset import ChestXrayDataset
from torchvision.models import ResNet18_Weights
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from tqdm import tqdm
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
from matplotlib import colormaps
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="where chest xray data is stored")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--log_dir", default="logs/")
    args = parser.parse_args()

    return args

def visualize_cam_on_original_image(
    cam_map_tensor: torch.Tensor,
    original_image_pil: Image.Image,
) -> np.ndarray:

    H_orig, W_orig = original_image_pil.height, original_image_pil.width
    cam_np = cam_map_tensor.detach().cpu().numpy()
    heatmap_final = cv2.resize(cam_np, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    img_np = np.array(original_image_pil)
    colormap = colormaps['jet']
    heatmap_color = colormap(heatmap_final)[:, :, :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    heatmap_bgr = cv2.cvtColor(heatmap_color, cv2.COLOR_RGB2BGR)

    overlaid_img = cv2.addWeighted(img_bgr, 0.8, heatmap_bgr, 0.2, 0)

    return cv2.cvtColor(overlaid_img, cv2.COLOR_BGR2RGB)


def get_model(args):
    return ResNet18(num_classes=2, load_pretrained_weights=False)

def get_transforms():
    return transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_test_loader(args):
    test_path = os.path.join(args.root, "test")

    if not os.path.exists(test_path):
        raise ValueError(f"Dataset test path: {test_path} does not exist")
    
    trnsfrms = get_transforms()
    
    test_ds = ChestXrayDataset(root=test_path, x_transform=trnsfrms)
    test_loader = DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    return test_loader



def get_target_layer(model: nn.Module):
    target_layer = model.model.layer4[-1].conv2
    return target_layer

def main():
    args = get_args()

    test_loader = get_test_loader(args)
    device = get_device()

    print("[DEBUG] using device:", device)

    model = get_model(args)
    model = model.to(device=device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True), strict=True) 
    
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    cam_dir = os.path.join(log_dir, "cams")
    pneumonia_cam_dir = os.path.join(cam_dir, "pneumonia")
    normal_cam_dir = os.path.join(cam_dir, "normal")
    pneumonia_cam_correct_dir = os.path.join(pneumonia_cam_dir, "correct")
    pneumonia_cam_incorrect_dir = os.path.join(pneumonia_cam_dir, "incorrect")
    normal_cam_correct_dir = os.path.join(normal_cam_dir, "correct")
    normal_cam_incorrect_dir = os.path.join(normal_cam_dir, "incorrect")
    os.makedirs(pneumonia_cam_correct_dir, exist_ok=True)
    os.makedirs(pneumonia_cam_incorrect_dir, exist_ok=True)
    os.makedirs(normal_cam_correct_dir, exist_ok=True)
    os.makedirs(normal_cam_incorrect_dir, exist_ok=True)


    criterion = nn.CrossEntropyLoss()

    target_layer = get_target_layer(model=model)
    gcam = GradCAM(model=model, target_layers=[target_layer])
    gcam.batch_size = 32

    avg_loss = 0.0
    total_elems = 0
    
    all_probs = []
    all_labels = []
    
    model.eval()

    for (imgs, labels, img_paths) in test_loader:
        imgs: torch.Tensor
        labels: torch.Tensor

        imgs = imgs.to(device=device)
        labels = labels.to(device=device)

        outputs = model(imgs)

        loss = criterion(outputs, labels)

        avg_loss += loss.detach().item() * imgs.shape[0]
        total_elems += imgs.shape[0]

        probs = outputs.detach().softmax(dim=1)
        all_probs.append(probs)
        all_labels.append(labels)

        preds = probs.argmax(dim=1).cpu()

        for img, label, pred, img_path in zip(imgs, labels, preds, img_paths):
            img = img.unsqueeze(0)
            label = int(label.cpu().item())
            pred = int(pred.item())

            targets = [ClassifierOutputTarget(pred)]
            with torch.enable_grad():
                grayscale_cam = gcam(input_tensor=img, targets=targets)
                grayscale_cam = grayscale_cam[0]  # [H, W], numpy

            cam_map_continuous = torch.from_numpy(grayscale_cam).to(device).float()
            cam_map_continuous = (cam_map_continuous - cam_map_continuous.min()) / (
                cam_map_continuous.max() - cam_map_continuous.min() + 1e-8
            )

            pil_img = Image.open(img_path, "r").convert("RGB")
            overlaid_img = visualize_cam_on_original_image(cam_map_tensor=cam_map_continuous, original_image_pil=pil_img)
            
            if label == 0:
                save_path = normal_cam_correct_dir if label == pred else normal_cam_incorrect_dir
            else:    
                save_path = pneumonia_cam_correct_dir if label == pred else pneumonia_cam_incorrect_dir
            
            img_fname = img_path.split(os.sep)[-1]
            plt.imsave(os.path.join(save_path, img_fname), overlaid_img)


    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = all_probs.argmax(dim=1)

    all_probs_np, all_labels_np, all_preds_np = all_probs.cpu().numpy(), all_labels.cpu().numpy(), all_preds.cpu().numpy()

    f1 = f1_score(y_true=all_labels_np, y_pred=all_preds_np)
    accuracy = accuracy_score(y_true=all_labels_np, y_pred=all_preds_np)
    prec = precision_score(y_true=all_labels_np, y_pred=all_preds_np)
    rec = recall_score(y_true=all_labels_np, y_pred=all_preds_np)
    
    avg_loss /= total_elems

    print(f"Test Loss: {avg_loss:.4f} | Test Acc: {accuracy:.4f} "
              f"F1-score: {f1:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f}")    


if __name__ == "__main__":
    main()

    
    

    





