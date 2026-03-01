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
import matplotlib.pyplot as plt
from torchvision import transforms

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="where chest xray data is stored")
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--use_pretrained", action="store_true")

    args = parser.parse_args()

    return args

def plot_graph(xs, ys, title, path, x_label, y_label, legend_labels):
    plt.cla()
    for x, y, label in zip(xs, ys, legend_labels):
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel=x_label)
    plt.ylabel(ylabel=y_label)
    plt.legend()

    plt.savefig(path)


def get_model(args):
    return ResNet18(num_classes=2, load_pretrained_weights=args.use_pretrained)

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def get_test_transforms():
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

def get_dataloaders(args):
    train_path = os.path.join(args.root, "train")
    val_path = os.path.join(args.root, "val")
    test_path = os.path.join(args.root, "test")

    if not os.path.exists(train_path):
        raise ValueError(f"Dataset train path: {train_path} does not exist")
    
    if not os.path.exists(val_path):
        raise ValueError(f"Dataset train path: {val_path} does not exist")
    
    if not os.path.exists(test_path):
        raise ValueError(f"Dataset train path: {test_path} does not exist")
    
    train_trnsfrms = get_train_transforms()
    test_trnsfrms = get_test_transforms()
    
    train_ds = ChestXrayDataset(root=train_path, x_transform=train_trnsfrms)
    val_ds = ChestXrayDataset(root=val_path, x_transform=test_trnsfrms)
    test_ds = ChestXrayDataset(root=test_path, x_transform=test_trnsfrms)

    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader

def train_for_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, device: torch.device):
    criterion = nn.CrossEntropyLoss()

    avg_loss = 0.0
    total_elems = 0
    num_correct = 0.0

    model.train()

    for (imgs, labels, _) in tqdm(train_loader, desc="Model training"):
        optimizer.zero_grad()

        imgs = imgs.to(device=device)
        labels = labels.to(device=device)

        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        avg_loss += loss.detach().item()
        total_elems += imgs.shape[0]

        num_correct += (outputs.detach().softmax(dim=1).argmax(dim=1) == labels).sum().cpu().item()

    avg_loss /= total_elems
    acc = num_correct / total_elems
    return avg_loss, acc

@torch.no_grad
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    criterion = nn.CrossEntropyLoss()

    avg_loss = 0.0
    total_elems = 0
    
    all_probs = []
    all_labels = []
    
    model.eval()

    for (imgs, labels, _) in loader:
        imgs: torch.Tensor
        labels: torch.Tensor

        imgs = imgs.to(device=device)
        labels = labels.to(device=device)

        outputs = model(imgs)

        loss = criterion(outputs, labels)

        avg_loss += loss.detach().item()
        total_elems += imgs.shape[0]

        probs = outputs.detach().softmax(dim=1)
        all_probs.append(probs)
        all_labels.append(labels)

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = all_probs.argmax(dim=1)

    all_probs_np, all_labels_np, all_preds_np = all_probs.cpu().numpy(), all_labels.cpu().numpy(), all_preds.cpu().numpy()

    f1 = f1_score(y_true=all_labels_np, y_pred=all_preds_np)
    accuracy = accuracy_score(y_true=all_labels_np, y_pred=all_preds_np)
    prec = precision_score(y_true=all_labels_np, y_pred=all_preds_np)
    rec = recall_score(y_true=all_labels_np, y_pred=all_preds_np)
    
    avg_loss /= total_elems
    
    return {"loss": avg_loss, "f1": f1, "acc": accuracy, "prec": prec, "rec": rec, "f1": f1}


def main():
    args = get_args()

    train_loader, val_loader, test_loader = get_dataloaders(args)
    device = get_device()

    print("[DEBUG] using device:", device)

    model = get_model(args)
    model = model.to(device=device)

    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    best_model_path = os.path.join(log_dir, "best.pth")

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=0.01)
    log_obj = {
        "train_loss": [],
        "train_accs": [],
        "val_metrics": []
    }

    best_val_loss = np.inf

    for epoch in tqdm(range(args.epochs), desc="Running epochs"):
        loss, train_acc = train_for_one_epoch(model=model, train_loader=train_loader, optimizer=optimizer, device=device)
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"Train loss: {loss:.4f} | Train Acc: {train_acc:.4f}")
        val_metrics = evaluate(model=model, loader=test_loader, device=device)

        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.4f} "
              f"F1-score: {val_metrics['f1']:.4f} | Prec: {val_metrics['prec']:.4f} | Recall: {val_metrics['rec']:.4f}")        
        log_obj["train_loss"].append(loss)
        log_obj["train_accs"].append(train_acc)
        log_obj["val_metrics"].append(val_metrics)

        if val_metrics["loss"] < best_val_loss:
            torch.save(model.state_dict(), best_model_path)
            best_val_loss = val_metrics["loss"]
    

    test_metrics = evaluate(model=model, loader=test_loader, device=device)
    log_obj["test_metrics"] = test_metrics

    print("---" * 10)
    print(f"Test Loss: {val_metrics['loss']:.4f} | Test Acc: {val_metrics['acc']:.4f} "
              f"F1-score: {val_metrics['f1']:.4f} | Prec: {val_metrics['prec']:.4f} | Recall: {val_metrics['rec']:.4f}")    
    with open(os.path.join(log_dir, "training_logs.json"), "w") as f:
        json.dump(log_obj, f)

    train_losses = log_obj["train_loss"]
    val_losses = list(map(lambda o: o["loss"], log_obj["val_metrics"]))

    epochs = np.arange(args.epochs) + 1

    train_accs = log_obj["train_accs"]
    val_accs = list(map(lambda o: o["acc"], log_obj["val_metrics"]))
    val_f1s = list(map(lambda o: o["f1"], log_obj["val_metrics"]))

    plot_graph(xs=[epochs, epochs], ys=[train_losses, val_losses], title="Loss vs. # Epochs", x_label="Epochs", y_label="Loss", legend_labels=["Train", "Test"], path=os.path.join(log_dir, "loss.png"))
    plot_graph(xs=[epochs, epochs], ys=[train_accs, val_accs], title="Accuracy vs. # Epochs", x_label="Epochs", y_label="Accuracy", legend_labels=["Train", "Test"], path=os.path.join(log_dir, "accuracy.png"))
    plot_graph(xs=[epochs], ys=[val_f1s], title="F1-score vs. # Epochs", x_label="Epochs", y_label="F1 Score", legend_labels=["Test F1-score"], path=os.path.join(log_dir, "f11.png"))



if __name__ == "__main__":
    main()

    
    

    





