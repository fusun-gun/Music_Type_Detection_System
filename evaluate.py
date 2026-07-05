"""
Eğitilmiş modeli validation seti üzerinde değerlendirir.
Confusion matrix ve classification report üretir.

Kullanım:
    python evaluate.py
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

VAL_DIR = "data/val"
CHECKPOINT_DIR = "checkpoints"
IMG_SIZE = 224
BATCH_SIZE = 32


def load_model(num_classes, device):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"{CHECKPOINT_DIR}/classes.json") as f:
        classes = json.load(f)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model(len(classes), device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n--- Classification Report ---")
    report = classification_report(all_labels, all_preds, target_names=classes)
    print(report)

    with open(f"{CHECKPOINT_DIR}/classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.title("Confusion Matrix - Müzik Türü Sınıflandırma")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{CHECKPOINT_DIR}/confusion_matrix.png")
    print(f"\nConfusion matrix kaydedildi: {CHECKPOINT_DIR}/confusion_matrix.png")
    print(f"Rapor kaydedildi: {CHECKPOINT_DIR}/classification_report.txt")


if __name__ == "__main__":
    main()
