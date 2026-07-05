"""
Mel-spektrogram görüntüleri üzerinde MobileNetV2 tabanlı transfer learning ile
müzik türü sınıflandırma modeli eğitir.

Kullanım:
    python train.py
"""

import json
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# --- Ayarlar -----------------------------------------------------------

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
CHECKPOINT_DIR = "checkpoints"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_WORKERS = 2

# -------------------------------------------------------------------------


def get_dataloaders():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Not: Spektrogram görüntülerinde yatay çevirme (flip) veya rotasyon
    # ANLAMSIZDIR (zaman ve frekans eksenlerini bozar). Bu yüzden burada
    # klasik görüntü augmentasyonu kullanmıyoruz, sadece boyutlandırıyoruz.
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)

    return train_loader, val_loader, train_ds.classes


def build_model(num_classes, device):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Transfer learning: önceden eğitilmiş özellik katmanlarını dondurup
    # sadece sınıflandırıcı katmanı bu veri setine göre eğitiyoruz.
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, classes = get_dataloaders()
    print(f"Sınıflar ({len(classes)}): {classes}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(f"{CHECKPOINT_DIR}/classes.json", "w") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    model = build_model(len(classes), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} | "
              f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
            print(f"  -> Yeni en iyi model kaydedildi (val_acc: {val_acc:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{CHECKPOINT_DIR}/training_history.png")
    print(f"\nEn iyi validation accuracy: {best_val_acc:.4f}")
    print(f"Grafik kaydedildi: {CHECKPOINT_DIR}/training_history.png")


if __name__ == "__main__":
    main()
