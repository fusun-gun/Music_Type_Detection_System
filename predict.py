"""
Tek bir ses dosyasını (wav/mp3) alır, 3 saniyelik parçalara böler, her parça
için tahmin yapar ve çoğunluk oyuyla (majority voting) genel bir tür tahmini
verir. Video demosu için kullanışlıdır.

Kullanım:
    python predict.py yol/dosya_adi.wav
"""

import sys
import json
from collections import Counter

import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

CHECKPOINT_DIR = "checkpoints"
IMG_SIZE = 224
SAMPLE_RATE = 22050
SEGMENT_DURATION = 3
N_MELS = 128


def audio_segment_to_tensor(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = plt.axes()
    ax.set_axis_off()
    librosa.display.specshow(mel_db, sr=sr, ax=ax)
    fig.canvas.draw()
    image = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return transform(image)


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python predict.py yol/dosya_adi.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"{CHECKPOINT_DIR}/classes.json") as f:
        classes = json.load(f)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    segment_len = SEGMENT_DURATION * sr
    n_segments = len(y) // segment_len

    if n_segments == 0:
        print("Ses dosyası çok kısa, en az 3 saniye olmalı.")
        sys.exit(1)

    predictions = []
    print(f"{n_segments} parça analiz ediliyor...\n")

    for i in range(n_segments):
        segment = y[i * segment_len:(i + 1) * segment_len]
        tensor = audio_segment_to_tensor(segment, sr).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_idx = probs.argmax().item()

        predictions.append(classes[pred_idx])
        print(f"  Parça {i+1} ({i*3}-{(i+1)*3}s): {classes[pred_idx]} "
              f"(güven: {probs[pred_idx].item()*100:.1f}%)")

    vote = Counter(predictions).most_common(1)[0]
    print(f"\nGenel tahmin (çoğunluk oyu): {vote[0]} ({vote[1]}/{len(predictions)} parça)")


if __name__ == "__main__":
    main()
