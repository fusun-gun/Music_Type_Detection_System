"""
GTZAN ses dosyalarını 3 saniyelik parçalara böler, her parça için mel-spektrogram
görüntüsü üretir ve train/val klasörlerine ayırır.

Matematiksel temel (rapor için):
  1. Ses sinyali kısa pencerelerle STFT (Short-Time Fourier Transform) ile
     frekans-zaman temsiline dönüştürülür.
  2. Güç spektrumu, insan işitmesine yakın olan Mel ölçeğine (logaritmik
     frekans ölçeği) eşlenir -> Mel-spektrogram.
  3. Değerler dB (logaritmik) ölçeğe çevrilir, böylece hem düşük hem yüksek
     enerjili bölgeler görüntüde ayırt edilebilir olur.
  4. Sonuç, CNN'in bir görüntü gibi işleyebileceği 2 boyutlu bir matris olur.

Kullanım:
    python prepare_data.py
"""

import random
from pathlib import Path

import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Ayarlar -------------------------------------------------------------

RAW_AUDIO_DIR = Path("data/genres_original")   # GTZAN wav dosyalarının bulunduğu yer
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")

SEGMENT_DURATION = 3          # her parça 3 saniye
SAMPLE_RATE = 22050
N_MELS = 128
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# GTZAN'da bilinen bozuk dosya, varsa atlanır
KNOWN_CORRUPT_FILES = {"jazz.00054.wav"}

# ---------------------------------------------------------------------------


def audio_to_melspectrogram_image(y, sr, out_path):
    """Bir ses parçasını mel-spektrogram görüntüsü olarak kaydeder."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)  # ~224x224 px -> CNN girişine uygun
    ax = plt.axes()
    ax.set_axis_off()
    librosa.display.specshow(mel_db, sr=sr, ax=ax)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def process_genre(genre_dir, train_out, val_out, rng):
    wav_files = sorted(genre_dir.glob("*.wav"))
    random.Random(rng).shuffle(wav_files)

    n_val_files = max(1, int(len(wav_files) * VAL_SPLIT))
    val_files = set(wav_files[:n_val_files])

    count_train, count_val = 0, 0

    for wav_path in wav_files:
        if wav_path.name in KNOWN_CORRUPT_FILES:
            print(f"  Atlandı (bilinen bozuk dosya): {wav_path.name}")
            continue

        try:
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"  Atlandı ({wav_path.name}): {e}")
            continue

        segment_len = SEGMENT_DURATION * sr
        n_segments = len(y) // segment_len

        target_dir = val_out if wav_path in val_files else train_out

        for i in range(n_segments):
            segment = y[i * segment_len:(i + 1) * segment_len]
            out_path = target_dir / f"{wav_path.stem}_seg{i}.png"
            audio_to_melspectrogram_image(segment, sr, out_path)

            if target_dir == train_out:
                count_train += 1
            else:
                count_val += 1

    return count_train, count_val


def main():
    if not RAW_AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"{RAW_AUDIO_DIR} bulunamadı. GTZAN veri setini indirip "
            f"data/genres_original/<tur>/*.wav şeklinde yerleştirin."
        )

    genres = sorted([d for d in RAW_AUDIO_DIR.iterdir() if d.is_dir()])
    print(f"{len(genres)} tür bulundu: {[g.name for g in genres]}\n")

    for genre_dir in genres:
        train_out = TRAIN_DIR / genre_dir.name
        val_out = VAL_DIR / genre_dir.name
        train_out.mkdir(parents=True, exist_ok=True)
        val_out.mkdir(parents=True, exist_ok=True)

        print(f"İşleniyor: {genre_dir.name}")
        n_train, n_val = process_genre(genre_dir, train_out, val_out, RANDOM_SEED)
        print(f"  -> train: {n_train} parça, val: {n_val} parça\n")

    print("Tamamlandı. data/train ve data/val klasörlerini kontrol edin.")


if __name__ == "__main__":
    main()
