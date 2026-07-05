# Derin Öğrenme ile Müzik Türü Tespiti

Ses parçalarından mel-spektrogram görüntüleri üreterek, transfer learning
(MobileNetV2) tabanlı bir CNN ile müzik türü sınıflandırması yapan bir
derin öğrenme projesi.

**Sınıflandırılan türler (GTZAN veri seti):** blues, classical, country,
disco, hiphop, jazz, metal, pop, reggae, rock (10 tür).

---

## 1. Programın Kurulumu

### 1.1 Gereksinimler

- Python 3.9 veya üzeri
- (Opsiyonel ama önerilir) NVIDIA GPU + CUDA — eğitim süresini büyük ölçüde kısaltır

### 1.2 Ortamı Hazırlama

Bir sanal ortam oluşturmanız önerilir:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 1.3 Kütüphaneleri Kurma

```bash
pip install -r requirements.txt
```

### 1.4 GPU Desteğini Kontrol Etme

Kurulumdan sonra CUDA'nın PyTorch tarafından görülüp görülmediğini kontrol edin:

```python
import torch
print(torch.cuda.is_available())   # True dönmeli
```

`False` dönerse, `requirements.txt` CPU sürümünü kurmuş olabilir. Sisteminize
uygun CUDA destekli kurulum komutunu şu adresten alın:
https://pytorch.org/get-started/locally/

---

## 2. Kullanılan Kütüphaneler ve Görevleri

| Kütüphane | Görevi |
|---|---|
| **librosa** | Ses dosyalarını okuma, mel-spektrogram gibi ses özniteliklerini çıkarma |
| **torch / torchvision** | Derin öğrenme modelini (MobileNetV2 tabanlı CNN) kurma ve eğitme |
| **matplotlib** | Mel-spektrogram görüntülerini oluşturma, eğitim grafiklerini çizme |
| **seaborn** | Confusion matrix görselleştirmesi |
| **scikit-learn** | Classification report (precision/recall/F1) ve confusion matrix hesaplama |
| **Pillow (PIL)** | Görüntü işleme (spektrogram görüntülerini okuma/boyutlandırma) |
| **numpy** | Sayısal işlemler |

---

## 3. Veri Seti

**Kullanılan veri seti:** GTZAN Genre Collection

- **Link:** https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
- **İçerik:** 10 tür, tür başına 100 adet 30 saniyelik `.wav` dosyası (toplam 1000 parça)
- **Boyut:** ~1.2 GB

İndirdikten sonra `genres_original` klasörünü proje kök dizininde
`data/genres_original/` olacak şekilde yerleştirin:

```
data/
  genres_original/
    blues/
      blues.00000.wav
      blues.00001.wav
      ...
    classical/
      ...
    ...
```

> Not: Veri setinde `jazz.00054.wav` dosyası bozuk olarak bilinir; `prepare_data.py`
> bu dosyayı otomatik olarak atlar.

---

## 4. Veri Hazırlama Adımları (Ham Ses → Model Girdisi)

Bu projede geleneksel bir "veri tabanı" kullanılmıyor; onun yerine ham ses
verisi işlenerek görüntü tabanlı bir veri kümesine dönüştürülüyor:

1. **Segmentasyon:** Her 30 saniyelik parça, 3 saniyelik 10 alt parçaya bölünür
   (hem veri miktarını artırır hem modelin daha kısa klipler üzerinde de
   çalışabilmesini sağlar).
2. **Öznitelik çıkarımı:** Her 3 saniyelik parça için mel-spektrogram hesaplanır
   (STFT → Mel ölçeği → dB dönüşümü).
3. **Görüntüye dönüştürme:** Mel-spektrogram, 224x224 piksel boyutunda bir PNG
   görüntüsü olarak kaydedilir.
4. **Train/Val ayrımı:** Parçalar tür bazında %80 train / %20 validation olacak
   şekilde ayrılır (aynı şarkının farklı parçalarının hem train hem val'de
   olmaması için ayrım şarkı seviyesinde yapılır, veri sızıntısı önlenir).

Bu adımları çalıştırmak için:

```bash
python prepare_data.py
```

Çıktı olarak `data/train/<tür>/` ve `data/val/<tür>/` klasörleri oluşur.

---

## 5. Eğitim

```bash
python train.py
```

- Model: MobileNetV2 (ImageNet üzerinde önceden eğitilmiş), transfer learning
  ile son katman bu veri setine göre yeniden eğitilir.
- En iyi model `checkpoints/best_model.pth` olarak kaydedilir.
- Eğitim/doğrulama loss ve accuracy grafiği `checkpoints/training_history.png`
  olarak kaydedilir.

---

## 6. Değerlendirme

```bash
python evaluate.py
```

Confusion matrix (`checkpoints/confusion_matrix.png`) ve sınıf bazlı
precision/recall/F1 raporu (`checkpoints/classification_report.txt`) üretir.

---

## 7. Tek Bir Ses Dosyası ile Tahmin (Demo)

```bash
python predict.py ornek_sarki.wav
```

Ses dosyasını 3 saniyelik parçalara böler, her parça için ayrı tahmin yapar
ve çoğunluk oyuyla genel bir tür tahmini verir.

---

## 8. Proje Yapısı

```
muzik-turu-tespiti/
  data/
    genres_original/   (ham ses dosyaları - kullanıcı tarafından indirilir)
    train/             (üretilen spektrogram görüntüleri - eğitim)
    val/               (üretilen spektrogram görüntüleri - doğrulama)
  checkpoints/
    best_model.pth
    classes.json
    training_history.png
    confusion_matrix.png
    classification_report.txt
  prepare_data.py
  train.py
  evaluate.py
  predict.py
  requirements.txt
  README.md
```

---

## 9. Sınırlılıklar

- GTZAN veri seti küçük (1000 orijinal parça) ve bazı türler arasında
  (örn. rock/country/disco) belirgin akustik benzerlik var, bu da
  karışıklığa yol açabilir.
- Veri setinde bazı türler arasında tür sınırları öznel olabilir
  (bir parça birden fazla türe ait özellikler taşıyabilir).
- Mel-spektrogram tabanlı yaklaşım, ritim/tempo gibi zamansal bazı
  bilgileri CNN'in görüntü olarak işlemesi nedeniyle dolaylı yoldan öğrenir.
