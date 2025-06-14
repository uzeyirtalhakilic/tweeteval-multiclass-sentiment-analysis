# Tweet Duygu Analizi

Bu proje, [tweet_eval](https://huggingface.co/datasets/tweet_eval) veri seti kullanılarak **çok sınıflı duygu analizi** (pozitif, nötr, negatif) gerçekleştirmeyi amaçlamaktadır.

## 📦 Veri Seti

Veri seti, Hugging Face'in datasets kütüphanesi üzerinden otomatik olarak indirilmektedir. Veri seti şu özelliklere sahiptir:

- Eğitim seti: 45,000 tweet
- Test seti: 10,000 tweet
- Etiketler: 0 (negatif), 1 (nötr), 2 (pozitif)

## 🚀 Kurulum

1. Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

2. Programı çalıştırın:

```bash
python src/main.py
```

## 📊 Model Performansı

Model, test seti üzerinde şu metriklerle değerlendirilir:

- Doğruluk (Accuracy)
- Kesinlik (Precision)
- Duyarlılık (Recall)
- F1-skoru

## 📝 Proje Yapısı

```
.
├── data/               # Veri seti
├── models/            # Eğitilmiş modeller
├── results/           # Değerlendirme sonuçları
├── src/               # Kaynak kodlar
│   ├── main.py       # Ana program
│   ├── model.py      # Model fonksiyonları
│   ├── preprocessing.py  # Veri ön işleme
│   └── evaluation.py # Değerlendirme fonksiyonları
└── notebooks/         # Jupyter notebook'lar
    └── sentiment_analysis.ipynb
```

## 🔍 Kullanım

1. Veri seti otomatik olarak indirilir ve yüklenir
2. Metinler TF-IDF ile vektörleştirilir
3. Lojistik regresyon modeli eğitilir
4. Model performansı değerlendirilir
5. Örnek tahminler yapılır

## 📈 Sonuçlar

Model performans metrikleri ve karmaşıklık matrisi `results/` dizininde saklanır.
