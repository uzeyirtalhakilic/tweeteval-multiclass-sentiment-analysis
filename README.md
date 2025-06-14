# Tweet Duygu Analizi

Bu proje, tweet_eval veri seti kullanılarak tweet'lerin duygu analizini gerçekleştiren bir makine öğrenmesi uygulamasıdır.

## Veri Seti

Proje, [tweet_eval](https://huggingface.co/datasets/tweet_eval) veri setinin "sentiment" alt kümesini kullanmaktadır. Veri seti şu özelliklere sahiptir:

- Eğitim seti: 45,615 tweet
- Test seti: 12,284 tweet
- Sınıf sayısı: 3 (negatif, nötr, pozitif)
- Etiket dağılımı:
  - Negatif (0): 3,972 tweet
  - Nötr (1): 5,937 tweet
  - Pozitif (2): 2,375 tweet

## Kullanılan Modeller

Projede aşağıdaki modeller denenmiştir:

1. **Logistic Regression**

   - Hiperparametreler: C, solver, max_iter
   - Performans: Accuracy, Precision, Recall, F1-Score

2. **Support Vector Machine (SVM)**

   - Hiperparametreler: C, max_iter
   - Performans: Accuracy, Precision, Recall, F1-Score

3. **Random Forest**

   - Hiperparametreler: n_estimators, max_depth, min_samples_split
   - Performans: Accuracy, Precision, Recall, F1-Score

4. **Naive Bayes**

   - Hiperparametreler: alpha
   - Performans: Accuracy, Precision, Recall, F1-Score

5. **Neural Network**
   - Hiperparametreler: hidden_layer_sizes, activation, alpha
   - Performans: Accuracy, Precision, Recall, F1-Score

## Kurulum

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

2. Projeyi çalıştırın:

```bash
python src/main.py
```

## Proje Yapısı

```
.
├── src/
│   ├── main.py           # Ana program
│   ├── model.py          # Model tanımlamaları ve eğitim fonksiyonları
│   ├── evaluation.py     # Model değerlendirme fonksiyonları
│   └── preprocessing.py  # Veri ön işleme fonksiyonları
├── models/               # Eğitilmiş modeller
├── results/             # Değerlendirme sonuçları
│   ├── performance_metrics_*.json  # Model performans metrikleri
│   ├── confusion_matrix_*.png      # Karmaşıklık matrisleri
│   ├── model_comparison.csv        # Model karşılaştırma tablosu
│   └── model_comparison.png        # Model karşılaştırma grafiği
└── notebooks/           # Jupyter notebook'lar
    └── sentiment_analysis.ipynb    # Analiz ve görselleştirme
```

## Model Performansı

Her model için aşağıdaki metrikler hesaplanmıştır:

- Accuracy (Doğruluk)
- Precision (Kesinlik)
- Recall (Duyarlılık)
- F1-Score (F1 Değeri)

Sonuçlar `results/` dizininde saklanmaktadır:

- `performance_metrics_{model_name}.json`: Her model için detaylı performans metrikleri
- `confusion_matrix_{model_name}.png`: Her model için karmaşıklık matrisi
- `model_comparison.csv`: Tüm modellerin karşılaştırmalı performans tablosu
- `model_comparison.png`: Tüm modellerin karşılaştırmalı performans grafiği

## Kullanım

1. Model eğitimi ve değerlendirme:

```python
python src/main.py
```

2. Örnek tahmin:

```python
from src.model import load_model, predict_sentiment

# En iyi modeli yükle
model, vectorizer = load_model("best_model_name")

# Tahmin yap
text = "I love this product! It's amazing!"
prediction, probability = predict_sentiment(model, vectorizer, text)
```

## Geliştirme

Projeyi geliştirmek için yapılabilecek iyileştirmeler:

1. Daha gelişmiş metin ön işleme teknikleri
2. BERT, RoBERTa gibi transformer modellerinin eklenmesi
3. Hiperparametre optimizasyonunun genişletilmesi
4. Veri dengesizliği sorununun çözülmesi
5. Daha detaylı model analizi ve görselleştirme
