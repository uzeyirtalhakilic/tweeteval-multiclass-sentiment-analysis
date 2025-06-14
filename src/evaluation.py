"""
Model değerlendirme modülü.
Bu modül, eğitilmiş modelin performansını değerlendirmek için gerekli fonksiyonları içerir.
"""

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

def evaluate_model(y_test, y_pred, labels=["negative", "neutral", "positive"]):
    """
    Model performansını değerlendirir ve sonuçları görselleştirir.
    
    Parametreler:
        y_test: Gerçek etiketler
        y_pred: Model tahminleri
        labels: Sınıf etiketleri
        
    Çıktılar:
        - Sınıflandırma raporu (precision, recall, f1-score)
        - Karmaşıklık matrisi grafiği (confusion matrix)
        - Performans metrikleri JSON dosyası
    """
    # Sınıflandırma raporunu hesapla ve yazdır
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    # Karmaşıklık matrisini oluştur
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Grafik boyutunu ayarla
    plt.figure(figsize=(10, 8))
    
    # Karmaşıklık matrisini görselleştir
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.title("Karmaşıklık Matrisi")
    
    # Grafiği kaydet
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # Performans metriklerini kaydet
    with open('results/metrics.json', 'w') as f:
        json.dump(report, f, indent=4)

def cross_validate_model(model, X, y, cv=5):
    """
    Modeli çapraz doğrulama ile değerlendirir.
    
    Parametreler:
        model: Değerlendirilecek model
        X: Özellikler
        y: Etiketler
        cv: Çapraz doğrulama katlama sayısı
        
    Çıktılar:
        - Her katlama için doğruluk skorları
        - Ortalama doğruluk ve standart sapma
    """
    # Çapraz doğrulama skorlarını hesapla
    scores = cross_val_score(model, X, y, cv=cv)
    
    # Sonuçları yazdır
    print(f"Çapraz doğrulama skorları: {scores}")
    print(f"Ortalama doğruluk: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    return scores
