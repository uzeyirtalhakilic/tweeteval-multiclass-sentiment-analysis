"""
Bu modül, tweet_eval veri seti için model değerlendirme fonksiyonlarını içerir.
"""

import logging
import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from sklearn.model_selection import cross_val_score

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(y_true, y_pred, model_name="model"):
    """
    Model performansını değerlendirir ve sonuçları kaydeder.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        model_name: Model adı (sonuç dosyaları için)
        
    Returns:
        Sınıflandırma raporu
    """
    # Sınıf isimleri
    labels = ['negative', 'neutral', 'positive']
    
    # Sınıflandırma raporu
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    # Sonuçları yazdır
    print(f"\n{model_name} Sınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=labels))
    
    # Karmaşıklık matrisi
    cm = confusion_matrix(y_true, y_pred)
    
    # Sonuçları kaydet
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Performans metriklerini kaydet
    metrics_path = os.path.join(results_dir, f"performance_metrics_{model_name}.json")
    with open(metrics_path, 'w') as f:
        json.dump(report, f, indent=4)
    logging.info(f"Performans metrikleri kaydedildi: {metrics_path}")
    
    # Karmaşıklık matrisini görselleştir ve kaydet
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(results_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_path)
    plt.close()
    logging.info(f"Karmaşıklık matrisi kaydedildi: {cm_path}")
    
    return report

def compare_models(results):
    """
    Modellerin performanslarını karşılaştırır ve sonuçları kaydeder.
    
    Args:
        results: Model sonuçları sözlüğü
        
    Returns:
        Karşılaştırma tablosu
    """
    # Karşılaştırma tablosu oluştur
    comparison_data = []
    
    for model_name, report in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': report['accuracy'],
            'F1-Score': report['weighted avg']['f1-score'],
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall']
        })
    
    comparison = pd.DataFrame(comparison_data)
    
    # Sonuçları kaydet
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    comparison_path = os.path.join(results_dir, "model_comparison.csv")
    comparison.to_csv(comparison_path, index=False)
    logging.info(f"Model karşılaştırma tablosu kaydedildi: {comparison_path}")
    
    # Karşılaştırma grafiği
    plt.figure(figsize=(12, 6))
    comparison.set_index('Model')[['Accuracy', 'F1-Score', 'Precision', 'Recall']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    graph_path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(graph_path)
    plt.close()
    logging.info(f"Model karşılaştırma grafiği kaydedildi: {graph_path}")
    
    return comparison

def cross_validate_model(model, X, y, cv=5):
    """
    Modeli çapraz doğrulama ile değerlendirir.
    
    Args:
        model: Değerlendirilecek model
        X: Özellikler
        y: Etiketler
        cv: Çapraz doğrulama katlama sayısı
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"\nÇapraz Doğrulama Sonuçları ({cv}-fold):")
    print(f"Ortalama Doğruluk: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
