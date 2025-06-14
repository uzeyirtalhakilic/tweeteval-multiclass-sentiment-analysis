"""
Tweet_eval veri seti için duygu analizi ana programı.
Bu modül, veri yükleme, model eğitimi ve değerlendirme adımlarını içerir.
"""

import logging
import os
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from model import get_models, train_model, save_model, predict_sentiment, load_model
from evaluation import evaluate_model, cross_validate_model, compare_models

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """
    Tweet_eval veri setini yükler ve pandas DataFrame'e dönüştürür.
    
    Returns:
        train_data: Eğitim veri seti
        test_data: Test veri seti
    """
    logging.info("Veri seti yükleniyor...")
    dataset = load_dataset("tweet_eval", "sentiment")
    
    # Veri setini pandas DataFrame'e dönüştür
    train_data = pd.DataFrame(dataset['train'])
    test_data = pd.DataFrame(dataset['test'])
    
    logging.info(f"Eğitim seti boyutu: {len(train_data)}")
    logging.info(f"Test seti boyutu: {len(test_data)}")
    
    return train_data, test_data

def main():
    """
    Ana program akışı.
    """
    # Veri setini yükle
    train_data, test_data = load_data()
    
    # TF-IDF vektörleştirici oluştur
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_data['text'])
    X_test = vectorizer.transform(test_data['text'])
    y_train = train_data['label']
    y_test = test_data['label']
    
    # Modelleri eğit ve değerlendir
    models = get_models()
    results = {}
    
    for model_name, model_info in models.items():
        # Modeli eğit
        model = train_model(X_train, y_train, model_name, 
                          model_info['model'], model_info['params'])
        
        # Modeli kaydet
        save_model(model, vectorizer, model_name.lower().replace(' ', '_'))
        
        # Test seti üzerinde tahminler yap
        y_pred = model.predict(X_test)
        
        # Modeli değerlendir
        report = evaluate_model(y_test, y_pred, model_name)
        results[model_name] = report
        
        # Çapraz doğrulama
        logging.info(f"{model_name} için çapraz doğrulama yapılıyor...")
        cross_validate_model(model, X_train, y_train)
    
    # Modelleri karşılaştır
    logging.info("Modeller karşılaştırılıyor...")
    comparison = compare_models(results)
    print("\nModel Karşılaştırma Tablosu:")
    print(comparison)
    
    # En iyi modeli seç
    best_model = comparison.loc[comparison['F1-Score'].idxmax()]
    print(f"\nEn iyi model: {best_model['Model']}")
    print(f"F1-Skoru: {best_model['F1-Score']:.3f}")
    
    # Örnek tahminler yap
    example_texts = [
        "I love this product! It's amazing!",
        "This is okay, nothing special.",
        "I hate this, it's terrible!"
    ]
    
    # En iyi modeli yükle
    best_model_name = best_model['Model'].lower().replace(' ', '_')
    model, vectorizer = load_model(best_model_name)
    
    print("\nÖrnek Tahminler (En İyi Model):")
    for text in example_texts:
        prediction, probability = predict_sentiment(model, vectorizer, text)
        sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}[prediction]
        print(f"\nMetin: {text}")
        print(f"Tahmin: {sentiment}")
        print(f"Olasılıklar: {probability}")

if __name__ == "__main__":
    main() 