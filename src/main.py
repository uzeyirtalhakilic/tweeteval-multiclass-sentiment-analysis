"""
Duygu analizi ana programı.
Bu program, duygu analizi modelinin eğitilmesi ve değerlendirilmesi için gerekli adımları içerir.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import clean_text, remove_stopwords
from model import train_logistic_regression, save_model, predict_sentiment
from evaluation import evaluate_model, cross_validate_model

def convert_labels(df):
    """
    Sayısal etiketleri metin etiketlerine dönüştürür.
    
    Parametreler:
        df: Etiketleri dönüştürülecek veri çerçevesi
        
    Dönüş:
        df: Etiketleri dönüştürülmüş veri çerçevesi
        
    Not:
        0 -> negative
        2 -> neutral
        4 -> positive
    """
    # Sayısal etiketleri metin etiketlerine dönüştür
    label_map = {0: 'negative', 2: 'neutral', 4: 'positive'}
    df['sentiment_text'] = df['sentiment'].map(label_map)
    return df

def load_data():
    """
    Eğitim veri setini yükler.
    
    Dönüş:
        df: Eğitim veri seti
        
    Not:
        - Veri seti CSV formatında olmalıdır
        - Sütunlar: sentiment, id, date, query, user, text
    """
    # Eğitim veri setini yükle
    df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', 
                     encoding='latin-1',
                     names=['sentiment', 'id', 'date', 'query', 'user', 'text'])
    
    # Etiketleri dönüştür
    df = convert_labels(df)
    
    return df

def preprocess_data(df):
    """
    Veri setini ön işleme adımlarından geçirir.
    
    Parametreler:
        df: İşlenecek veri çerçevesi
        
    Dönüş:
        df: İşlenmiş veri çerçevesi
        
    Yapılan işlemler:
        1. Metin temizleme
        2. Gereksiz kelimeleri kaldırma
    """
    # Metinleri temizle
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Gereksiz kelimeleri kaldır
    df['processed_text'] = df['cleaned_text'].apply(remove_stopwords)
    
    return df

def main():
    """
    Ana program akışı.
    
    Adımlar:
        1. Veri yükleme
        2. Veri ön işleme
        3. Model eğitimi
        4. Model değerlendirme
        5. Örnek tahminler
    """
    # Veri setini yükle
    print("Veri seti yükleniyor...")
    df = load_data()
    
    # Verileri ön işle
    print("Veriler ön işleniyor...")
    df = preprocess_data(df)
    
    # Veriyi eğitim ve test setlerine böl
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['sentiment_text'],
        test_size=0.2,
        random_state=42
    )
    
    # Modeli eğit
    print("Model eğitiliyor...")
    model, vectorizer = train_logistic_regression(X_train, y_train)
    
    # Modeli kaydet
    print("Model kaydediliyor...")
    save_model(model, vectorizer)
    
    # Test seti üzerinde değerlendir
    print("\nTest seti üzerinde değerlendirme yapılıyor...")
    y_test_pred = model.predict(vectorizer.transform(X_test))
    evaluate_model(y_test, y_test_pred, labels=['negative', 'neutral', 'positive'])
    
    # Çapraz doğrulama yap
    print("\nÇapraz doğrulama yapılıyor...")
    cross_validate_model(model, vectorizer.transform(X_train), y_train)
    
    # Örnek tahminler
    print("\nÖrnek tahminler:")
    example_texts = [
        "I love this product! It's amazing!",
        "This is okay, nothing special.",
        "I hate this product, it's terrible!"
    ]
    
    for text in example_texts:
        cleaned_text = clean_text(text)
        processed_text = remove_stopwords(cleaned_text)
        prediction, probability = predict_sentiment(processed_text, model, vectorizer)
        print(f"\nMetin: {text}")
        print(f"Tahmin: {prediction}")
        print(f"Güven: {max(probability):.2f}")

if __name__ == "__main__":
    main() 