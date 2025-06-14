"""
Duygu analizi modeli modülü.
Bu modül, metin sınıflandırma modelinin oluşturulması, eğitilmesi ve kullanılması için gerekli fonksiyonları içerir.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def train_logistic_regression(X_train, y_train):
    """
    Lojistik regresyon modelini eğitir.
    
    Parametreler:
        X_train: Eğitim verileri (metin)
        y_train: Eğitim etiketleri (duygu sınıfları)
        
    Dönüş:
        tuple: (eğitilmiş model, vektörizer)
        
    Not:
        - TF-IDF vektörizer: Metinleri sayısal özelliklere dönüştürür
        - Lojistik regresyon: Çok sınıflı sınıflandırma için kullanılır
    """
    # TF-IDF vektörizer oluştur (en fazla 5000 özellik)
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Metinleri vektörlere dönüştür
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Çok sınıflı lojistik regresyon modeli oluştur
    model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    
    # Modeli eğit
    model.fit(X_train_vec, y_train)
    
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    """
    Verilen metnin duygu durumunu tahmin eder.
    
    Parametreler:
        text (str): Tahmin yapılacak metin
        model: Eğitilmiş model
        vectorizer: Eğitilmiş vektörizer
        
    Dönüş:
        tuple: (tahmin edilen sınıf, sınıf olasılıkları)
    """
    # Metni vektöre dönüştür
    text_vec = vectorizer.transform([text])
    
    # Tahmin yap
    prediction = model.predict(text_vec)
    
    # Sınıf olasılıklarını hesapla
    probability = model.predict_proba(text_vec)
    
    return prediction[0], probability[0]

def save_model(model, vectorizer, model_path='models'):
    """
    Eğitilmiş modeli ve vektörizeri kaydeder.
    
    Parametreler:
        model: Kaydedilecek model
        vectorizer: Kaydedilecek vektörizer
        model_path (str): Kayıt dizini (varsayılan: 'models')
    """
    # Kayıt dizinini oluştur (yoksa)
    os.makedirs(model_path, exist_ok=True)
    
    # Modeli kaydet
    joblib.dump(model, os.path.join(model_path, 'sentiment_model.joblib'))
    
    # Vektörizeri kaydet
    joblib.dump(vectorizer, os.path.join(model_path, 'vectorizer.joblib'))

def load_model(model_path='models'):
    """
    Kaydedilmiş modeli ve vektörizeri yükler.
    
    Parametreler:
        model_path (str): Model dizini (varsayılan: 'models')
        
    Dönüş:
        tuple: (yüklenen model, yüklenen vektörizer)
    """
    # Modeli yükle
    model = joblib.load(os.path.join(model_path, 'sentiment_model.joblib'))
    
    # Vektörizeri yükle
    vectorizer = joblib.load(os.path.join(model_path, 'vectorizer.joblib'))
    
    return model, vectorizer
