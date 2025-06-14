"""
Tweet_eval veri seti için duygu analizi modelleri.
"""

import logging
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_models():
    """
    Kullanılacak modelleri ve hiperparametrelerini döndürür.
    
    Returns:
        Modeller ve hiperparametreleri
    """
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(),
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs'],
                'max_iter': [1000]
            }
        },
        'SVM': {
            'model': LinearSVC(dual=False),
            'params': {
                'C': [0.1, 1, 10],
                'max_iter': [1000]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'Naive Bayes': {
            'model': MultinomialNB(),
            'params': {
                'alpha': [0.1, 1.0, 10.0]
            }
        },
        'Neural Network': {
            'model': MLPClassifier(),
            'params': {
                'hidden_layer_sizes': [(100,), (100, 50)],
                'activation': ['relu'],
                'alpha': [0.0001, 0.001]
            }
        }
    }
    return models

def train_model(X_train, y_train, model_name, model, params):
    """
    Belirtilen modeli eğitir ve en iyi hiperparametreleri bulur.
    
    Args:
        X_train: Eğitim verileri
        y_train: Eğitim etiketleri
        model_name: Model adı
        model: Model nesnesi
        params: Hiperparametreler
        
    Returns:
        En iyi model
    """
    logging.info(f"{model_name} modeli eğitiliyor...")
    
    # Grid arama ile en iyi parametreleri bul
    grid_search = GridSearchCV(
        model, params, cv=3, n_jobs=-1, 
        scoring='accuracy', verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    logging.info(f"En iyi parametreler: {grid_search.best_params_}")
    logging.info(f"En iyi skor: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

def predict_sentiment(model, vectorizer, text):
    """
    Verilen metin için duygu tahmini yapar.
    
    Args:
        model: Eğitilmiş model
        vectorizer: TF-IDF vektörleştirici
        text: Tahmin yapılacak metin
        
    Returns:
        Tahmin edilen duygu (0: negatif, 1: nötr, 2: pozitif)
        Duygu olasılıkları
    """
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    
    # LinearSVC için olasılık hesaplama
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(text_tfidf)[0]
    else:
        # LinearSVC için olasılık hesaplama
        decision_function = model.decision_function(text_tfidf)
        probability = decision_function[0]
        # Min-max normalizasyonu
        probability = (probability - probability.min()) / (probability.max() - probability.min())
    
    return prediction, probability

def save_model(model, vectorizer, model_name):
    """
    Modeli ve vektörleştiriciyi kaydeder.
    
    Args:
        model: Kaydedilecek model
        vectorizer: Kaydedilecek vektörleştirici
        model_name: Model dosyasının adı
    """
    # Models dizinini oluştur
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Model ve vektörleştiriciyi kaydet
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    vectorizer_path = os.path.join(models_dir, f"{model_name}_vectorizer.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    logging.info(f"Model kaydedildi: {model_path}")
    logging.info(f"Vektörleştirici kaydedildi: {vectorizer_path}")

def load_model(model_name):
    """
    Kaydedilmiş modeli ve vektörleştiriciyi yükler.
    
    Args:
        model_name: Model dosyasının adı
        
    Returns:
        Yüklenen model ve vektörleştirici
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    vectorizer_path = os.path.join(models_dir, f"{model_name}_vectorizer.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model veya vektörleştirici dosyası bulunamadı.")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    logging.info(f"Model yüklendi: {model_path}")
    logging.info(f"Vektörleştirici yüklendi: {vectorizer_path}")
    
    return model, vectorizer
