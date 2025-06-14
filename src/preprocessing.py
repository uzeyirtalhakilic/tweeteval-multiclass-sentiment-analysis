"""
Metin ön işleme modülü.
Bu modül, duygu analizi için metin verilerini temizleme ve hazırlama işlemlerini içerir.
"""

import re
import nltk
from nltk.corpus import stopwords
import unicodedata
from datasets import load_dataset
import logging

# NLTK'nın gerekli verilerini indir
# Eğer stopwords verisi yoksa indir
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    """
    Metni temizler ve standartlaştırır.
    
    Parametreler:
        text (str): Temizlenecek metin
        
    Dönüş:
        str: Temizlenmiş metin
        
    Yapılan işlemler:
    1. Küçük harfe çevirme
    2. URL'leri kaldırma
    3. Kullanıcı adlarını (@username) kaldırma
    4. Özel karakterleri kaldırma (Türkçe karakterler hariç)
    5. Türkçe karakterleri normalize etme
    6. Fazla boşlukları temizleme
    """
    # Küçük harfe çevir
    text = text.lower()
    
    # URL'leri kaldır (http:// veya https:// ile başlayan)
    text = re.sub(r"http\S+", "", text)
    
    # Kullanıcı adlarını kaldır (@ ile başlayan)
    text = re.sub(r"@\w+", "", text)
    
    # Özel karakterleri kaldır (Türkçe karakterler hariç)
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]", "", text)
    
    # Türkçe karakterleri normalize et
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    
    # Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def remove_stopwords(text, language='turkish'):
    """
    Metinden gereksiz kelimeleri (stopwords) kaldırır.
    
    Parametreler:
        text (str): İşlenecek metin
        language (str): Dil seçeneği (varsayılan: 'turkish')
        
    Dönüş:
        str: Gereksiz kelimeleri kaldırılmış metin
        
    Not:
        - Stopwords: Bir dilde sık kullanılan ama anlam taşımayan kelimeler
        - Örnek: "ve", "veya", "ile", "için" gibi
    """
    # Seçilen dilin stopwords listesini al
    stop_words = set(stopwords.words(language))
    
    # Metni kelimelere ayır
    words = text.split()
    
    # Stopwords olmayan kelimeleri seç
    filtered_words = [word for word in words if word not in stop_words]
    
    # Kelimeleri tekrar birleştir
    return ' '.join(filtered_words)

# Veri setini yükle
dataset = load_dataset("tweet_eval", "sentiment")

# Eğitim ve test verilerini al
train_data = dataset['train']
test_data = dataset['test']

def convert_labels(df):
    """
    Sayısal etiketleri metin etiketlerine dönüştürür.
    
    Parametreler:
        df: Etiketleri dönüştürülecek veri çerçevesi
        
    Dönüş:
        df: Etiketleri dönüştürülmüş veri çerçevesi
        
    Not:
        - 0: negative
        - 1: neutral
        - 2: positive
    """
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df['sentiment_text'] = df['label'].map(label_map)
    return df

def prepare_data_for_training(df):
    """
    Veri setini model eğitimi için hazırlar.
    
    Parametreler:
        df: İşlenecek veri çerçevesi
        
    Dönüş:
        X: Özellikler (metinler)
        y: Etiketler
    """
    X = df['text']
    y = df['sentiment_text']
    return X, y
