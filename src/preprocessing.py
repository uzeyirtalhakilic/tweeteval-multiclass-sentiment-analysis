"""
Metin ön işleme modülü.
Bu modül, duygu analizi için metin verilerini temizleme ve hazırlama işlemlerini içerir.
"""

import re
import nltk
from nltk.corpus import stopwords
import unicodedata

# NLTK'nın gerekli verilerini indir
# Eğer stopwords verisi yoksa indir
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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
