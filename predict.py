"""
Tweet duygu analizi için tahmin scripti.
"""

from src.model import load_model, predict_sentiment

def get_user_input():
    """
    Kullanıcıdan metin alır.
    
    Returns:
        str: Kullanıcının girdiği metin
    """
    print("\nTweet duygu analizi için metin girin (çıkmak için 'q' yazın):")
    return input("> ")

def main():
    # Modeli yükle (eğitilmiş modellerden birini seçin)
    model_name = "logistic_regression"  # veya "svm", "random_forest", "naive_bayes", "neural_network"
    print(f"\n{model_name} modeli yükleniyor...")
    model, vectorizer = load_model(model_name)
    print("Model yüklendi!")
    
    while True:
        # Kullanıcıdan metin al
        text = get_user_input()
        
        # Çıkış kontrolü
        if text.lower() == 'q':
            print("\nProgram sonlandırılıyor...")
            break
        
        # Boş metin kontrolü
        if not text.strip():
            print("Lütfen bir metin girin!")
            continue
        
        # Tahmin yap
        prediction, probability = predict_sentiment(model, vectorizer, text)
        sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}[prediction]
        
        # Sonuçları yazdır
        print("\nSonuçlar:")
        print(f"Metin: {text}")
        print(f"Tahmin: {sentiment}")
        print(f"Olasılıklar:")
        print(f"- Negatif: {probability[0]:.2%}")
        print(f"- Nötr: {probability[1]:.2%}")
        print(f"- Pozitif: {probability[2]:.2%}")

if __name__ == "__main__":
    main() 