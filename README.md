# 🎬 IMDB Sentiment Analysis (Duygu Analizi)

Bu proje, **IMDB Film Yorumları** veri seti üzerinde Doğal Dil İşleme (NLP) ve Makine Öğrenmesi teknikleri kullanılarak film yorumlarının olumlu (positive) ya da olumsuz (negative) olarak sınıflandırılmasını amaçlar.

---

## 📌 Proje Özeti

- **Metin Ön İşleme**: Küçük harfe dönüştürme, noktalama işaretleri ve İngilizce etkisiz kelimelerin (*stop words*) ayıklanması.
- **Vektörizasyon Karşılaştırması**:
  - `CountVectorizer` (Kelimelerin frekansına dayalı yaklaşım)
  - `TfidfVectorizer` (Kelimelerin metindeki önem derecesine dayalı yaklaşım)
- **Modelleme**: Scikit-Learn `Pipeline` mimarisi ile vektörizasyon ve sınıflandırma modellerinin entegrasyonu.
- **Değerlendirme**: Accuracy, Precision, Recall, F1-Score metrikleri ve Karmaşıklık Matrisi (*Confusion Matrix*) görselleştirmesi.
- **Hata Analizi**: Modelin yanlış tahmin ettiği örneklerin incelenmesi ve en etkili kelime katsayılarının analizi.

---

## 📂 Proje Yapısı

```text
├── datasets/
│   └── movie.csv          # IMDB film yorumları veri seti (50K yorum)
├── imdb.ipynb             # Veri analizi, modelleme ve değerlendirme notebook'u
├── .gitignore             # Git tarafından yoksayılan dosyalar
└── README.md              # Proje dökümantasyonu
```

---

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

1. **Repoyu klonlayın:**
   ```bash
   git clone https://github.com/52BaranHaydar/imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis
   ```

2. **Sanal ortam oluşturun ve aktif edin (Opsiyonel):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows için: venv\Scripts\activate
   ```

3. **Gerekli kütüphaneleri yükleyin:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```

4. **Jupyter Notebook'u başlatın:**
   ```bash
   jupyter notebook imdb.ipynb
   ```

---

## 📊 Kullanılan Teknolojiler

- **Python 3**
- **Pandas & NumPy** (Veri manipülasyonu)
- **Scikit-Learn** (Doğal Dil İşleme ve Makine Öğrenmesi)
- **Matplotlib & Seaborn** (Veri Görselleştirme)
- **Jupyter Notebook** (Etkileşimli geliştirme ortamı)

---

## 👤 Geliştirici

- **Baran Haydar** - [GitHub Profili](https://github.com/52BaranHaydar)
