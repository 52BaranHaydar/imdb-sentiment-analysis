# 🎬 IMDB SENTIMENT ANALYSIS (DUYGU ANALİZİ)

Bu proje; 50.000 film incelemesinden oluşan popüler **IMDB Movie Reviews** veri kümesi üzerinde, **Doğal Dil İşleme (Natural Language Processing - NLP)** ve **Makine Öğrenmesi (Machine Learning)** yöntemlerini kullanarak film yorumlarının duygu durumunu (**Pozitif / Negatif**) yüksek doğrulukla sınıflandıran uçtan uca bir veri bilimi çalışmasıdır.

---

## 📑 İÇİNDEKİLER

1. [Proje Hakkında ve Amaç](#-proje-hakkında-ve-amaç)
2. [Veri Kümesi Analizi](#-veri-kümesi-analizi)
3. [Doğal Dil İşleme (NLP) ve Ön İşleme Adımları](#-doğal-dil-işleme-nlp-ve-ön-i̇şleme-adımları)
4. [Metin Vektörizasyonu: CountVectorizer vs TfidfVectorizer](#-metin-vektörizasyonu-countvectorizer-vs-tfidfvectorizer)
5. [Model Mimarisi ve Scikit-Learn Pipeline](#-model-mimarisi-ve-scikit-learn-pipeline)
6. [Model Değerlendirme ve Performans Metrikleri](#-model-değerlendirme-ve-performans-metrikleri)
7. [Hata Analizi ve Önemli Kelime Katsayıları](#-hata-analizi-ve-önemli-kelime-katsayıları)
8. [Proje Dosya Yapısı](#-proje-dosya-yapısı)
9. [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
10. [Geliştirici](#-geliştirici)

---

## 🎯 PROJE HAKKINDA VE AMAÇ

Kullanıcı yorumları gibi yapılandırılmamış (*unstructured*) metin verileri, işletmeler ve platformlar için büyük değer taşır. Bu projede:
- İnsan dilindeki metinlerin makine öğrenmesi modellerinin anlayabileceği sayısal vektörlere nasıl dönüştürüldüğü gösterilmiştir.
- İki temel vektörizasyon tekniği olan **CountVectorizer** ve **TF-IDF** deneysel olarak karşılaştırılmıştır.
- **Scikit-Learn Pipeline** yapısıyla veri sızıntısı (*data leakage*) riski olmadan temiz ve tekrarlanabilir bir modelleme süreci kurulmuştur.
- Elde edilen modelin başarısı karmaşıklık matrisi (*confusion matrix*) ve metriklerle derinlemesine incelenmiştir.

---

## 📊 VERİ KÜMESİ ANALİZİ

- **Kaynak**: IMDB Dataset of 50K Movie Reviews (`movie.csv`)
- **Örnek Sayısı**: 50.000 adet kullanıcı yorumu
- **Sütunlar**:
  - `text`: Kullanıcıların yazdığı ham film inceleme metni
  - `label`: Yorumun duygu etiketi (`0` / `negative` = Olumsuz, `1` / `positive` = Olumlu)
- **Sınıf Dağılımı**: 25.000 olumlu, 25.000 olumsuz yorum ile tamamen **dengeli (balanced)** bir veri kümesidir. Bu durum sınıf dengesizliği (*class imbalance*) problemlerini önler.

---

## 🛠 DOĞAL DİL İŞLEME (NLP) VE ÖN İŞLEME ADIMLARI

Ham metinler doğrudan modeller tarafından işlenemez. Bu doğrultuda şu adımlar uygulanmıştır:

1. **Küçük Harfe Dönüştürme (Case Lowering)**:
   - "Great", "GREAT" ve "great" kelimelerinin farklı özellikler (*features*) olarak algılanmasını önlemek için tüm metinler küçük harfe dönüştürülmüştür.
2. **Stop Words (Etkisiz Kelimeler) Ayıklama**:
   - İngilizce dilinde sıkça geçen fakat duygu analizine anlam katmayan ("the", "is", "at", "which", "on" vb.) sözcükler filtrelenmiştir.
3. **Noktalama İşaretleri ve Özel Karakter Temizliği**:
   - Gürültü (*noise*) yaratan karakterler elenerek kelime köklerinin netliği korunmuştur.

---

## ⚖️ METİN VEKTÖRİZASYONU: COUNTVECTORIZER VS TFIDFVECTORIZER

Projeyi diğerlerinden ayıran önemli bir adım, iki popüler vektörleştirme tekniğinin kıyaslanmasıdır:

| Özellik | `CountVectorizer` (Bag of Words) | `TfidfVectorizer` (TF-IDF) |
| :--- | :--- | :--- |
| **Çalışma Mantığı** | Kelimenin belgedeki toplam geçiş frekansını sayar. | Terim Sıklığı (TF) ile Ters Belge Sıklığını (IDF) çarpar. |
| **Karakteristiği** | Çok sık geçen kelimelere yüksek ağırlık verir. | Tüm dokümanlarda sık geçen kelimeleri cezalandırır, ayırt edici kelimeleri öne çıkarır. |
| **Avantajı** | Basit ve hızlıdır. | Yüksek bilgi yoğunluğu sağlar, aşırı yaygın kelimelerin yanlılığını engeller. |
| **Model Performansı** | İyi | **Daha yüksek doğruluk ve genelleme kapasitesi** |

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

---

## 🧱 MODEL MİMARİSİ VE SCIKIT-LEARN PIPELINE

Model eğitimi sürecinde `Pipeline` sınıfı kullanılmıştır. Bu yaklaşım:
1. **Veri Sızıntısını (Data Leakage) Önler**: Vektörleştiricinin `fit` işlemi yalnızca eğitim setinde yapılır; test seti ise yalnızca `transform` edilir.
2. **Kodu Modüler Kılar**: Veri hazırlama ve modelleme tek bir nesne altında yönetilir.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", LogisticRegression(max_iter=1000))
])
```

- **Eğitim / Test Bölünmesi**: Veri kümesi %80 Eğitim (`train`), %20 Test (`test`) olarak ayrılmıştır (`test_size=0.2, random_state=42`).

---

## 📈 MODEL DEĞERLENDİRME VE PERFORMANS METRİKLERİ

Test seti üzerindeki tahminler şu metriklerle değerlendirilmiştir:

- **Accuracy (Doğruluk)**: Modelin doğru tahmin ettiği yorumların genel test setine oranı (~%88-%90 aralığında yüksek başarı).
- **Precision (Kesinlik)**: Olumlu/Olumsuz olarak tahmin edilenlerin ne kadarının gerçekten doğru olduğu.
- **Recall (Duyarlılık)**: Gerçek pozitif/negatif sınıfların ne kadarının yakalanabildiği.
- **F1-Score**: Precision ve Recall değerlerinin harmonik ortalaması.
- **Confusion Matrix (Karmaşıklık Matrisi)**:
  - `True Positive (TP)`: Başarıyla tahmin edilen olumlu yorumlar
  - `True Negative (TN)`: Başarıyla tahmin edilen olumsuz yorumlar
  - `False Positive (FP)`: Gerçekte olumsuzken yanlışlıkla olumlu tahmin edilenler
  - `False Negative (FN)`: Gerçekte olumluyken yanlışlıkla olumsuz tahmin edilenler

---

## 🔍 HATA ANALİZİ VE ÖNEMLİ KELİME KATSAYILARI

- **En Etkili Kelimeler**: Modelin ağırlık katsayıları (*coefficients*) incelenerek, olumlu kararı en çok tetikleyen ("excellent", "superb", "brilliant", "great") ve olumsuz kararı en çok tetikleyen ("worst", "waste", "awful", "boring", "terrible") kelimeler tespit edilmiştir.
- **Hatalı Tahminlerin Analizi**: İronik (sarkastik) dil içeren veya uzun süre olumsuz cümleler kurup sonunda "ama yine de güzeldi" şeklinde biten karmaşık cümlelerde modelin nerelerde zorlandığı notebook içerisinde detaylıca incelenmiştir.

---

## 📂 PROJE DOSYA YAPISI

```text
IMDB/
├── datasets/
│   └── movie.csv              # 50.000 satırlık IMDB veri seti (52.7 MB)
├── imdb.ipynb                 # Adım adım veri analizi, karşılaştırmalar ve model kodları
├── .gitignore                 # venv/, geçici dosyalar ve önbellek filtreleri
└── README.md                  # Kapsamlı proje dokümantasyonu
```

---

## 🚀 KURULUM VE ÇALIŞTIRMA

Projeyi bilgisayarınızda çalıştırmak için:

1. **Repoyu klonlayın:**
   ```bash
   git clone https://github.com/52BaranHaydar/imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis
   ```

2. **Python sanal ortamı oluşturun ve aktif edin:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # Windows için: venv\Scripts\activate
   ```

3. **Gereksinimleri yükleyin:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```

4. **Notebook'u başlatın:**
   ```bash
   jupyter notebook imdb.ipynb
   ```

---

## 👤 GELİŞTİRİCİ

- **Baran Haydar**
- GitHub: [@52BaranHaydar](https://github.com/52BaranHaydar)
