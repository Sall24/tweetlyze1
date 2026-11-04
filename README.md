# Tweetlyze

Aplikasi web analisis sentimen untuk tweet berbahasa Indonesia, dibangun dengan **Streamlit** dan model **CNN** (TensorFlow/Keras). Pengguna dapat menganalisis sentimen dari URL tweet atau dari kalimat yang diketik langsung.

## Fitur

- **Analisis dari URL tweet:** teks tweet diambil otomatis, lalu diprediksi sentimennya
- **Analisis kalimat langsung:** ketik teks bebas atau opini, lalu lihat hasilnya
- **Klasifikasi 3 kelas:** Negative, Neutral, Positive, lengkap dengan confidence tiap kelas
- **Kata Trending dan Sentimen Trending:** saat ini masih data contoh (statis) sebagai ilustrasi tampilan

## Struktur Repository

```
.
├── Data/                # dataset tweet
├── model/               # model CNN dan tokenizer
│   ├── sentiment_cnn.h5
│   └── tokenizer.pkl.gz
├── tweetlyze-app.py     # aplikasi Streamlit
├── requirements.txt     # daftar dependensi
└── README.md
```

## Cara Menjalankan

1. Clone repository:
   ```bash
   git clone https://github.com/Sall24/tweetlyze1.git
   cd tweetlyze1
   ```
2. (Disarankan) Buat virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```
3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi:
   ```bash
   streamlit run tweetlyze-app.py
   ```

## Model dan Pipeline

1. **Dataset:** 1.268 tweet berbahasa Indonesia
2. **Pelabelan:** otomatis (weak labeling) dengan model `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`
3. **Pra-pemrosesan:** case folding, penghapusan URL, mention, hashtag, dan karakter non-alfanumerik, serta stopword removal bahasa Indonesia
4. **Pemodelan:** CNN dengan tokenizer Keras, panjang input maksimal 100 token
5. **Eksplorasi tambahan:** WordCloud per sentimen dan topic modeling (LDA)

### Hasil evaluasi

Dievaluasi pada 522 data uji (seimbang, 174 data per kelas):

| Kelas | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Negative | 0,97 | 0,59 | 0,73 |
| Neutral | 0,89 | 1,00 | 0,94 |
| Positive | 0,76 | 0,97 | 0,86 |
| **Akurasi** | | | **0,85** |
| **Macro avg** | 0,87 | 0,85 | 0,84 |

> Label dihasilkan oleh model lain (bukan anotasi manusia), sehingga hasil evaluasi mengukur kesesuaian dengan label tersebut.

## Catatan

- Pengambilan tweet lewat URL memakai `snscrape`, yang bergantung pada akses publik ke Twitter/X dan bisa gagal jika platform mengubah aksesnya. Fitur analisis kalimat langsung tetap berjalan tanpa itu.
- Pastikan file `tokenizer.pkl.gz` dan `sentiment_cnn.h5` berada di folder `model/` dengan nama persis seperti di atas.

## Teknologi

Python, Streamlit, TensorFlow/Keras, scikit-learn, pandas, NumPy, WordCloud, snscrape

## Author

**Faisal Akbar Ilham Arya Ardhi**
Informatika, Universitas Amikom Yogyakarta
GitHub: [@Sall24](https://github.com/Sall24)
