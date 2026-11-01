import os
import certifi
import streamlit as st
import tensorflow as tf
import numpy as np
import gzip
import pickle
import re
import snscrape.modules.twitter as sntwitter
from wordcloud import WordCloud
from collections import Counter

# ========== Konfigurasi SSL (untuk Windows + snscrape) ==========
os.environ['SSL_CERT_FILE'] = certifi.where()

# ========== Load Model & Tokenizer ==========
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/sentiment_cnn.h5")

@st.cache_data
def load_tokenizer():
    tokenizer_path = 'model/tokenizer.pkl.gz'
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer tidak ditemukan di: {tokenizer_path}")
    with gzip.open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    if not hasattr(tokenizer, 'texts_to_sequences'):
        raise ValueError("File tokenizer tidak valid.")
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()
MAX_LEN = 100

# ========== Preprocessing ==========
stopwords_id = set([
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'untuk', 'dengan', 'pada', 'ada',
    'karena', 'sudah', 'saja', 'juga', 'mereka', 'kami', 'kita', 'saya', 'aku',
    'tidak', 'bukan', 'apa', 'mengapa', 'bagaimana', 'dalam', 'itu', 'bahwa',
    'oleh', 'sebagai', 'akan', 'atau', 'jadi', 'lagi', 'hanya', 'lebih',
    'kalau', 'ya', 'oh', 'nah', 'kok', 'lah', 'yg', 'gw', 'gua', 'gue', 'orang', 'lu', 'ga', 'kau'
])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@[\w]+', '', text)
    text = re.sub(r'\#[\w]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([w for w in text.split() if w not in stopwords_id])
    return text

# ========== Prediksi Sentimen ==========
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0]

    labels = ["Negative", "Neutral", "Positive"]
    sentiment = labels[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    return sentiment, confidence, prediction

# ========== Ambil Teks dari URL Tweet ==========
def get_tweet_text_from_url(url):
    try:
        tweet_id = url.strip().split("/")[-1]
        scraper = sntwitter.TwitterTweetScraper(tweet_id)
        tweet = next(scraper.get_items())
        return tweet.content
    except Exception as e:
        return f"[ERROR] Gagal ambil tweet: {e}"

# ========== UI Streamlit ==========
st.set_page_config(page_title="Tweetlyze", layout="wide")
st.title("\U0001F4AC Tweetlyze - Analisis Sentimen Twitter")

col_input, col_result = st.columns([2, 3])

with col_input:
    st.subheader("\U0001F517 Masukan URL Tweet")
    tweet_url = st.text_input("https://twitter.com/...")
    if st.button("\U0001F50D Analisis Sentimen"):
        if tweet_url:
            tweet_text = get_tweet_text_from_url(tweet_url)
            if "[ERROR]" in tweet_text:
                st.error(tweet_text)
            else:
                cleaned = clean_text(tweet_text)
                sentiment, confidence, probas = predict_sentiment(cleaned)
                st.session_state['text'] = tweet_text
                st.session_state['sentiment'] = sentiment
                st.session_state['confidence'] = confidence
                st.session_state['probas'] = probas
        else:
            st.warning("Masukkan URL Tweet terlebih dahulu.")

with col_result:
    st.subheader("\U0001F4CA Hasil Analisis")
    if 'sentiment' in st.session_state:
        st.write(f"\"{st.session_state['text']}\"")
        st.markdown(f"**Sentimen:** `{st.session_state['sentiment']}`")
        st.progress(int(st.session_state['confidence']))

        st.write("\n**Confidence Detail:**")
        for label, p in zip(["Negative", "Neutral", "Positive"], st.session_state['probas']):
            st.write(f"- {label}: {p:.2%}")
    else:
        st.info("Masukkan URL terlebih dahulu untuk mendapatkan hasil analisis.")

# ========== Analisis Kalimat Langsung ==========
st.markdown("---")
st.subheader("\u270D\ufe0f Analisis Kalimat Langsung")
user_input = st.text_area("Masukkan teks bebas atau opini:")

if st.button("\U0001F50D Analisis Kalimat"):
    if user_input:
        cleaned_input = clean_text(user_input)
        sentiment, confidence, probas = predict_sentiment(cleaned_input)
        st.success(f"Sentimen: {sentiment} ({confidence:.2f}%)")
        st.write("\n**Confidence Detail:**")
        for label, p in zip(["Negative", "Neutral", "Positive"], probas):
            st.write(f"- {label}: {p:.2%}")
    else:
        st.warning("Tolong masukkan kalimat terlebih dahulu.")

# ========== Kata Trending Dummy (Statik) ==========
st.markdown("---")
col_words, col_trend = st.columns(2)

with col_words:
    st.subheader("\U0001F4CC Kata Trending")
    trending_words = ["Indonesia Gelap", "Pendidikan", "Harga", "Listrik", "Gaji", "Pemerintah", "PHK", "Demo", "Ekonomi", "Dolar"]
    for i, word in enumerate(trending_words, 1):
        st.write(f"{i}. {word}")

with col_trend:
    st.subheader("\U0001F4C8 Sentimen Trending")
    st.line_chart({
        "Positive": [20, 35, 40, 55, 70, 85],
        "Negative": [30, 25, 20, 15, 10, 5],
        "Neutral":  [50, 40, 40, 30, 20, 10]
    })
