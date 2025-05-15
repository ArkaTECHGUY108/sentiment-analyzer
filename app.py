from flask import Flask, render_template, request
import tensorflow as tf
import pickle
import os
import gdown
from tensorflow.keras.preprocessing.sequence import pad_sequences
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------- Google Drive IDs ---------------- #
file_ids = {
    "goemotions_model": "10g9HcBEAMjjyW8P8aAnmMSIK5RA2IGMS",
    "goemotions_tokenizer": "1_b6vFEZ_kOMD9O4fpGXZSH6SjTBObX3h",
    "goemotions_mlb": "1bHJXYY_QNiLwF0yry9fSM2PDrb6HaEM1",
    "sentiment_model": "1Hb3o7UQe4oaWAaPLa9sO_0jhdgUFAVtO",
    "tokenizer": "1Kzd_2GWB0MpwzuiUhXH9cyRllmGrwiR0"
}

# ---------------- File Paths ---------------- #
file_paths = {
    "goemotions_model": "goemotions_lstm_model.h5",
    "goemotions_tokenizer": "goemotions_tokenizer.pkl",
    "goemotions_mlb": "goemotions_mlb.pkl",
    "sentiment_model": "sentiment_lstm_model.h5",
    "tokenizer": "tokenizer.pkl"
}

# ---------------- Download if Missing ---------------- #
for key, path in file_paths.items():
    if not os.path.exists(path):
        print(f"📥 Downloading {path}...")
        gdown.download(f"https://drive.google.com/uc?export=download&id={file_ids[key]}", path, quiet=False)

# ---------------- Load Models ---------------- #
sentiment_model = tf.keras.models.load_model(file_paths["sentiment_model"])
goemotions_model = tf.keras.models.load_model(file_paths["goemotions_model"])

with open(file_paths["tokenizer"], "rb") as f:
    tokenizer = pickle.load(f)

with open(file_paths["goemotions_tokenizer"], "rb") as f:
    goemotions_tokenizer = pickle.load(f)

with open(file_paths["goemotions_mlb"], "rb") as f:
    goemotions_mlb = pickle.load(f)

# ---------------- Flask App ---------------- #
app = Flask(__name__, static_folder='static')

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = ""
    tone = ""
    emotion_list = []

    if request.method == "POST":
        text = request.form["text"]

        # 1. Tone Analysis
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)

        if scores['compound'] >= 0.5:
            tone = "Happy 😊"
        elif scores['compound'] <= -0.5:
            tone = "Angry 😠"
        elif -0.5 < scores['compound'] < 0:
            tone = "Sad 😢"
        elif 0 < scores['compound'] < 0.5:
            tone = "Calm 🙂"
        else:
            tone = "Neutral 😐"

        # 2. Sentiment Classification (Binary)
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        pred = sentiment_model.predict(padded)[0][0]
        sentiment = "Positive ✅" if pred > 0.5 else "Negative ❌"

        # 3. GoEmotions (Multi-label)
        seq2 = goemotions_tokenizer.texts_to_sequences([text])
        pad2 = pad_sequences(seq2, maxlen=100)
        emotion_probs = goemotions_model.predict(pad2)[0]
        top_indices = emotion_probs.argsort()[-3:][::-1]
        top_scores = emotion_probs[top_indices]

        emotions = goemotions_mlb.classes_
        emojis = ["😄", "😢", "😡", "😱", "🤔", "😎", "❤️", "😂", "🙄", "😴", "😬", "😇", "😨", "😍", "😭", "😳", "😤", "😷", "😈", "💔", "💀", "👀", "🎉", "👍", "🙏", "🔥", "🌟"]

        emotion_list = [(emotions[i], emojis[i % len(emojis)], float(top_scores[j])) for j, i in enumerate(top_indices)]

    return render_template("index.html", sentiment=sentiment, tone=tone, emotion_list=emotion_list)

if __name__ == "__main__":
    app.run(debug=True)
