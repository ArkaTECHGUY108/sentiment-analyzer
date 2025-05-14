from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# Load Positive/Negative LSTM model
model = tf.keras.models.load_model("sentiment_lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load Multi-label Emotion LSTM model
emotion_model = tf.keras.models.load_model("goemotions_lstm_model.h5")
with open("goemotions_tokenizer.pkl", "rb") as f:
    emotion_tokenizer = pickle.load(f)
with open("goemotions_mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

# Emoji map
emoji_map = {
    "admiration": "😍", "amusement": "😂", "anger": "😠", "annoyance": "😒", "approval": "👍",
    "caring": "🤗", "confusion": "😕", "curiosity": "🤔", "desire": "😋", "disappointment": "😞",
    "disapproval": "👎", "disgust": "🤮", "embarrassment": "😳", "excitement": "😃", "fear": "😨",
    "gratitude": "🙏", "grief": "😭", "joy": "😊", "love": "❤️", "nervousness": "😬",
    "optimism": "🌞", "pride": "😎", "realization": "💡", "relief": "😌", "remorse": "😔",
    "sadness": "😢", "surprise": "😮", "neutral": "😐"
}

app = Flask(__name__, static_folder='static')

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = ""
    tone = ""
    emotion_list = []

    if request.method == "POST":
        text = request.form["text"]

        # VADER Tone Detection
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

        # Binary Sentiment (Positive/Negative)
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        pred = model.predict(padded)[0][0]
        sentiment = "Positive ✅" if pred > 0.5 else "Negative ❌"

        # Multi-label Emotion Detection
        emo_seq = emotion_tokenizer.texts_to_sequences([text])
        emo_pad = pad_sequences(emo_seq, maxlen=100)
        emo_pred = emotion_model.predict(emo_pad)[0]

        # Get top 3 emotions
        top_indices = np.argsort(emo_pred)[::-1][:3]
        # Load class map manually (GoEmotions strings)
        with open("google-research/goemotions/data/emotions.txt", "r") as f:
            emotion_labels = f.read().splitlines()

        emotion_list = [(emotion_labels[i], emoji_map.get(emotion_labels[i], "❓"), float(emo_pred[i])) for i in top_indices]


    return render_template("index.html", sentiment=sentiment, tone=tone, emotion_list=emotion_list)

if __name__ == "__main__":
    app.run(debug=True)
