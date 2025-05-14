import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import re
import pickle
import contractions
import emoji

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load GoEmotions data (train + dev)
train_df = pd.read_csv("google-research/goemotions/data/train.tsv", sep="\t", header=None, names=["text", "labels", "ids"])
dev_df = pd.read_csv("google-research/goemotions/data/dev.tsv", sep="\t", header=None, names=["text", "labels", "ids"])
df = pd.concat([train_df, dev_df], ignore_index=True)

# Convert label string → list of int
df['labels'] = df['labels'].apply(lambda x: list(map(int, x.split(','))))

# Load emotion labels
with open("google-research/goemotions/data/emotions.txt", "r") as f:
    emotion_labels = f.read().splitlines()

# Clean text
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = contractions.fix(text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(w) for w in text if w not in stop_words]
    return " ".join(text)

df['clean_text'] = df['text'].apply(clean_text)

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
padded = pad_sequences(sequences, maxlen=100)

# Multi-label binarization
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['labels'])

# Prepare train/test data
X = padded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.4))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(y.shape[1], activation='sigmoid'))  # multi-label output

# Use binary cross-entropy for multi-label
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
print("🚀 Training Multi-Label LSTM on GoEmotions...")
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1)
print("✅ LSTM model trained!")

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {accuracy:.4f}")

# Save model + tokenizer + encoder
model.save("goemotions_lstm_model.h5")
with open("goemotions_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("goemotions_mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)

print("✅ Model, tokenizer, and multilabel binarizer saved!")


from sklearn.metrics import f1_score, classification_report

# Predict on test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Calculate F1 score
f1_micro = f1_score(y_test, y_pred_binary, average='micro')
f1_macro = f1_score(y_test, y_pred_binary, average='macro')

print(f"\n🎯 Micro F1-score: {f1_micro:.4f}")
print(f"🎯 Macro F1-score: {f1_macro:.4f}")

optimal_thresh = 0.4  # instead of 0.5
y_pred_binary = (y_pred >= optimal_thresh).astype(int)
