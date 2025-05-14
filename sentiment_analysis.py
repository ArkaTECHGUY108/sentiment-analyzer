import pandas as pd
import re
import nltk
import emoji
import contractions
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



# Deep learning imports
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv("sentiment-env/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
df.columns = ["target", "ids", "date", "flag", "user", "text"]

# Use 50,000 sample tweets
df = df[['target', 'text']]
df = df.sample(n=50000, random_state=42).reset_index(drop=True)

# Convert 4 → 1 (positive), keep 0 as negative
df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)

# Text Preprocessing Setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = contractions.fix(text)                             # Expand contractions
    text = emoji.replace_emoji(text, replace='')              # Remove emojis
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)       # Remove URLs
    text = re.sub(r"@\w+|#\w+", '', text)                     # Remove mentions & hashtags
    text = re.sub(r"[^a-zA-Z\s]", '', text)                   # Remove non-letter chars
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Tokenization and padding
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['clean_text'])
X_seq = tokenizer.texts_to_sequences(df['clean_text'])
X_pad = pad_sequences(X_seq, maxlen=100)
y = df['target'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
print("\n📈 Training LSTM model...\n")
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy}")

# Save model and tokenizer
model.save("sentiment_lstm_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("\n✅ Model and tokenizer saved successfully!")
