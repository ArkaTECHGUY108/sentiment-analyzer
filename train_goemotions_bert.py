from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import DistilBertTokenizerFast
import pandas as pd

# Load GoEmotions dataset from HuggingFace
dataset = load_dataset("go_emotions")

# Use only train split for now
data = dataset['train']
df = pd.DataFrame(data)

# Load emotion labels list
emotion_labels = dataset['train'].features['labels'].feature.names

# Convert multi-labels to binary matrix
mlb = MultiLabelBinarizer(classes=range(len(emotion_labels)))
y = mlb.fit_transform(df['labels'])

# Save labels mapping
label_map = {i: emotion_labels[i] for i in range(len(emotion_labels))}
pd.DataFrame.from_dict(label_map, orient='index').to_csv("label_map.csv", header=['emotion'])

# Load BERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(df['text'].tolist(), truncation=True, padding=True)

import torch
from torch.utils.data import Dataset
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

# Create PyTorch dataset
class GoEmotionsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# Prepare dataset
dataset = GoEmotionsDataset(encodings, y)

# Load BERT model for multi-label classification
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=y.shape[1], problem_type="multi_label_classification")

# Training config
training_args = TrainingArguments(
    output_dir='./goemotions_bert',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir='./logs',
)


# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Start training
print("🔥 Training BERT on GoEmotions...")
trainer.train()

# Save the final model and tokenizer
model.save_pretrained("./goemotions_model")
tokenizer.save_pretrained("./goemotions_model")

print("✅ Model saved to goemotions_model/")


print(f"Sample encoded text:\n{df['text'][0]}\n{encodings['input_ids'][0]}")
print(f"Sample labels: {y[0]}")
