# 🧠 Sentiment Analysis using BERT

<p align="center">
  <img src="https://img.shields.io/badge/Model-BERT-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-Sentiment140-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-Binary%20Sentiment-green?style=for-the-badge" />
</p>

<p align="center">
  A sentiment classification system built on <b>BERT (bert-base-uncased)</b>, fine-tuned on the <b>Sentiment140</b> dataset of 1.6 million tweets for binary positive/negative classification.
</p>

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Training on Google Colab](#-training-on-google-colab)
- [Run Locally](#-run-locally)
- [Requirements](#-requirements)
- [Authors](#-authors)

---

## 🔍 Overview

This project fine-tunes **BERT (bert-base-uncased)** for Twitter sentiment classification using the Sentiment140 dataset. BERT's powerful contextual embeddings are leveraged to classify tweets as positive or negative with high accuracy.

This approach is:
- ✅ **Simple and effective** — BERT handles context naturally
- ✅ **Lightweight** — only 440MB, runs on free Colab GPU
- ✅ **Fast to train** — converges in 3 epochs
- ✅ **High accuracy** — strong baseline for sentiment tasks

---

## 🏗 Model Architecture

Input Text
↓
BERT Tokenizer
↓
BERT (bert-base-uncased)
↓
[CLS] Token Representation
↓
Linear Classifier
↓
Softmax → Positive / Negative

**Key design choices:**

| Component | Detail |
|---|---|
| Backbone | `bert-base-uncased` |
| Classifier | Linear → Softmax |
| Loss function | Cross Entropy Loss |
| Optimizer | Adam |
| Batch size | 8 |
| Epochs | 3 |

---

## 📊 Results

| Model | Accuracy |
|---|---|
| BERT (Ours) | XX% |

### 📈 Visualizations

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### Training Loss
![Loss Graph](images/loss_graph.png)

### Accuracy Graph
![Accuracy Graph](images/accuracy_graph.png)

---

## 📁 Dataset

We use the **Sentiment140** dataset — 1.6 million tweets labeled as:
- `0` → Negative
- `4` → Positive (remapped to `1` in our pipeline)

**Download:** [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## 📂 Project Structure

sentiment-bert/
│
├── images/
│   ├── confusion_matrix.png
│   ├── loss_graph.png
│   └── accuracy_graph.png
│
├── sentiment140.csv
├── train.ipynb
├── predict.py
├── sentiment_model_bert.pth
├── .gitignore
└── README.md

---

## 🚀 Training on Google Colab

**Step 1 — Enable GPU**

**Step 2 — Install dependencies**
```python
!pip install transformers datasets torch scikit-learn matplotlib
```

**Step 3 — Load the dataset**
```python
import pandas as pd
from datasets import Dataset

df = pd.read_csv(
    "/content/sentiment140.csv",
    encoding='latin-1',
    header=None,
    names=['label', 'id', 'date', 'query', 'user', 'text']
)
df['label'] = df['label'].map({0: 0, 4: 1})
df = df[['text', 'label']]
dataset = Dataset.from_pandas(df)

train_data = dataset.shuffle(seed=42).select(range(20000))
test_data = dataset.shuffle(seed=42).select(range(20000, 25000))
```

**Step 4 — Load BERT**
```python
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)
```

**Step 5 — Save model to Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
torch.save(model.state_dict(), "/content/drive/MyDrive/sentiment_model_bert.pth")
print("Model saved!")
```

**Step 6 — Save visualizations**
```python
import matplotlib.pyplot as plt

plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.savefig("loss_graph.png", dpi=150, bbox_inches='tight')

from google.colab import files
files.download("confusion_matrix.png")
files.download("loss_graph.png")
```

---

## 💻 Run Locally

**Step 1 — Clone the repo**
```bash
git clone https://github.com/vinu4545/BERT_Sentiment.git
cd sentiment-bert
```

**Step 2 — Install requirements**
```bash
pip install torch transformers scikit-learn matplotlib
```

**Step 3 — Run predictions**
```python
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)

model = SentimentModel(base_model)
model.load_state_dict(torch.load("sentiment_model_bert.pth", map_location="cpu"))
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    _, pred = torch.max(outputs, dim=1)
    return "Positive" if pred.item() == 1 else "Negative"

print(predict("I love this product!"))
print(predict("This is the worst thing ever"))
```

> ✅ BERT is only **440MB** — runs easily on most local machines.

---

## 📦 Requirements

```bash
pip install torch transformers datasets pandas scikit-learn matplotlib
```

---

## 👨‍💻 Authors

- **Vinay** — Model design, training pipeline, research

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">Made with ❤️ for NLP Research</p>
