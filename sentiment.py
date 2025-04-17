import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch.nn.functional as F

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MODEL_DIR = "./local_model"  # <-- Local directory to save/load

# Check if local model exists
if os.path.exists(MODEL_DIR):
    print("Loading model from local directory...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    config = AutoConfig.from_pretrained(MODEL_DIR)
else:
    print("Downloading model from Hugging Face Hub...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    
    # Save locally for next time
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    config.save_pretrained(MODEL_DIR)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def predict_sentiment(text):
    processed_text = preprocess(text)
    encoded_input = tokenizer(processed_text, return_tensors='pt')
    output = model(**encoded_input)
    probs = F.softmax(output.logits, dim=1)
    index_of_sentiment = probs.argmax().item()
    sentiment = config.id2label[index_of_sentiment]
    score = probs[0, index_of_sentiment].item()
    
    return sentiment, score
