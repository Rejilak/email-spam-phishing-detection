from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import BERT_OUTPUT_DIR, LABELS
from .preprocess import clean_text

class BertEmailClassifier:
    def __init__(self, model_dir: Path = BERT_OUTPUT_DIR):
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str):
        text = clean_text(text)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        pred_id = int(torch.argmax(logits, dim=1).item())
        return LABELS[pred_id], probs
