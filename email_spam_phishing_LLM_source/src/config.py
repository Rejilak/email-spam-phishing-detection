from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

LABELS = ["Legitimate", "Spam", "Phishing"]
TEXT_COL = "text"
LABEL_COL = "label"

BASE_MODEL_NAME = "distilbert-base-uncased"
BERT_OUTPUT_DIR = MODELS_DIR / "bert_email_classifier"
