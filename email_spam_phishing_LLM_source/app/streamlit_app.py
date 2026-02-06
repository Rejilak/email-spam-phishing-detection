import streamlit as st
from pathlib import Path
from src.config import BERT_OUTPUT_DIR, LABELS
from src.predict_bert import BertEmailClassifier

st.set_page_config(page_title="Email Detector (LLM)", layout="centered")
st.title("📧 Email Spam & Phishing Detection (LLM - DistilBERT)")
st.write("Paste an email below. The Transformer model predicts: Legitimate / Spam / Phishing.")

email_text = st.text_area("Email Content", height=220)

if st.button("Classify"):
    if not email_text.strip():
        st.warning("Please paste some email content.")
        st.stop()

    if not Path(BERT_OUTPUT_DIR).exists():
        st.error("LLM model not found. Train first: `python -m src.train_bert --data data/sample_emails.csv`")
        st.stop()

    clf = BertEmailClassifier()
    label, probs = clf.predict(email_text)

    st.success(f"Prediction: **{label}**")
    st.write("Probabilities:")
    for l, p in zip(LABELS, probs):
        st.write(f"- {l}: {p:.4f}")
