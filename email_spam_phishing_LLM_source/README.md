# Email Spam & Phishing Detection (LLM + Deep Learning)

## LLM Part (Transformer)
The LLM component is implemented using **DistilBERT**.
- Train: `src/train_bert.py`
- Predict: `src/predict_bert.py`
- Demo UI: `app/streamlit_app.py`

## Setup
```bash
pip install -r requirements.txt
```

## Train LLM Model
```bash
python -m src.train_bert --data data/sample_emails.csv --epochs 2 --batch 8
```

## Run Demo App
```bash
streamlit run app/streamlit_app.py
```
