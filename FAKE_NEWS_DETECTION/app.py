import streamlit as st
import joblib
import pickle
import re
import os
import numpy as np
from collections import Counter

st.set_page_config(page_title="Fake News Detection (Ensemble Model)", layout="centered")

# Base directory where this app.py resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def safe_load(filename):
    full_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(full_path):
        st.warning(f"⚠️ File not found: {full_path}")
        return None
    try:
        return joblib.load(full_path)
    except Exception:
        try:
            with open(full_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"❌ Failed to load {filename}: {e}")
            return None

@st.cache_resource
def load_models():
    tfidf = safe_load("tfidf_vectorizer_new.pkl")           # Update if filename different
    svm = safe_load("svm_model_v1.pkl")
    nb = safe_load("naive_bayes_model_v2.pkl")
    bert_xgb = safe_load("bert_xgb_model.pkl")
    le = safe_load("label_encoder.joblib")
    return tfidf, svm, nb, bert_xgb, le

tfidf, svm_model, nb_model, bert_xgb_model, label_encoder = load_models()

def predict_model(model, X, label_encoder=None):
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            pred_idx = np.argmax(probs, axis=1)[0]
            pred_label = model.classes_[pred_idx]
            conf = probs[0, pred_idx]
        else:
            pred_label = model.predict(X)[0]
            conf = 1.0
        if label_encoder is not None:
            try:
                pred_label = label_encoder.inverse_transform([pred_label])[0]
            except Exception:
                pass
        return pred_label, conf
    except Exception as e:
        return f"Error: {e}", None

st.title("📰 Fake News Detection (Ensemble-Based App)")
st.write("Predicts whether a news article is Fake or Real using SVM, Naive Bayes, and BERT+XGBoost — then combines them via ensemble voting.")

text_input = st.text_area("Enter News Article Text", height=250)

if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(text_input)
        if tfidf is None:
            st.error("TF-IDF vectorizer not loaded. Check file name.")
        else:
            X_tfidf = tfidf.transform([cleaned_text])

            svm_pred, svm_conf = predict_model(svm_model, X_tfidf, label_encoder)
            nb_pred, nb_conf = predict_model(nb_model, X_tfidf, label_encoder)
            bert_pred, bert_conf = predict_model(bert_xgb_model, [cleaned_text], label_encoder)

            st.subheader("📊 Individual Model Predictions")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("SVM Prediction", svm_pred, f"Confidence: {svm_conf:.2f}" if svm_conf else "")
            with col2:
                st.metric("Naive Bayes Prediction", nb_pred, f"Confidence: {nb_conf:.2f}" if nb_conf else "")
            with col3:
                st.metric("BERT+XGBoost Prediction", bert_pred, f"Confidence: {bert_conf:.2f}" if bert_conf else "")

            predictions = [svm_pred, nb_pred, bert_pred]
            valid_preds = [p for p in predictions if not str(p).startswith("Error")]

            if valid_preds:
                final_label = Counter(valid_preds).most_common(1)[0][0]
                st.markdown("---")
                st.subheader("🧩 Ensemble Final Prediction")
                st.success(f"**Final Verdict:** {final_label}")
                st.write(f"Votes → {Counter(valid_preds)}")
            else:
                st.error("No valid predictions could be made. Check models or vectorizer compatibility.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Ensemble Learning (SVM + NB + BERT+XGBoost).")
