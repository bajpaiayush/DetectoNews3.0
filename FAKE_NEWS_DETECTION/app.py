import streamlit as st
import joblib
import pickle
import re
import os
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer  # ✅ Needed for BERT embeddings

st.set_page_config(page_title="Fake News Detection (Ensemble Model)", layout="centered")

# === Base directory ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Clean text ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Safe load ===
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

# === Cache the BERT model so it doesn't reload every time ===
@st.cache_resource
def get_bert_model(model_name):
    return SentenceTransformer(model_name)

# === Load models ===
@st.cache_resource
def load_models():
    tfidf = safe_load("tfidf_vectorizer_new.pkl")
    svm = safe_load("svm_model_v1.pkl")
    nb = safe_load("naive_bayes_model_v2.pkl")
    bert_xgb = safe_load("bert_xgb_wrapper_fixed.pkl")
    le = safe_load("label_encoder.joblib")
    return tfidf, svm, nb, bert_xgb, le

tfidf, svm_model, nb_model, bert_xgb_model, label_encoder = load_models()

# === Predict helper ===
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

# === Streamlit UI ===
st.title("📰 Fake News Detection (Ensemble-Based App)")
st.write("Predicts whether a news article is **Fake or Real** using SVM, Naive Bayes, and BERT+XGBoost — then combines them via ensemble voting.")

text_input = st.text_area("Enter News Article Text", height=250)

if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(text_input)
        if tfidf is None:
            st.error("TF-IDF Vectorizer not loaded. Check file name.")
        else:
            X_tfidf = tfidf.transform([cleaned_text])

            # --- Predictions for SVM + NB ---
            svm_pred, svm_conf = predict_model(svm_model, X_tfidf, label_encoder)
            nb_pred, nb_conf = predict_model(nb_model, X_tfidf, label_encoder)

            # --- BERT + XGBoost (dict wrapper fix) ---
            bert_pred, bert_conf = "N/A", None
            try:
                if isinstance(bert_xgb_model, dict):
                    bert_model_name = bert_xgb_model.get("embed_model_name")
                    xgb_model = bert_xgb_model.get("xgb_model")

                    if bert_model_name is not None and xgb_model is not None:
                        bert_model = get_bert_model(bert_model_name)
                        embedding = bert_model.encode([cleaned_text])
                        pred = xgb_model.predict(embedding)[0]
                        if hasattr(xgb_model, "predict_proba"):
                            prob = xgb_model.predict_proba(embedding)
                            conf = float(np.max(prob))
                        else:
                            conf = 1.0
                        bert_pred, bert_conf = pred, conf
                    else:
                        raise ValueError("Missing 'embed_model_name' or 'xgb_model' in wrapper dict.")
                else:
                    bert_pred, bert_conf = predict_model(bert_xgb_model, [cleaned_text], label_encoder)
            except Exception as e:
                bert_pred, bert_conf = f"Error: {e}", None

            # --- Display results ---
            st.subheader("📊 Individual Model Predictions")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("SVM Prediction", svm_pred, f"Confidence: {svm_conf:.2f}" if svm_conf else "")
            with col2:
                st.metric("Naive Bayes Prediction", nb_pred, f"Confidence: {nb_conf:.2f}" if nb_conf else "")
            with col3:
                st.metric("BERT+XGBoost Prediction", bert_pred, f"Confidence: {bert_conf:.2f}" if bert_conf else "")

            # --- Ensemble voting ---
            predictions = [svm_pred, nb_pred, bert_pred]
            valid_preds = [p for p in predictions if not str(p).startswith("Error") and p != "N/A"]

            if valid_preds:
                final_label = Counter(valid_preds).most_common(1)[0][0]
                if isinstance(final_label, (int, np.integer)):
                    final_label = "Fake" if final_label == 0 else "Real"

                st.markdown("---")
                st.subheader("🧩 Ensemble Final Prediction")
                st.success(f"**Final Verdict:** {final_label}")
                st.write(f"Votes → {Counter(valid_preds)}")
            else:
                st.error("No valid predictions could be made. Check models or vectorizer compatibility.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Ensemble Learning (SVM + NB + BERT+XGBoost wrapper).")

