# app.py
import streamlit as st
import joblib
import pickle
import re
import os
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer  # BERT embeddings

st.set_page_config(page_title="Fake News Detection", layout="centered")

# -----------------------
# Custom CSS / Header
# -----------------------
st.markdown(
    """
    <style>
    .stApp {
      background: linear-gradient(180deg, #f7f9fc 0%, #eef6ff 100%);
      font-family: "Inter", "Segoe UI", Roboto, sans-serif;
      color: #0f1724;
    }
    .custom-header { display:flex; align-items:center; gap:12px; margin-bottom: 8px; }
    .custom-header h1 { margin: 0; font-size: 1.6rem; letter-spacing: -0.2px; }
    .underline { height:6px; width:90px; border-radius:12px; background: linear-gradient(90deg,#6ee7b7,#60a5fa,#a78bfa); animation: slide 3s linear infinite; opacity: .95; }
    @keyframes slide { 0% { transform: translateX(-10px); } 50% { transform: translateX(10px); } 100% { transform: translateX(-10px); } }
    .card { background: white; border-radius: 12px; padding: 14px; box-shadow: 0 6px 20px rgba(15,23,42,0.06); border: 1px solid rgba(15,23,42,0.04); margin-bottom: 12px; }
    textarea { min-height: 160px !important; border-radius: 10px !important; padding: 12px !important; box-shadow: inset 0 2px 8px rgba(2,6,23,0.04); border: 1px solid rgba(15,23,42,0.08) !important; font-size: 14px; resize: vertical; }
    div.stButton > button { background: linear-gradient(90deg,#4f46e5,#06b6d4); color: #fff; border-radius: 10px; padding: 10px 18px; border: none; box-shadow: 0 8px 18px rgba(6,95,70,0.08); font-weight: 600; transition: transform .14s ease, box-shadow .14s ease, opacity .14s ease; }
    div.stButton > button:hover { transform: translateY(-3px); box-shadow: 0 12px 28px rgba(6,95,70,0.12); opacity: 0.98; }
    .metric-card { background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(250,250,255,0.95)); border-radius: 12px; padding: 10px; text-align: center; box-shadow: 0 8px 20px rgba(2,6,23,0.04); border: 1px solid rgba(15,23,42,0.03); }
    .metric-title { font-size: 0.9rem; color: #475569; margin-bottom:6px; }
    .metric-val { font-size: 1.05rem; font-weight:700; }
    .small-note { font-size: 0.88rem; color: #64748b; }
    .stMarkdown hr { border: 0; height: 1px; background: linear-gradient(90deg, rgba(99,102,241,0.12), rgba(34,197,94,0.08)); margin: 12px 0; }
    @media (max-width: 600px) { .custom-header h1 { font-size: 1.25rem; } .underline { width: 60px; height: 5px; } }
    </style>
    <div class="custom-header">
      <h1>📰 Fake News Detection</h1>
      <div class="underline"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

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


# === Cache the BERT model ===
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


# === Helpers for prediction / extracting P(Fake) ===
def predict_model_simple(model, X):
    """
    Returns (pred_label, conf, probs, classes)
    - pred_label: predicted label (as returned by model.predict)
    - conf: confidence for the predicted class (float) or None
    - probs: full probability array if available (2D) else None
    - classes: model.classes_ if available else None
    """
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            pred_idx = np.argmax(probs, axis=1)[0]
            pred_label = model.classes_[pred_idx] if hasattr(model, "classes_") else pred_idx
            conf = float(probs[0, pred_idx])
            return pred_label, conf, probs, getattr(model, "classes_", None)
        else:
            pred = model.predict(X)
            pred_label = pred[0]
            return pred_label, 1.0, None, getattr(model, "classes_", None)
    except Exception as e:
        return f"Error: {e}", None, None, None


def get_prob_fake_from_preds(pred_label, conf, probs, classes, label_encoder):
    """
    Attempts to return P(Fake) for a single model prediction using:
     - If probs and classes available: return prob at index for "Fake"
     - Else, if pred_label available and conf (confidence for predicted class) available:
         return conf if pred_label corresponds to 'Fake' else (1 - conf)
     - Uses label_encoder if labels are numeric and 'Fake' string is present in encoder
    Returns None if cannot compute.
    """
    # Try probs + classes first
    try:
        if probs is not None and classes is not None:
            # Find index corresponding to 'Fake' in classes
            classes_list = list(classes)
            # Try string 'Fake'
            for candidate in ("Fake", "fake"):
                if candidate in [str(c) for c in classes_list]:
                    idx = [str(c) for c in classes_list].index(candidate)
                    return float(probs[0, idx])
            # Try numeric label via label_encoder if present
            if label_encoder is not None:
                try:
                    encoded = label_encoder.transform(["Fake"])[0]
                    for i, c in enumerate(classes_list):
                        if str(c) == str(encoded):
                            return float(probs[0, i])
                except Exception:
                    pass
            # If no 'Fake' named class, try to infer: often 0->Fake, 1->Real
            # we check if classes contain 0
            for i, c in enumerate(classes_list):
                if str(c) in ("0", "0.0"):
                    return float(probs[0, i])
    except Exception:
        pass
    # Fallback to pred_label + conf
    try:
        if pred_label is not None and conf is not None:
            # If pred_label is a string that matches 'Fake'
            if str(pred_label).lower() == "fake":
                return float(conf)
            # If label encoder exists, check mapping
            if label_encoder is not None:
                try:
                    encoded = label_encoder.transform(["Fake"])[0]
                    if str(pred_label) == str(encoded):
                        return float(conf)
                except Exception:
                    pass
            # Otherwise assume binary: conf is probability of predicted class, so return 1-conf if pred != Fake
            return 1.0 - float(conf)
    except Exception:
        pass
    return None


# === UI: Input card ===
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("**Enter News Article Text**")
text_input = st.text_area("", height=250, placeholder="Paste headline or article text here...")
st.markdown('</div>', unsafe_allow_html=True)

# Ensemble weight sliders
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⚖️ Ensemble Weights (adjustable)")
w_nb = st.slider("Weight: Naive Bayes", 0.0, 1.0, 0.15, step=0.01)
w_svm = st.slider("Weight: SVM", 0.0, 1.0, 0.30, step=0.01)
w_bert = st.slider("Weight: BERT+XGBoost", 0.0, 1.0, 0.55, step=0.01)
# Normalize weights to sum to 1 for display / usage
total_w = w_nb + w_svm + w_bert
if total_w == 0:
    w_nb_norm = w_svm_norm = w_bert_norm = 1.0 / 3.0
else:
    w_nb_norm = w_nb / total_w
    w_svm_norm = w_svm / total_w
    w_bert_norm = w_bert / total_w
st.markdown(f"<div class='small-note'>Normalized weights → NB: {w_nb_norm:.2f}, SVM: {w_svm_norm:.2f}, BERT-XGB: {w_bert_norm:.2f}</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Predict button and logic
if st.button("Predict"):

    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(text_input)

        if tfidf is None:
            st.error("TF-IDF Vectorizer not loaded. Check file name.")
        else:
            X_tfidf = tfidf.transform([cleaned_text])

            # --- Predictions (SVM + NB) ---
            svm_pred, svm_conf, svm_probs, svm_classes = (None, None, None, None)
            nb_pred, nb_conf, nb_probs, nb_classes = (None, None, None, None)
            bert_pred, bert_conf, bert_probs, bert_classes = (None, None, None, None)

            if svm_model is not None:
                svm_pred, svm_conf, svm_probs, svm_classes = predict_model_simple(svm_model, X_tfidf)
            else:
                st.warning("SVM model not loaded.")

            if nb_model is not None:
                nb_pred, nb_conf, nb_probs, nb_classes = predict_model_simple(nb_model, X_tfidf)
            else:
                st.warning("Naive Bayes model not loaded.")

            # --- BERT + XGBoost wrapper prediction ---
            try:
                if bert_xgb_model is None:
                    st.warning("BERT+XGBoost wrapper not loaded.")
                    bert_pred, bert_conf, bert_probs, bert_classes = ("N/A", None, None, None)
                elif isinstance(bert_xgb_model, dict):
                    bert_model_name = bert_xgb_model.get("embed_model_name")
                    xgb_model = bert_xgb_model.get("xgb_model")
                    if bert_model_name and xgb_model:
                        bert_model = get_bert_model(bert_model_name)
                        embedding = bert_model.encode([cleaned_text])
                        # Ensure correct shape for XGB
                        emb_arr = np.array(embedding)
                        # If xgb has predict_proba
                        if hasattr(xgb_model, "predict_proba"):
                            prob = xgb_model.predict_proba(emb_arr)
                            pred_idx = int(np.argmax(prob, axis=1)[0])
                            pred_label = xgb_model.classes_[pred_idx] if hasattr(xgb_model, "classes_") else pred_idx
                            bert_pred, bert_conf, bert_probs, bert_classes = pred_label, float(prob[0, pred_idx]), prob, getattr(xgb_model, "classes_", None)
                        else:
                            pred = xgb_model.predict(emb_arr)[0]
                            bert_pred, bert_conf, bert_probs, bert_classes = pred, 1.0, None, getattr(xgb_model, "classes_", None)
                    else:
                        raise ValueError("Missing embed_model_name or xgb_model in wrapper dict.")
                else:
                    # Maybe the wrapper is a sklearn-like estimator
                    bert_pred, bert_conf, bert_probs, bert_classes = predict_model_simple(bert_xgb_model, [cleaned_text])
            except Exception as e:
                bert_pred, bert_conf, bert_probs, bert_classes = (f"Error: {e}", None, None, None)

            # --- Show Individual Results ---
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Individual Model Predictions")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card"><div class="metric-title">SVM</div><div class="metric-val">{}</div></div>'.format(svm_pred), unsafe_allow_html=True)
                if svm_conf: st.markdown(f"<div class='small-note'>Confidence: {svm_conf:.2f}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card"><div class="metric-title">Naive Bayes</div><div class="metric-val">{}</div></div>'.format(nb_pred), unsafe_allow_html=True)
                if nb_conf: st.markdown(f"<div class='small-note'>Confidence: {nb_conf:.2f}</div>", unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card"><div class="metric-title">BERT+XGBoost</div><div class="metric-val">{}</div></div>'.format(bert_pred), unsafe_allow_html=True)
                if bert_conf: st.markdown(f"<div class='small-note'>Confidence: {bert_conf:.2f}</div>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # --- Ensemble: try soft-voting using P(Fake) if available ---
            p_fake_list = []
            weight_list = []
            # SVM
            p_svm = get_prob_fake_from_preds(svm_pred, svm_conf, svm_probs, svm_classes, label_encoder)
            if p_svm is not None:
                p_fake_list.append((p_svm, w_svm_norm))
                weight_list.append(w_svm_norm)
            # NB
            p_nb = get_prob_fake_from_preds(nb_pred, nb_conf, nb_probs, nb_classes, label_encoder)
            if p_nb is not None:
                p_fake_list.append((p_nb, w_nb_norm))
                weight_list.append(w_nb_norm)
            # BERT
            p_bert = get_prob_fake_from_preds(bert_pred, bert_conf, bert_probs, bert_classes, label_encoder)
            if p_bert is not None:
                p_fake_list.append((p_bert, w_bert_norm))
                weight_list.append(w_bert_norm)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🧩 Ensemble Final Prediction")

            final_label = None
            if p_fake_list and sum(weight_list) > 0:
                # compute weighted average of available p_fakes
                weighted_sum = sum(p * w for (p, w) in p_fake_list)
                # normalize by sum of weights that were used
                normalized_weights_sum = sum(w for (_, w) in p_fake_list)
                p_final = weighted_sum / normalized_weights_sum
                final_label = "Fake" if p_final >= 0.5 else "Real"
                st.success(f"**Final Verdict:** {final_label}")
                st.write(f"Soft-vote P(Fake) = {p_final:.3f} (used {len(p_fake_list)} model probabilities)")
                # Also show votes as fallback info
                votes = [str(v) for v in [svm_pred, nb_pred, bert_pred] if not str(v).startswith("Error") and v != "N/A"]
                if votes:
                    st.write(f"Votes → {Counter(votes)}")
            else:
                # fallback to majority vote
                predictions = [svm_pred, nb_pred, bert_pred]
                valid_preds = [p for p in predictions if not str(p).startswith("Error") and p != "N/A"]
                if valid_preds:
                    final_label = Counter(valid_preds).most_common(1)[0][0]
                    if isinstance(final_label, (int, np.integer)):
                        # map numeric to Fake/Real assumption
                        final_label = "Fake" if final_label == 0 else "Real"
                    st.success(f"**Final Verdict:** {final_label} (majority vote fallback)")
                    st.write(f"Votes → {Counter(valid_preds)}")
                else:
                    st.error("No valid predictions could be made. Check models and vectorizer compatibility.")

            st.markdown('</div>', unsafe_allow_html=True)
