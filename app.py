from flask import Flask, render_template, request
import joblib
import pandas as pd
from urllib.parse import urlparse

from feature_extraction import extract_features
from explain import SHAPExplainer

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model/rf_modelowdataset.pkl")

feature_names = [
    "url_length","has_ip_address","dot_count","https_flag","url_entropy",
    "token_count","subdomain_count","query_param_count",
    "tld_length","path_length","has_hyphen_in_domain",
    "number_of_digits","tld_popularity","suspicious_file_extension",
    "domain_name_length","percentage_numeric_chars",
    "has_phishing_keyword","is_trusted_domain"
]

shap_helper = SHAPExplainer(model, feature_names)

# =========================
# CONTEXT CHECK
# =========================
def context_check(domain):
    trusted = [
        "google.com", "youtube.com",
        "facebook.com", "instagram.com",
        "shopee.co.id", "tokopedia.com", 
        "chatgpt.com"
    ]

    brands = ["google", "facebook", "shopee", "instagram"]

    if any(domain.endswith(t) for t in trusted):
        return "trusted"

    if any(b in domain for b in brands):
        return "spoof"

    return "unknown"

# =========================
# HIGHLIGHT URL
# =========================
def highlight_url(url):
    suspicious_keywords = ["login", "verify", "secure", "account", "bank"]

    highlighted = url

    for k in suspicious_keywords:
        highlighted = highlighted.replace(
            k,
            f"<span style='color:red; font-weight:bold'>{k}</span>"
        )

    return highlighted

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        url = request.form["url"]

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # =========================
        # FEATURE EXTRACTION
        # =========================
        features = extract_features(url)

        if len(features) != len(feature_names):
            return f"ERROR: jumlah fitur ({len(features)}) tidak sama dengan feature_names ({len(feature_names)})"

        X = pd.DataFrame([features], columns=feature_names)

        # =========================
        # PREDICTION
        # =========================
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)

        # 🔥 FIX LABEL (phishing = 1, legit = 0)
        idx_legit = classes.index(0)
        idx_phishing = classes.index(1)

        legit_prob = proba[idx_legit]
        phishing_prob = proba[idx_phishing]

        risk_score = phishing_prob * 100

        # =========================
        # THRESHOLD
        # =========================
        if risk_score < 40:
            prediction = "Legitimate"
        elif risk_score < 70:
            prediction = "Suspicious"
        else:
            prediction = "Phishing"

        # =========================
        # CONTEXT ADJUSTMENT
        # =========================
        context = context_check(domain)

        if context == "trusted":
            prediction = "Legitimate"
            risk_score *= 0.5

        elif context == "spoof":
            prediction = "Phishing"
            risk_score = max(risk_score, 85)

        # =========================
        # SHAP EXPLANATION
        # =========================
        explanation_raw = shap_helper.explain(X, features)

        explanations = []
        risk_factors = []

        # ambil hanya kontribusi positif
        positive_explanations = [e for e in explanation_raw if e["value"] > 0]

        # urutkan berdasarkan impact terbesar
        positive_explanations = sorted(
            positive_explanations,
            key=lambda x: x["impact"],
            reverse=True
        )

        top_explanations = positive_explanations[:3]
        total_impact = sum(e["impact"] for e in top_explanations) + 1e-6

        for e in top_explanations:
            text = e["text"]
            impact = e["impact"]

            contribution_pct = (impact / total_impact) * risk_score
            contribution_pct = round(contribution_pct, 2)

            explanations.append(f"🔴 {text} (+{contribution_pct}%)")
            risk_factors.append(text)

        # fallback kalau semua negatif
        if not explanations and explanation_raw:
            top = explanation_raw[0]
            fallback_pct = round(risk_score, 2)

            explanations.append(f"🔴 {top['text']} (+{fallback_pct}%)")
            risk_factors.append(top["text"])

        # =========================
        # SUMMARY
        # =========================
        if prediction == "Legitimate":
            summary = (
                f"URL ini terdeteksi AMAN ({round(risk_score,2)}%). "
                "Tidak ditemukan indikator phishing yang signifikan."
            )

        elif prediction == "Suspicious":
            summary = (
                f"URL ini terdeteksi MENCURIGAKAN ({round(risk_score,2)}%). "
                + ("Beberapa indikator seperti: " + ", ".join(risk_factors[:2]) if risk_factors else "")
            )

        else:
            summary = (
                f"URL ini terdeteksi PHISHING ({round(risk_score,2)}%). "
                + ("Indikator utama: " + ", ".join(risk_factors[:3]) if risk_factors else "")
            )

        # =========================
        # RESULT
        # =========================
        result = {
    "url": url,
    "prediction": prediction,
    "risk_score": round(risk_score, 2),
    "explanations": explanations,
    "summary": summary,

    "highlighted_url": highlight_url(url),

    "url_parts": {
        "scheme": parsed.scheme,
        "domain": parsed.netloc,
        "path": parsed.path,
        "query": parsed.query
    },

    "top_features": [
        (e["text"], round((e["impact"] / total_impact) * risk_score, 2))
        for e in top_explanations
    ]
}
        
    return render_template("index.html", result=result)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("MODEL CLASSES:", model.classes_)
    app.run(debug=True)