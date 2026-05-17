import shap
import numpy as np

class SHAPExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X, raw_features):
        shap_values = self.explainer.shap_values(X)

        # ambil class phishing (0)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.array(shap_values).flatten()

        explanations = []

        for i, val in enumerate(shap_values):

            # 🔥 safety check
            if i >= len(self.feature_names) or i >= len(raw_features):
                continue

            name = self.feature_names[i]
            value = raw_features[i]

            desc = self.map_feature(name, value)

            if desc:
                explanations.append({
                    "feature": name,
                    "text": desc,
                    "value": float(val),     # arah (positif = phishing)
                    "impact": abs(val)       # kekuatan pengaruh
                })

        # 🔥 urutkan berdasarkan pengaruh terbesar
        explanations = sorted(explanations, key=lambda x: x["impact"], reverse=True)

        return explanations[:5]

    def map_feature(self, name, value):

        # =========================
        # DOMAIN & STRUCTURE
        # =========================
        if name == "subdomain_count":
            if value >= 2:
                return f"Memiliki banyak subdomain ({value})"
            elif value == 1:
                return "Memiliki subdomain"
            return None  # 0 = normal

        # =========================
        # KEYWORD PHISHING
        # =========================
        elif name == "has_phishing_keyword":
            if value == 1:
                return "Mengandung kata mencurigakan seperti login atau verify"
            return None

        # =========================
        # ENTROPY (ACAK)
        # =========================
        elif name == "url_entropy":
            if value > 4.2:
                return "Struktur URL terlihat acak / tidak biasa"
            return None

        # =========================
        # PANJANG URL
        # =========================
        elif name == "url_length":
            if value > 75:
                return f"URL sangat panjang ({value} karakter)"
            elif value > 40:
                return f"URL cukup panjang ({value} karakter)"
            return None

        # =========================
        # PARAMETER
        # =========================
        elif name == "query_param_count":
            if value >= 2:
                return "Memiliki banyak parameter URL"
            elif value == 1:
                return "Memiliki parameter URL"
            return None

        # =========================
        # ANGKA DALAM URL
        # =========================
        elif name == "number_of_digits":
            if value >= 5:
                return f"Mengandung banyak angka ({value}) dalam URL"
            return None

        # =========================
        # DOMAIN STRUCTURE
        # =========================
        elif name == "has_hyphen_in_domain":
            if value == 1:
                return "Domain menggunakan tanda '-'"
            return None

        # =========================
        # HTTPS
        # =========================
        elif name == "https_flag":
            if value == 0:
                return "Tidak menggunakan HTTPS"
            return None  # HTTPS = normal → tidak ditampilkan

        # =========================
        # TRUSTED DOMAIN
        # =========================
        elif name == "is_trusted_domain":
            if value == 0:
                return "Domain bukan termasuk domain terpercaya"
            return None  # trusted = aman → skip

        # =========================
        # FILE EXTENSION
        # =========================
        elif name == "suspicious_file_extension":
            if value == 1:
                return "Mengandung ekstensi file mencurigakan"
            return None

        return None