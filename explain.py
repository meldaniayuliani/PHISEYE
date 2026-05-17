import shap
import numpy as np


class SHAPExplainer:

    def __init__(self, model, feature_names):

        self.model = model
        self.feature_names = feature_names

        # gunakan explainer modern
        self.explainer = shap.Explainer(model)

    # =====================================================
    # EXPLAIN
    # =====================================================
    def explain(self, X, raw_features):

        # hitung shap value
        shap_values = self.explainer(X)

        # =========================
        # HANDLE BINARY CLASSIFICATION
        # =========================
        # ambil class phishing (class 1)
        values = shap_values.values

        if len(values.shape) == 3:
            # shape: (sample, feature, class)
            values = values[0, :, 1]

        elif len(values.shape) == 2:
            # shape: (sample, feature)
            values = values[0]

        explanations = []

        # =========================
        # LOOP FEATURE
        # =========================
        for i, shap_val in enumerate(values):

            if i >= len(self.feature_names):
                continue

            if i >= len(raw_features):
                continue

            feature_name = self.feature_names[i]
            feature_value = raw_features[i]

            # deskripsi fitur
            description = self.map_feature(
                feature_name,
                feature_value,
                shap_val
            )

            if description is None:
                continue

            explanations.append({

                "feature": feature_name,

                "feature_value": feature_value,

                # nilai shap asli
                "value": float(shap_val),

                # besar pengaruh
                "impact": abs(float(shap_val)),

                # arah pengaruh
                "direction": (
                    "increase"
                    if shap_val > 0
                    else "decrease"
                ),

                # teks penjelasan
                "text": description
            })

        # =========================
        # SORT BERDASARKAN IMPACT
        # =========================
        explanations = sorted(
            explanations,
            key=lambda x: x["impact"],
            reverse=True
        )

        return explanations[:5]

    # =====================================================
    # MAP FEATURE
    # =====================================================
    def map_feature(self, name, value, shap_value):

        # arah kontribusi
        if shap_value > 0:
            effect = "meningkatkan risiko phishing"
            icon = "🔴"
        else:
            effect = "menurunkan risiko phishing"
            icon = "🟢"

        # =================================================
        # URL LENGTH
        # =================================================
        if name == "url_length":

            if value > 75:
                return f"{icon} URL sangat panjang ({value} karakter) ({effect})"

            elif value > 40:
                return f"{icon} URL cukup panjang ({value} karakter) ({effect})"

        # =================================================
        # URL ENTROPY
        # =================================================
        elif name == "url_entropy":

            if value > 4.2:
                return f"{icon} Struktur URL terlihat acak / tidak biasa ({effect})"

        # =================================================
        # SUBDOMAIN
        # =================================================
        elif name == "subdomain_count":

            if value >= 3:
                return f"{icon} Memiliki banyak subdomain ({value}) ({effect})"

            elif value >= 1:
                return f"{icon} Memiliki subdomain ({effect})"

        # =================================================
        # QUERY PARAMETER
        # =================================================
        elif name == "query_param_count":

            if value >= 2:
                return f"{icon} Memiliki banyak parameter URL ({effect})"

            elif value == 1:
                return f"{icon} Memiliki parameter URL ({effect})"

        # =================================================
        # NUMBER OF DIGITS
        # =================================================
        elif name == "number_of_digits":

            if value >= 5:
                return f"{icon} Mengandung banyak angka ({value}) dalam URL ({effect})"

        # =================================================
        # PHISHING KEYWORD
        # =================================================
        elif name == "has_phishing_keyword":

            if value == 1:
                return f"{icon} Mengandung kata mencurigakan seperti login atau verify ({effect})"

        # =================================================
        # HTTPS
        # =================================================
        elif name == "https_flag":

            if value == 0:
                return f"{icon} Tidak menggunakan HTTPS ({effect})"

            elif value == 1 and shap_value < 0:
                return f"{icon} Menggunakan HTTPS ({effect})"

        # =================================================
        # HYPHEN DOMAIN
        # =================================================
        elif name == "has_hyphen_in_domain":

            if value == 1:
                return f"{icon} Domain menggunakan tanda '-' ({effect})"

        # =================================================
        # SUSPICIOUS EXTENSION
        # =================================================
        elif name == "suspicious_file_extension":

            if value == 1:
                return f"{icon} Mengandung ekstensi file mencurigakan ({effect})"

        # =================================================
        # TRUSTED DOMAIN
        # =================================================
        elif name == "is_trusted_domain":

            if value == 1 and shap_value < 0:
                return f"{icon} Domain termasuk domain terpercaya ({effect})"

            elif value == 0 and shap_value > 0:
                return f"{icon} Domain bukan termasuk domain terpercaya ({effect})"

        # =================================================
        # DOMAIN LENGTH
        # =================================================
        elif name == "domain_name_length":

            if value > 25:
                return f"{icon} Nama domain cukup panjang ({effect})"

        # =================================================
        # PERCENTAGE NUMERIC
        # =================================================
        elif name == "percentage_numeric_chars":

            if value > 0.15:
                return f"{icon} Persentase angka dalam URL cukup tinggi ({effect})"

        return None