import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("dataset/dataset_updated.csv")

# =========================
# LABEL
# 1 = legitimate, 0 = phishing
# =========================
y = df["ClassLabel"]

# =========================
# FEATURE ENGINEERING
# =========================
from feature_extraction import extract_features

X = []
for url in df["URL"]:
    X.append(extract_features(url))

feature_names = [
    "url_length","has_ip_address","dot_count","https_flag","url_entropy",
    "token_count","subdomain_count","query_param_count",
    "tld_length","path_length","has_hyphen_in_domain",
    "number_of_digits","tld_popularity","suspicious_file_extension",
    "domain_name_length","percentage_numeric_chars",
    "has_phishing_keyword","is_trusted_domain"
]

X = pd.DataFrame(X, columns=feature_names)

# =========================
# SPLIT DATA (IMPORTANT)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# 1. RANDOM FOREST (MAIN MODEL - CONTROL OVERFITTING)
# =========================================================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,              # penting untuk hindari overfitting
    min_samples_leaf=5,       # penting
    random_state=42
)

rf.fit(X_train, y_train)

# =========================================================
# 2. DECISION TREE (BASELINE)
# =========================================================
dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

dt.fit(X_train, y_train)

# =========================================================
# 3. LOGISTIC REGRESSION (LINEAR BASELINE)
# =========================================================
lr = LogisticRegression(
    max_iter=1000
)

lr.fit(X_train, y_train)

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, name):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"\n===== {name} =====")
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print(classification_report(y_test, model.predict(X_test)))

# =========================
# EVALUATE ALL MODELS
# =========================
evaluate(rf, "Random Forest")
evaluate(dt, "Decision Tree")
evaluate(lr, "Logistic Regression")

# =========================
# SAVE BEST MODEL (RF biasanya terbaik)
# =========================
joblib.dump(rf, "model/rf_modelnewtrainbgt.pkl")
joblib.dump(dt, "model/dt_modelnewtrainbgt.pkl")
joblib.dump(lr, "model/lr_modelnewtrainbgt.pkl")

print("\nModels saved successfully!")