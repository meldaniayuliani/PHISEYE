import pickle

with open("model/phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

print("MODEL BERHASIL LOAD")