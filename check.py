import pickle

with open("artifacts/metadata.pkl", "rb") as f:
    meta = pickle.load(f)

print("Feature names:", meta["feature_names"])
print("Original columns:", meta["original_columns"])