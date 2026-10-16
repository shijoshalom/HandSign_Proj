import pandas as pd

data = pd.read_csv("hand_sign_data.csv")

print("\nCURRENT HAND SIGN LABELS")
print("-" * 35)

for label, count in data["label"].value_counts().items():
    print(f"{label:<30} {count:>5} samples")

print("\nTotal samples:", len(data))
print("Unique labels:", data["label"].nunique())
