import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "hand_sign_data.csv"
MODEL_FILE = "hand_sign_model.pkl"

# Load the dataset
data = pd.read_csv(DATA_FILE)

print("Total samples:", len(data))
print("Number of labels:", data["label"].nunique())

# Separate features and labels
X = data.drop(columns=["label"])
y = data["label"]

# Remove incomplete rows, if any
valid = X.notna().all(axis=1)
X = X.loc[valid]
y = y.loc[valid]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create and train the classifier
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", probability=True, random_state=42)
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model
joblib.dump(model, MODEL_FILE)

print(f"\nModel saved successfully: {MODEL_FILE}")
