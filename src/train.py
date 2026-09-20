import pandas as pd
import joblib
import yaml
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

n_estimators = params["train"]["n_estimators"]
random_state = params["train"]["random_state"]
test_size = params["train"]["test_size"]

# Load dataset
df = pd.read_csv("data/Iris.csv")

# Features
X = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

# Target
y = df["Species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=random_state
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Model")
print("Number of trees:", n_estimators)
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "models/random_forest.pkl")

# Save metrics
with open("results/metrics.json", "w") as file:
    json.dump({"accuracy": accuracy}, file, indent=4)

print("Model saved to models/random_forest.pkl")
print("Metrics saved to results/metrics.json")