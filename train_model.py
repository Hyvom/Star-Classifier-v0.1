import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

df = pd.read_csv("6 class csv.csv")

feature_cols = [
    "Temperature (K)",
    "Luminosity(L/Lo)",
    "Radius(R/Ro)",
    "Absolute magnitude(Mv)",
]
X = df[feature_cols]
y = df["Star type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "star_model.joblib")
print("Model saved to star_model.joblib")

# Save metrics
metrics = {
    "accuracy": acc,
    "report": report
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Metrics saved to metrics.json")
