import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import joblib
import json
from sklearn.tree import plot_tree

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

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.xlabel("Predicted type")
plt.ylabel("True type")
plt.title("Star Type Confusion Matrix")
plt.tight_layout()
plt.savefig("star_confusion_matrix.png", dpi=150)
plt.close()

# Simple HR‑style scatter: Temperature vs Luminosity colored by type
plt.figure(figsize=(5, 4))
scatter = plt.scatter(
    X["Temperature (K)"],
    X["Luminosity(L/Lo)"],
    c=y,
    cmap="viridis",
    s=25,
    alpha=0.8
)
cbar = plt.colorbar(scatter)
cbar.set_label("Star type")
plt.gca().invert_xaxis()  # hot on the left like real HR diagram
plt.xlabel("Temperature (K)")
plt.ylabel("Luminosity (L/Lo)")
plt.title("Stars in Temperature–Luminosity Space")
plt.tight_layout()
plt.savefig("star_hr_scatter.png", dpi=150)
plt.close()
plt.figure(figsize=(14, 8))
plot_tree(model, feature_names=feature_cols, class_names=[str(c) for c in sorted(y.unique())],
          filled=True, max_depth=3)
plt.title("Decision Tree (first 3 levels)")
plt.tight_layout()
plt.savefig("star_decision_tree.png", dpi=150)
plt.close()

# Save metrics
metrics = {
    "accuracy": acc,
    "report": report
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Metrics saved to metrics.json")
