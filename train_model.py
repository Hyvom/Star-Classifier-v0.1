import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load data
df = pd.read_csv("6 class csv.csv")

# 2. Features and target
# Use numeric physical features as X and "Star type" as y
feature_cols = [
    "Temperature (K)",
    "Luminosity(L/Lo)",
    "Radius(R/Ro)",
    "Absolute magnitude(Mv)",
]
X = df[feature_cols]
y = df["Star type"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    random_state=42,
)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Save model
joblib.dump(model, "star_model.joblib")
print("Model saved to star_model.joblib")
