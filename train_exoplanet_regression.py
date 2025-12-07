import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("cleaned_5250.csv")

# 2. Select features and target
feature_cols = [
    "distance",
    "stellar_magnitude",
    "orbital_radius",
    "orbital_period",
    "eccentricity",
    "discovery_year",
]

target_col = "mass_multiplier"

# Keep only rows with all needed values
df_model = df[feature_cols + [target_col]].dropna()

X = df_model[feature_cols]
y = df_model[target_col]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model: Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R^2:", r2)

# 6. Save model
joblib.dump(model, "exoplanet_model.joblib")
print("Model saved to exoplanet_model.joblib")

# 7. Save metrics to JSON (for API)
import json
metrics = {
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "r2": r2,
}
with open("exoplanet_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Metrics saved to exoplanet_metrics.json")

# 8. Diagrams

# True vs predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
plt.xlabel("True Mass Multiplier")
plt.ylabel("Predicted Mass Multiplier")
plt.title("Exoplanet Mass: True vs Predicted")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
plt.legend()
plt.tight_layout()
plt.savefig("exoplanet_true_vs_pred.png", dpi=150)
plt.close()

# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 5))
plt.scatter(y_pred, residuals, alpha=0.6, edgecolor="k")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Mass Multiplier")
plt.ylabel("Residual (True - Predicted)")
plt.title("Exoplanet Mass Residuals")
plt.tight_layout()
plt.savefig("exoplanet_residuals.png", dpi=150)
plt.close()

print("Charts saved: exoplanet_true_vs_pred.png, exoplanet_residuals.png")