from gettext import install

import pandas as pd
import pip
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_csv("minna_solar_data.csv", index_col=0, parse_dates=True)

# Add time-based features
df["month"] = df.index.month
df["day_of_year"] = df.index.dayofyear

# Define features (X) and target (y)
features = ["CLRSKY_SFC_SW_DWN", "T2M", "RH2M", "CLOUD_AMT", "WS2M", "PRECTOTCORR", "month", "day_of_year"]
X = df[features]
y = df["ALLSKY_SFC_SW_DWN"]

# Use last 2 years as test set, rest as training
split_date = "2023-01-01"
X_train, X_test = X[df.index < split_date], X[df.index >= split_date]
y_train, y_test = y[df.index < split_date], y[df.index >= split_date]

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"MAE: {mae:.3f} kWh/m²/day")
print(f"RMSE: {rmse:.3f} kWh/m²/day")

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# R² score (your "accuracy" equivalent for regression)
r2 = r2_score(y_test, predictions)
print(f"R² Score: {r2:.3f}")

# Plot 1: Predicted vs Actual over time
plt.figure(figsize=(14, 5))
plt.plot(y_test.index, y_test.values, label="Actual", linewidth=1)
plt.plot(y_test.index, predictions, label="Predicted", linewidth=1, alpha=0.7)
plt.title("Actual vs Predicted Solar Irradiance (Test Set: 2023–2024)")
plt.xlabel("Date")
plt.ylabel("ALLSKY_SFC_SW_DWN (kWh/m²/day)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Plot 2: Scatter plot (perfect predictions would fall on the diagonal line)
plt.figure(figsize=(7, 7))
plt.scatter(y_test, predictions, alpha=0.4, s=15)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect prediction")
plt.xlabel("Actual Irradiance (kWh/m²/day)")
plt.ylabel("Predicted Irradiance (kWh/m²/day)")
plt.title(f"Predicted vs Actual (R² = {r2:.3f})")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

import pandas as pd

importances = pd.Series(model.feature_importances_, index=features)
print(importances.sort_values(ascending=False))

df["ALLSKY_lag1"] = df["ALLSKY_SFC_SW_DWN"].shift(1)
df["CLOUD_AMT_lag1"] = df["CLOUD_AMT"].shift(1)
df = df.dropna()  # first row will have NaN from the shift


from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_r2 = r2_score(y_test, xgb_predictions)

print(f"XGBoost — MAE: {xgb_mae:.3f}, RMSE: {xgb_rmse:.3f}, R²: {xgb_r2:.3f}")

