import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame
df['PRICE'] = df.pop('MedHouseVal')

X = df.drop(columns=['PRICE'])
y = df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestRegressor(
    max_depth=30, min_samples_leaf=5, min_samples_split=10, n_estimators=500, random_state=42
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

xgb_model = XGBRegressor(max_depth=30, learning_rate=0.1, n_estimators=500, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

def build_nn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')  
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

nn_model = build_nn_model(X_train_scaled.shape[1])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = nn_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

nn_preds = nn_model.predict(X_test_scaled).flatten()

rf_r2 = r2_score(y_test, rf_preds)
rf_mse = mean_squared_error(y_test, rf_preds)
xgb_r2 = r2_score(y_test, xgb_preds)
xgb_mse = mean_squared_error(y_test, xgb_preds)
nn_r2 = r2_score(y_test, nn_preds)
nn_mse = mean_squared_error(y_test, nn_preds)

print(f"Random Forest - R2: {rf_r2:.2f}, MSE: {rf_mse:.2f}")
print(f"XGBoost - R2: {xgb_r2:.2f}, MSE: {xgb_mse:.2f}")
print(f"Neural Network - R2: {nn_r2:.2f}, MSE: {nn_mse:.2f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.scatter(y_test, rf_preds, alpha=0.5)
plt.plot([0, 5], [0, 5], color='red', linestyle='--')
plt.title("Random Forest: Actual vs Predicted")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.subplot(1, 3, 2)
plt.scatter(y_test, xgb_preds, alpha=0.5)
plt.plot([0, 5], [0, 5], color='red', linestyle='--')
plt.title("XGBoost: Actual vs Predicted")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.subplot(1, 3, 3)
plt.scatter(y_test, nn_preds, alpha=0.5)
plt.plot([0, 5], [0, 5], color='red', linestyle='--')
plt.title("Neural Network: Actual vs Predicted")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.tight_layout()
plt.show()
