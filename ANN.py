import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

df = pd.read_csv("final_merged.csv")

df.rename(
    columns={
        "Temp2m": "T2M",
        "Humidity2m": "RH2M",
        "SolarZenith": "SZA",
        "Forecasted_SW_DWN": "Forecasted",
    },
    inplace=True,
)

X = df[["MO", "DY", "HR", "T2M", "RH2M", "SZA"]]
y = df["Forecasted"]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)


model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(X.shape[1],)))
model.add(Dense(64, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=30, batch_size=32)

y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = y.values

mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

model.save("solar_irradiance_ann_model.keras")

df["Predicted"] = y_pred
df.to_csv("predictions_vs_actual.csv", index=False)

hours = df["HR"].values
days = df.index // 24

plt.figure(figsize=(12, 5))
plt.plot(
    df.index, df["Forecasted"], label="Original Forecast", color="orange", linewidth=1
)
plt.plot(
    df.index,
    df["Predicted"],
    label="Predicted (ANN)",
    color="green",
    alpha=0.7,
    linewidth=1,
)
plt.xlabel("Time (Hours)")
plt.ylabel("Irradiance (Wh/m²)")
plt.title("Original vs ANN Predicted Solar Irradiance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


errors = np.abs(df["Forecasted"].values - df["Predicted"].values)
valid_len = (len(errors) // 24) * 24
daily_error = errors[:valid_len].reshape(-1, 24).mean(axis=1)

plt.figure(figsize=(10, 4))
plt.plot(daily_error, color="red", label="Daily MAE")
plt.xlabel("Days")
plt.ylabel("Mean Absolute Error (Wh/m²)")
plt.title("Daily Mean Absolute Error Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
