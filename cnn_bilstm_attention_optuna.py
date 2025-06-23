import pandas as pd
import numpy as np
import optuna
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Bidirectional,
    LSTM,
    Dense,
    Attention,
    LayerNormalization,
    Add,
    GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Nadam

import matplotlib.pyplot as plt

# === Load and Preprocess Data ===
df = pd.read_csv("final_merged.csv")
df.rename(
    columns={
        "Temp2m": "T2M",
        "Humidity2m": "RH2M",
        "SolarZenith": "SZA",
        "Forecasted_SW_DWN": "Forecasted",
        "SolarIrradiance": "SolarOutput",
        "CloudCover": "Cld_Amt",
        "DewPoint2m": "D2M",
    },
    inplace=True,
)

X = df[["MO", "DY", "HR", "T2M", "RH2M", "SZA", "Cld_Amt", "D2M"]]
y = df["SolarOutput"]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

joblib.dump(scaler_X, "scaler_X.save")
joblib.dump(scaler_y, "scaler_y.save")


# === Sequence Builder (24-hour blocks) ===
def create_daily_blocks(X, y, block_size=24):
    X_seq, y_seq = [], []
    for i in range(0, len(X) - block_size):
        X_block = X[i : i + block_size]
        y_block = y[i : i + block_size]
        if len(X_block) == 24 and len(y_block) == 24:
            X_seq.append(X_block)
            y_seq.append(y_block.flatten())
    return np.array(X_seq), np.array(y_seq)


# === Model Architecture ===
def build_model(input_shape):
    inp = Input(shape=input_shape)  # shape = (24, 8)
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(inp)
    x = LayerNormalization()(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = LayerNormalization()(x)
    attn_output = Attention()([x, x])
    x = Add()([x, attn_output])
    x = GlobalAveragePooling1D()(x)
    out = Dense(24)(x)  # Predict 24 hours
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Nadam(), loss="mse")
    return model


# === Optuna Tuning (fast: 8 trials only) ===
def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    X_seq, y_seq = create_daily_blocks(X_scaled, y_scaled, 24)
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    model = build_model((24, X.shape[1]))
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=batch_size,
        verbose=0,
    )
    return history.history["val_loss"][-1]


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=8)

# === Final Training ===
best_batch = study.best_params["batch_size"]
print(f"\nBest Batch Size: {best_batch}")

X_seq, y_seq = create_daily_blocks(X_scaled, y_scaled, 24)
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

final_model = build_model((24, X.shape[1]))
final_model.fit(X_train, y_train, epochs=20, batch_size=best_batch, verbose=1)

# === Evaluation ===
y_pred_scaled = final_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print(f"\nMAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

# === Plot First Sample ===
plt.figure(figsize=(12, 5))
plt.plot(y_actual[0], label="Actual")
plt.plot(y_pred[0], label="Predicted")
plt.title("24-Hour Solar Irradiance Prediction")
plt.xlabel("Hour")
plt.ylabel("Irradiance (Wh/m²)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Save Final Model ===
final_model.save("cnn_bilstm_attention_24hr_model.keras")
print("\nModel and scalers saved successfully.")
