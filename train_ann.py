
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_excel("data/Cooling Load Dataset.xlsx")
X = df.iloc[:, :8].values
y = df[["Y1", "Y2"]].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(2, activation="linear")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=40,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=500,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

predictions = model.predict(X_test, verbose=0)

print("\n========== FINAL TEST RESULTS ==========")

for i, name in enumerate(["Heating Load", "Cooling Load"]):
    actual = y_test[:, i]
    predicted = predictions[:, i]

    print(f"\n{name}")
    print(f"R2 Score: {r2_score(actual, predicted):.4f}")
    print(f"MAE: {mean_absolute_error(actual, predicted):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(actual, predicted)):.4f}")

model.save("building_energy_ann.keras")
joblib.dump(scaler, "feature_scaler.pkl")

results = pd.DataFrame({
    "Actual_Heating_Load": y_test[:, 0],
    "Predicted_Heating_Load": predictions[:, 0],
    "Actual_Cooling_Load": y_test[:, 1],
    "Predicted_Cooling_Load": predictions[:, 1]
})

results.to_csv("test_predictions.csv", index=False)

print("\nSUCCESS! Model and prediction files saved.")
