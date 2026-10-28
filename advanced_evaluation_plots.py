
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "advanced_results"
PREDICTIONS = RESULTS / "advanced_predictions.csv"
COMPARISON = RESULTS / "advanced_model_comparison.csv"
PLOT_DIR = RESULTS / "evaluation_plots"
PLOT_DIR.mkdir(exist_ok=True)

# 1. Load saved predictions; no model training required
df = pd.read_csv(PREDICTIONS)
comparison = pd.read_csv(COMPARISON)

models = ["Baseline_ANN", "Deep_ANN", "Regularized_ANN"]
targets = {
    "Heating": ("Actual_Heating_Load", "Heating load"),
    "Cooling": ("Actual_Cooling_Load", "Cooling load")
}

# 2. Actual versus predicted plots for the Deep ANN
for key, (actual_col, title) in targets.items():
    pred_col = f"Deep_ANN_{key}_Prediction"

    actual = df[actual_col]
    predicted = df[pred_col]

    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.75)
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel(f"Actual {title}")
    plt.ylabel(f"Predicted {title}")
    plt.title(f"Deep ANN: Actual vs Predicted {title}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"deep_ann_actual_vs_predicted_{key.lower()}.png",
                dpi=300)
    plt.close()

# 3. Residual plots for the Deep ANN
for key, (actual_col, title) in targets.items():
    pred_col = f"Deep_ANN_{key}_Prediction"
    actual = df[actual_col].to_numpy()
    predicted = df[pred_col].to_numpy()
    residuals = actual - predicted

    plt.figure(figsize=(7, 5))
    plt.scatter(predicted, residuals, alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel(f"Predicted {title}")
    plt.ylabel("Residual (actual - predicted)")
    plt.title(f"Deep ANN: Residual Analysis - {title}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"deep_ann_residuals_{key.lower()}.png",
                dpi=300)
    plt.close()

# 4. Compare models using R²
comparison["Model"] = pd.Categorical(
    comparison["Model"], categories=models, ordered=True
)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, target, title in zip(
    axes, ["Y1", "Y2"], ["Heating Load", "Cooling Load"]
):
    subset = comparison[comparison["Target"] == target].sort_values("Model")
    ax.bar(subset["Model"].astype(str), subset["R2"])
    ax.set_title(f"Model Comparison: {title}")
    ax.set_ylabel("R² score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig(PLOT_DIR / "model_r2_comparison.png", dpi=300)
plt.close()

# 5. Recalculate test-set metrics from saved predictions
rows = []
for model in models:
    for key, (actual_col, title) in targets.items():
        pred_col = f"{model}_{key}_Prediction"
        actual = df[actual_col]
        predicted = df[pred_col]

        rows.append({
            "Model": model,
            "Target": title,
            "R2": r2_score(actual, predicted),
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted))
        })

metrics = pd.DataFrame(rows)
metrics.to_csv(PLOT_DIR / "verified_prediction_metrics.csv", index=False)

print("\n========== TEST SET METRICS ==========")
print(metrics.round(4).to_string(index=False))
print(f"\nEvaluation graphs and metrics saved in:\n{PLOT_DIR}")
