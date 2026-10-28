
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("test_predictions.csv")

targets = [
    ("Heating Load", "Actual_Heating_Load", "Predicted_Heating_Load"),
    ("Cooling Load", "Actual_Cooling_Load", "Predicted_Cooling_Load")
]

for name, actual_col, predicted_col in targets:
    actual = df[actual_col]
    predicted = df[predicted_col]
    errors = actual - predicted

    # Graph 1: Actual versus predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.75)

    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    plt.plot([lower, upper], [lower, upper], "r--",
             label="Perfect prediction")

    plt.xlabel("Actual load")
    plt.ylabel("Predicted load")
    plt.title(f"ANN: Actual vs Predicted {name}")
    plt.legend()
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(f"{filename}_actual_vs_predicted.png", dpi=300)
    plt.show()
    plt.close()

    # Graph 2: Prediction errors
    plt.figure(figsize=(7, 6))
    plt.scatter(predicted, errors, alpha=0.75)
    plt.axhline(0, color="red", linestyle="--")

    plt.xlabel("Predicted load")
    plt.ylabel("Actual - Predicted")
    plt.title(f"ANN: Prediction Errors for {name}")
    plt.tight_layout()

    plt.savefig(f"{filename}_residuals.png", dpi=300)
    plt.show()
    plt.close()

print("All four graphs have been saved successfully.")
