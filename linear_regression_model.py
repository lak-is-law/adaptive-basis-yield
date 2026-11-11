"""Simple crop-yield linear regression using rainfall, soil pH, and temperature.

Install: pip install pandas scikit-learn
Run:     python linear_regression_model.py
"""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA = Path(__file__).with_name("crop_yield.csv")
FEATURES = ["rainfall_mm", "soil_pH", "temperature_c"]
TARGET = "yield_kg_ha"


def main():
    data = pd.read_csv(DATA).dropna(subset=FEATURES + [TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=42
    )

    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Linear Regression: Yield ~ Rainfall + Soil pH + Temperature")
    print(f"Intercept: {model.intercept_:.2f}")
    for feature, coefficient in zip(FEATURES, model.coef_):
        print(f"{feature}: {coefficient:+.2f} kg/ha per unit")
    print(f"\nTest MAE:  {mean_absolute_error(y_test, predictions):,.1f} kg/ha")
    print(f"Test RMSE: {mean_squared_error(y_test, predictions) ** 0.5:,.1f} kg/ha")
    print(f"Test R²:   {r2_score(y_test, predictions):.3f}")

    example = pd.DataFrame([{
        "rainfall_mm": 185,
        "soil_pH": 6.5,
        "temperature_c": 25,
    }])
    estimate = model.predict(example)[0]
    print(f"\nExample estimate (rainfall 185 mm, pH 6.5, temperature 25°C): "
          f"{estimate:,.0f} kg/ha")


if __name__ == "__main__":
    main()
