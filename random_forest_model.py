"""Train a Random Forest crop-yield model from crop_yield.csv.

Install dependencies: pip install pandas scikit-learn
Run: python random_forest_model.py
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

DATA = Path(__file__).with_name("crop_yield.csv")
NUMERIC = [
    "rainfall_mm", "temperature_c", "fertilizer_kg_ha",
    "nitrogen_mg_kg", "phosphorus_mg_kg", "potassium_mg_kg",
    "soil_pH", "soil_moisture_pct",
]
CATEGORICAL = ["crop_type"]
TARGET = "yield_kg_ha"


def train_model():
    data = pd.read_csv(DATA).dropna(subset=[TARGET])
    X = data[NUMERIC + CATEGORICAL]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocess = ColumnTransformer([
        ("numbers", "passthrough", NUMERIC),
        ("crop", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = make_pipeline(
        preprocess,
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(f"Rows: {len(data)} (train {len(X_train)}, test {len(X_test)})")
    print(f"MAE:  {mean_absolute_error(y_test, predicted):,.1f} kg/ha")
    print(f"RMSE: {mean_squared_error(y_test, predicted) ** 0.5:,.1f} kg/ha")
    print(f"R²:   {r2_score(y_test, predicted):.3f}")

    example = pd.DataFrame([{
        "rainfall_mm": 185,
        "temperature_c": 25,
        "fertilizer_kg_ha": 150,
        "nitrogen_mg_kg": 70,
        "phosphorus_mg_kg": 45,
        "potassium_mg_kg": 40,
        "soil_pH": 6.5,
        "soil_moisture_pct": 50,
        "crop_type": "Rice",
    }])
    estimate = model.predict(example)[0]
    print(f"Example Rice yield estimate: {estimate:,.0f} kg/ha")
    return model


if __name__ == "__main__":
    train_model()
