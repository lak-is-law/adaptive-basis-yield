"""Five-fold validation for the Unit 2 classifier and Unit 5 yield models.

Run after `python3 -m pip install -r requirements.txt`:
    python3 cross_validate_models.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FOLDS = 5


def unit2_cv():
    data = pd.read_csv(ROOT / "crop-water-stress-prediction/dataset.csv").head(5000)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    scores = {"accuracy": [], "macro_f1": []}
    for train, test in cv.split(X, y):
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
        model.fit(X.iloc[train], y.iloc[train])
        pred = model.predict(X.iloc[test])
        scores["accuracy"].append(accuracy_score(y.iloc[test], pred))
        scores["macro_f1"].append(f1_score(y.iloc[test], pred, average="macro"))
    return scores


def unit5_cv():
    data = pd.read_csv(ROOT / "adaptive-basis-yield/crop_yield.csv")
    numeric = ["rainfall_mm", "temperature_c", "fertilizer_kg_ha", "nitrogen_mg_kg",
               "phosphorus_mg_kg", "potassium_mg_kg", "soil_pH", "soil_moisture_pct"]
    target = data["yield_kg_ha"]
    cv = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    scales = {"rainfall_mm": 85, "temperature_c": 8, "fertilizer_kg_ha": 75,
              "nitrogen_mg_kg": 35, "phosphorus_mg_kg": 25, "potassium_mg_kg": 22,
              "soil_pH": 1.1, "soil_moisture_pct": 18}
    results = {name: {metric: [] for metric in ("MAE", "RMSE", "R2")}
               for name in ("adaptive_basis", "linear", "random_forest")}
    for train, test in cv.split(data):
        tr, te = data.iloc[train], data.iloc[test]
        # Keep crop-specific basis centres inside each training fold.
        def predict_basis(row):
            centres = tr[tr.crop_type == row.crop_type]
            if centres.empty:
                centres = tr
            distance = sum(((row[col] - centres[col]) / scales[col]) ** 2 for col in numeric)
            weights = np.exp(-distance / 2.6)
            return float(np.average(centres.yield_kg_ha, weights=weights))
        basis_pred = te.apply(predict_basis, axis=1).to_numpy()
        lin = LinearRegression().fit(tr[["rainfall_mm", "soil_pH", "temperature_c"]], tr.yield_kg_ha)
        linear_pred = lin.predict(te[["rainfall_mm", "soil_pH", "temperature_c"]])
        encoder = ColumnTransformer([("crop", OneHotEncoder(handle_unknown="ignore"), ["crop_type"]),
                                     ("numeric", "passthrough", numeric)])
        forest = make_pipeline(encoder, RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                             max_features=0.8, random_state=SEED, n_jobs=-1))
        forest.fit(tr[numeric + ["crop_type"]], tr.yield_kg_ha)
        forest_pred = forest.predict(te[numeric + ["crop_type"]])
        for name, pred in (("adaptive_basis", basis_pred), ("linear", linear_pred),
                           ("random_forest", forest_pred)):
            vals = results[name]
            vals["MAE"].append(mean_absolute_error(te.yield_kg_ha, pred))
            vals["RMSE"].append(np.sqrt(mean_squared_error(te.yield_kg_ha, pred)))
            vals["R2"].append(r2_score(te.yield_kg_ha, pred))
    return results


def summarize(values):
    return {metric: f"{np.mean(scores):.3f} ± {np.std(scores, ddof=1):.3f}"
            for metric, scores in values.items()}


if __name__ == "__main__":
    print(f"Seed={SEED}; folds={FOLDS}; folds shuffled; Unit 2 stratified")
    u2, u5 = unit2_cv(), unit5_cv()
    print("Unit 2 (first 5,000 rows):")
    for key, values in u2.items():
        print(f"  {key}: {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}")
    print("Unit 5 (kg/ha except R²):")
    for key, values in u5.items():
        print(f"  {key}: {summarize(values)}")
