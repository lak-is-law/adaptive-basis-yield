# Adaptive Basis Function Model

Static, GitHub Pages ready crop yield explorer. It estimates yield using a Gaussian radial basis function model: each field observation contributes a local basis response, weighted by similarity across rainfall, temperature, NPK, soil pH, and moisture. Local basis responses let relationships vary across low, medium, and high input ranges.

## Run locally

Open `index.html` through a local HTTP server (CSV fetch is blocked by some browsers for `file://` URLs):

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`. On GitHub, enable Pages from the repository's `main` branch, root folder. The included workflow also publishes automatically on pushes to `main`.

## Python baseline models

The repository also includes two command-line baselines that use the same CSV. Python 3.10+ is recommended.

```bash
python3 -m pip install -r requirements.txt
python3 linear_regression_model.py
python3 random_forest_model.py
python3 cross_validate_models.py
```

The linear model uses rainfall, soil pH, and temperature. The Random Forest uses the numeric soil/weather/NPK fields plus crop type. Both print holdout MAE, RMSE, and R². The five-fold validation script evaluates all three Unit 5 candidates on identical shuffled folds and prints mean ± standard deviation.

## Data

`crop_yield.csv` is included in this folder (945 rows, 10 crop labels). The related Kaggle listing is [Smart Farming Sensor Data for Yield Prediction by Atharva Soundankar](https://www.kaggle.com/datasets/atharvasoundankar/smart-farming-sensor-data-for-yield-prediction). The local CSV was already present in the workspace and is not a verified raw Kaggle download; confirm exact row-level provenance and license against Kaggle before treating it as source data or redistributing it. The source fields available here include rainfall, temperature, soil pH/moisture, N, P, K, and yield, but not applied fertilizer. `fertilizer_kg_ha` is an N + P + K proxy, never a measured application rate. This dataset is a small demo sample, not a vast corpus; predictions need local validation before agronomic decisions.

## Project files

- `index.html`, `styles.css`, `app.js` — static responsive interface and in-browser model
- `crop_yield.csv` — packaged dataset
- `linear_regression_model.py`, `random_forest_model.py`, `cross_validate_models.py` — reproducible Python models and five-fold evaluation
- `requirements.txt` — pandas and scikit-learn dependencies for the baselines
- `.github/workflows/pages.yml` — GitHub Pages deployment workflow
