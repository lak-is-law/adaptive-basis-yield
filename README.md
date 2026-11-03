# Adaptive Basis Function Model

Static, GitHub Pages ready crop yield explorer. It estimates yield using a Gaussian radial basis function model: each field observation contributes a local basis response, weighted by similarity across rainfall, temperature, NPK, soil pH, and moisture. Local basis responses let relationships vary across low, medium, and high input ranges.

## Run locally

Open `index.html` through a local HTTP server (CSV fetch is blocked by some browsers for `file://` URLs):

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`. On GitHub, enable Pages from the repository's `main` branch, root folder. The included workflow also publishes automatically on pushes to `main`.

## Data

`crop_yield.csv` is included in this folder (945 observations). It contains climate, soil, nutrients, crop type, and yield fields from the workspace's Kaggle referenced agriculture sensor dataset. The source records soil nitrogen, phosphorus, and potassium but not applied fertilizer; `fertilizer_kg_ha` is a proxy (N + P + K) and is never a measured application rate. The CSV is included as requested; verify the upstream dataset's exact attribution and license before redistribution. The dataset is modest, not a large corpus, and predictions should be validated with local field data before agronomic decisions.

## Project files

- `index.html`, `styles.css`, `app.js` — static responsive interface and in-browser model
- `crop_yield.csv` — packaged dataset
- `.github/workflows/pages.yml` — GitHub Pages deployment workflow
