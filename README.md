# Real Estate Price Assessment

I built a multimodal pipeline that predicts apartment prices (EUR) by fusing structured property attributes with visual features from listing images. The project covers end‑to‑end data preparation, standalone tabular and image models, multiple fusion strategies (simple averaging, voting, stacking), and exports artifacts (predictions, checkpoints, and reports) for reproducibility.

- Languages: Jupyter Notebook, Python
- Data modalities: tabular features + room images (bathroom/kitchen/rooms/terrace)
- Models: XGBoost (tabular), ResNet (image regression), fusion (averaging, voting, stacking)

---

## Goals

- Produce price estimates from both metadata and photos.
- Compare single‑modality models to fused models.
- Persist best hyperparameters and trained artifacts for reuse.
- Make the pipeline reproducible and auditable through saved predictions and JSON reports.

---

## Repository Structure

| Path / File | Purpose |
| --- | --- |
| property_data.csv | Raw dataset with IDs and mixed categorical/numeric attributes. |
| filtered_property_data.csv | Cleaned dataset used for training (duplicates removed, critical NAs resolved). |
| pretprocesiranje/ | Preprocessing assets (logical staging). |
| property_images/ | Root of images; subfolders: kupatilo (bath), kuhinja (kitchen), sobe (rooms), terasa (terrace). |
| XGboost_predikcija_karakteristike.py | Train/tune XGBoost on tabular features; save best params and predictions. |
| best_params_xgboost_random_search.pkl | Saved best XGBoost hyperparameters (joblib). |
| xgb_predictions.csv | XGBoost predictions on held‑out data. |
| image_price_model.py | Full image regression pipeline (ResNet18/ResNet50), training, validation, predictions, residuals, stats, checkpoints. |
| checkpoint.pth, checkpoint.pth.tar, checkpoint_image_price.pth | PyTorch checkpoints for image models. |
| combined_model.py | Simple model fusion: XGBoost (tabular) + ResNet18 (image) via averaging. |
| kombinovani_model_voting.py | Voting ensemble over multiple regressors (serialized to voting_regressor.pkl). |
| stack_fusion.py | Stacking fusion (meta‑learner over base model predictions). |
| voting_regressor.pkl | Serialized voting ensemble. |
| image_price_predictions*.csv | Image‑only price predictions (variants for ResNet18/ResNet50 and price heads). |
| image_residuals_resnet18.csv | Residuals for image model analysis. |
| image_price_stats.json | Summary statistics for the image price model. |
| fused_predictions*.csv | Fused outputs (averaging/voting/stacking variants across ResNet18/ResNet50). |
| fusion_report*.json | JSON reports summarizing fusion configuration and metrics (for each experiment). |
| model_predikcije_slike.ipynb | Notebook for image‑only experiments and quick iteration. |
| docs/, Presentation (1).pptx, Procena cene stanova.pdf | Documentation artifacts (slides and PDF write‑up). |
| .vscode/, models/, klasifikacija slika/, nasi stanovi/, ostale procene/ | Development config, auxiliary models and outputs, categorized experiments, and additional appraisals. |

---

## Data and Features

Core feature columns:

- kvadratura (area m²)
- grad, opstina, kvart (city, municipality, neighborhood)
- broj_soba (rooms), spratnost (floor)
- stanje (condition), grejanje (heating)
- lift, podrum (basement)
- cena (target, EUR)

Processing steps:

1. Remove duplicates; drop rows missing crucial fields (kvadratura, cena).
2. Extract and keep ID only for image linking; drop ID from model feature matrix.
3. Label‑encode categorical columns (fill NAs with "missing").
4. Standard‑scale selected features.
5. Standard‑scale target (cena) for the image regression head; invert scale on output when needed.

Image conventions:

- Subfolders: kupatilo, kuhinja, sobe, terasa.
- Filenames start with the property ID followed by underscore (ID mapping back to CSV).
- Transforms: resize → center crop (512) → tensor → ImageNet normalization.

---

## Models

### XGBoost (tabular)
- Random search tuning; best params stored in best_params_xgboost_random_search.pkl.
- Trained on selected/encoded/scaled tabular features.
- Predictions saved to xgb_predictions.csv.

### ResNet18 / ResNet50 (image regression)
- Pretrained ImageNet backbones; backbone frozen; final FC replaced with a single linear output (price).
- Early stopping utility available (early_stopping.py) for longer training loops.
- Checkpoints saved to checkpoint.pth/.tar or checkpoint_image_price.pth.
- Predictions exported to image_price_predictions*.csv; residuals (image_residuals_resnet18.csv); summary stats (image_price_stats.json).

### Fusion

1. Simple averaging (combined_model.py)
   - Predict with XGBoost and image model, inverse‑scale image outputs, then average.
   - Outputs used for scatter diagnostics and comparison.

2. Voting ensemble (kombinovani_model_voting.py)
   - Ensemble over multiple base regressors (including XGBoost).
   - Serialized to voting_regressor.pkl; produces fused_predictions*.csv.

3. Stacking (stack_fusion.py)
   - Meta‑learner trained on base model predictions.
   - Produces fused_resnet50_experiment.csv, fused_resnet50_ridge.csv, etc., with corresponding JSON reports (fusion_report*.json).

---

## Installation

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost joblib matplotlib pillow
pip install torch torchvision  # match CUDA version if using GPU
pip install jupyter            # for the notebook
```

---

## Running the Pipelines

### 1) XGBoost (tabular)

```bash
python XGboost_predikcija_karakteristike.py
```

Produces:
- best_params_xgboost_random_search.pkl
- xgb_predictions.csv

### 2) Image model (ResNet18/50)

```bash
python image_price_model.py
```

Produces:
- checkpoint_image_price.pth (and/or checkpoint.pth/.tar)
- image_price_predictions*.csv
- image_residuals_resnet18.csv
- image_price_stats.json

Expected structure for images:
```
property_images/
  kupatilo/
    <id>_*.jpg
  kuhinja/
    <id>_*.jpg
  sobe/
    <id>_*.jpg
  terasa/
    <id>_*.jpg
```

### 3) Simple fusion (averaging)

```bash
python combined_model.py
```

Loads:
- filtered_property_data.csv
- best_params_xgboost_random_search.pkl
- image checkpoints (if present)

Performs averaging XGBoost + ResNet18; shows scatter plot and can export predictions if configured.

### 4) Voting ensemble

```bash
python kombinovani_model_voting.py
```

Produces:
- voting_regressor.pkl
- fused_predictions*.csv
- fusion_report*.json

### 5) Stacking fusion

```bash
python stack_fusion.py
```

Produces:
- fused_resnet50_experiment.csv / fused_resnet50_ridge.csv (and similar)
- fusion_report_resnet50*.json (and variants)
- fused_predictions_price_resnet50.csv / fused_predictions_resnet18_price.csv (when applicable)

---

## Evaluation

I rely on visual diagnostics (Actual vs Predicted scatter) and exported metrics in the JSON reports:

- fusion_report*.json: fusion settings and metrics for each experiment/variant.
- image_price_stats.json: summary stats for image‑only price model.
- Residual analysis: image_residuals_resnet18.csv.

To compute additional metrics on the fly:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)
print(mae, rmse, r2)
```

Note: When the image head predicts on a scaled target, inverse transform is applied prior to evaluation.

---

## Reproducibility

- Random seeds set where applicable (e.g., random_state=42).
- Best hyperparameters and trained artifacts are saved for reuse (joblib/pkl for tabular models, .pth/.tar for PyTorch).
- Filenames assume the first token before underscore is the numeric/string property ID that matches the CSV.

---

## Quick Start

```bash
# 0) Prepare environment and data
pip install -r <see Installation>
# Ensure property_images/ and filtered_property_data.csv are in place

# 1) Train/tune XGBoost (tabular)
python XGboost_predikcija_karakteristike.py

# 2) Train image price model (ResNet)
python image_price_model.py

# 3) Fuse predictions
python combined_model.py          # averaging
python kombinovani_model_voting.py
python stack_fusion.py
```

Artifacts appear as:
- xgb_predictions.csv
- image_price_predictions*.csv
- fused_predictions*.csv
- fusion_report*.json
- checkpoints (.pth/.tar), best_params_xgboost_random_search.pkl, voting_regressor.pkl

---

## Acknowledgments

This work builds on standard libraries (PyTorch, torchvision, XGBoost, scikit‑learn) and ImageNet pretraining for transfer learning.
