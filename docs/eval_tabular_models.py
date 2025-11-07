import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


DEFAULT_FEATURES = [
    "kvadratura",
    "grad",
    "opstina",
    "kvart",
    "broj_soba",
    "spratnost",
    "grejanje",
    "lift",
    "podrum",
]

DEFAULT_CAT = [
    "grad",
    "opstina",
    "kvart",
    "broj_soba",
    "spratnost",
    "stanje",
    "grejanje",
    "lift",
    "podrum",
    "terasa",
]


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(a), eps)
    return float(np.mean(np.abs(a - b) / denom) * 100.0)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    # Require target and a core numeric feature
    df = df.dropna(subset=[c for c in ["kvadratura", "cena"] if c in df.columns])
    # Normalize id to string when present
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    return df.reset_index(drop=True)


def choose_features(df: pd.DataFrame, requested: List[str]) -> List[str]:
    # keep only available features, prefer requested order
    available = [c for c in requested if c in df.columns]
    # ensure target is not in features
    return [c for c in available if c != "cena" and c != "id"]


def kfold_eval(
    df: pd.DataFrame,
    features: List[str],
    cat_cols: List[str],
    model_name: str,
    model_params: Dict,
    k: int = 5,
    seed: int = 42,
    out_pred_csv: str = None,
) -> Tuple[Dict, pd.DataFrame]:
    """Run KFold with per-fold fit of encoders/scalers for fair comparison.
    Returns metrics dict and predictions dataframe (id,y_true,y_pred)."""
    y_all: List[float] = []
    p_all: List[float] = []
    id_all: List[str] = []

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    X_df = df.copy()
    y = df["cena"].to_numpy()

    # Track fold metrics
    fold_mae: List[float] = []
    fold_mse: List[float] = []

    for tr_idx, te_idx in kf.split(X_df):
        tr = X_df.iloc[tr_idx].copy()
        te = X_df.iloc[te_idx].copy()

        # Encode categorical cols fold-wise with UNK handling
        for c in cat_cols:
            if c in tr.columns:
                tr_vals = tr[c].fillna("missing").astype(str)
                te_vals = te[c].fillna("missing").astype(str)
                cats = sorted(tr_vals.unique().tolist())
                if "__UNK__" not in cats:
                    cats.append("__UNK__")
                tr_codes = pd.Categorical(tr_vals, categories=cats).codes
                te_codes = pd.Categorical(te_vals, categories=cats).codes
                # Replace unseen (-1) with UNK index
                unk_idx = cats.index("__UNK__")
                te_codes = np.where(te_codes == -1, unk_idx, te_codes)
                tr[c] = tr_codes
                te[c] = te_codes

        # Ensure features subset exists after encoding
        feats = [f for f in features if f in tr.columns]
        if not feats:
            raise ValueError("No valid features found after preprocessing.")

        # Scale numeric-like features (including encoded) fold-wise to avoid leakage
        scaler = StandardScaler()
        tr_X = scaler.fit_transform(tr[feats])
        te_X = scaler.transform(te[feats])
        tr_y = tr["cena"].to_numpy()
        te_y = te["cena"].to_numpy()

        # Build model
        if model_name == "xgb":
            if xgb is None:
                raise RuntimeError("xgboost is not installed in this environment")
            model = xgb.XGBRegressor(random_state=seed, **model_params)
        elif model_name == "rf":
            model = RandomForestRegressor(random_state=seed, **model_params)
        elif model_name == "gbrt":
            model = GradientBoostingRegressor(random_state=seed, **model_params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(tr_X, tr_y)
        te_p = model.predict(te_X)

        fold_mae.append(mean_absolute_error(te_y, te_p))
        fold_mse.append(mean_squared_error(te_y, te_p))

        y_all.extend(te_y.tolist())
        p_all.extend(te_p.tolist())
        if "id" in te.columns:
            id_all.extend(te["id"].astype(str).tolist())
        else:
            # fallback to global row indices for traceability
            id_all.extend([str(i) for i in te.index.tolist()])

    y_all_np = np.array(y_all, dtype=float)
    p_all_np = np.array(p_all, dtype=float)

    metrics = {
        "model": model_name,
        "params": model_params,
        "n": int(len(y_all_np)),
        "mae": mae(y_all_np, p_all_np),
        "rmse": rmse(y_all_np, p_all_np),
        "mape": mape(y_all_np, p_all_np),
        "mae_cv_mean": float(np.mean(fold_mae)),
        "mae_cv_std": float(np.std(fold_mae)),
    }

    preds_df = pd.DataFrame({
        "id": id_all,
        "y_true_eur": y_all,
        "y_pred_eur": p_all,
    })
    if out_pred_csv:
        preds_df.to_csv(out_pred_csv, index=False)

    return metrics, preds_df


def main():
    ap = argparse.ArgumentParser(description="Evaluate tabular models (GBRT, RF, XGB) with consistent KFold and logging")
    ap.add_argument("--data_csv", type=str, default="filtered_property_data.csv")
    ap.add_argument("--models", type=str, default="all", help="Comma-separated: xgb,rf,gbrt or 'all'")
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="docs/reports")
    ap.add_argument("--use_default_features", action="store_true", help="Use default feature list; otherwise intersect with dataset")
    ap.add_argument("--features", type=str, default=None, help="Comma-separated custom feature list to use (overrides defaults)")
    ap.add_argument("--ids_csv", type=str, default=None, help="Optional CSV with a column 'id' to filter data to these ids")
    # Optional overrides for params
    ap.add_argument("--xgb_params_json", type=str, default=None, help="Path to JSON with XGB params; if omitted, tries to load joblib best_params files or uses sensible defaults")
    ap.add_argument("--rf_params_json", type=str, default=None)
    ap.add_argument("--gbrt_params_json", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_data(args.data_csv)
    # Optional filter by ids for apples-to-apples subsets
    if args.ids_csv and os.path.exists(args.ids_csv):
        try:
            ids_df = pd.read_csv(args.ids_csv)
            if "id" in ids_df.columns and "id" in df.columns:
                keep = set(ids_df["id"].astype(str).unique().tolist())
                df = df[df["id"].astype(str).isin(keep)].reset_index(drop=True)
                print(f"[INFO] Filtered data by ids_csv -> rows: {len(df)}")
        except Exception as e:
            print(f"[WARN] Failed to filter by ids_csv: {e}")

    # Feature selection
    if args.features:
        requested = [c.strip() for c in args.features.split(',') if c.strip()]
        feats = choose_features(df, requested)
    else:
        feats = choose_features(df, DEFAULT_FEATURES) if args.use_default_features else choose_features(df, list(set(DEFAULT_FEATURES + DEFAULT_CAT)))
    # Categorical columns limited to those present
    cat_cols = [c for c in DEFAULT_CAT if c in df.columns]

    # Default/tuned params
    xgb_params: Dict = {
        "n_estimators": 800,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
        "tree_method": "hist",
        "n_jobs": -1,
    }
    # Try to load previous best params if available
    try:
        import joblib  # type: ignore

        for cand in [
            "best_params_xgboost_random_search.pkl",
            "best_params_xgboost_cleaned.pkl",
            os.path.join(os.path.dirname(args.data_csv), "best_params_xgboost_random_search.pkl"),
        ]:
            if os.path.exists(cand):
                loaded = joblib.load(cand)
                if isinstance(loaded, dict):
                    xgb_params.update(loaded)
                break
    except Exception:
        pass
    if args.xgb_params_json and os.path.exists(args.xgb_params_json):
        xgb_params.update(json.load(open(args.xgb_params_json, "r", encoding="utf-8")))

    rf_params: Dict = {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
        "n_jobs": -1,
    }
    if args.rf_params_json and os.path.exists(args.rf_params_json):
        rf_params.update(json.load(open(args.rf_params_json, "r", encoding="utf-8")))

    gbrt_params: Dict = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
    }
    if args.gbrt_params_json and os.path.exists(args.gbrt_params_json):
        gbrt_params.update(json.load(open(args.gbrt_params_json, "r", encoding="utf-8")))

    models_to_run = [m.strip() for m in (args.models if args.models != "all" else "xgb,rf,gbrt").split(",")]

    summary: List[Dict] = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for m in models_to_run:
        if m == "xgb" and xgb is None:
            print("[WARN] xgboost not installed; skipping XGB")
            continue

        params = xgb_params if m == "xgb" else rf_params if m == "rf" else gbrt_params
        pred_name = os.path.join(args.out_dir, f"{m}_predictions_eval.csv")
        metrics, _preds = kfold_eval(
            df=df,
            features=feats,
            cat_cols=cat_cols,
            model_name=m,
            model_params=params,
            k=args.kfolds,
            seed=args.seed,
            out_pred_csv=pred_name,
        )
        rec = {
            "model": m,
            "data_csv": args.data_csv,
            "features": feats,
            "cat_cols": cat_cols,
            "kfolds": args.kfolds,
            "seed": args.seed,
            "metrics": metrics,
            "params": params,
            "predictions_csv": os.path.relpath(pred_name),
            "timestamp": timestamp,
        }
        summary.append(rec)
        print(f"{m.upper()} -> n={metrics['n']} MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} MAPE={metrics['mape']:.2f}%")

    # Save JSON and CSV summary
    json_path = os.path.join(args.out_dir, f"tabular_benchmarks_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Flat CSV for quick table
    rows = []
    for r in summary:
        rows.append({
            "model": r["model"],
            "n": r["metrics"]["n"],
            "mae": r["metrics"]["mae"],
            "rmse": r["metrics"]["rmse"],
            "mape": r["metrics"]["mape"],
            "kfolds": r["kfolds"],
            "seed": r["seed"],
            "data_csv": r["data_csv"],
            "predictions_csv": r["predictions_csv"],
        })
    csv_path = os.path.join(args.out_dir, f"tabular_benchmarks_{timestamp}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("Saved:")
    print(f" - {json_path}")
    print(f" - {csv_path}")


if __name__ == "__main__":
    main()
