import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib

try:
    import xgboost as xgb
except Exception as e:
    raise RuntimeError("xgboost must be installed to run this tuner") from e

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
DEFAULT_CATS = [
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


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    # require these columns exist for target and at least one numeric
    if "cena" not in df.columns:
        raise ValueError("Dataset must contain a 'cena' target column")
    # drop rows without target or kvadratura
    keep = [c for c in ["kvadratura", "cena"] if c in df.columns]
    df = df.dropna(subset=keep)
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="RandomizedSearchCV tuner for XGB (saves best_params_xgboost_random_search.pkl)")
    ap.add_argument("--data_csv", type=str, default="filtered_property_data.csv")
    ap.add_argument("--out_pkl", type=str, default="best_params_xgboost_random_search.pkl")
    ap.add_argument("--out_report", type=str, default="docs/reports/xgb_random_search_report.json")
    ap.add_argument("--n_iter", type=int, default=50)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", type=str, default=None, help="Comma-separated feature list; default sensible set")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)

    df = load_data(args.data_csv)

    # choose features present
    requested = [c.strip() for c in args.features.split(',')] if args.features else DEFAULT_FEATURES
    feats = [c for c in requested if c in df.columns and c != 'cena' and c != 'id']
    cat_cols = [c for c in DEFAULT_CATS if c in df.columns and c in feats]
    num_cols = [c for c in feats if c not in cat_cols]
    if not feats:
        raise ValueError("No usable features found. Check dataset/feature names.")

    # preprocessing inside CV via Pipeline
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ],
        remainder="drop",
    )

    reg = xgb.XGBRegressor(
        random_state=args.seed,
        tree_method="hist",
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("reg", reg),
    ])

    # parameter distributions (prefixed with reg__ for pipeline)
    param_distributions: Dict[str, object] = {
        "reg__n_estimators": randint(200, 1500),
        "reg__max_depth": randint(3, 12),
        "reg__learning_rate": uniform(0.01, 0.29),  # ~[0.01, 0.3]
        "reg__subsample": uniform(0.5, 0.5),        # [0.5, 1.0]
        "reg__colsample_bytree": uniform(0.5, 0.5), # [0.5, 1.0]
        "reg__gamma": uniform(0.0, 0.5),
        "reg__min_child_weight": randint(1, 10),
        "reg__reg_alpha": uniform(0.0, 0.1),        # L1 ~[0,0.1]
        "reg__reg_lambda": uniform(0.5, 2.0),       # L2 ~[0.5,2.5]
    }

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring=scorer,
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=args.seed,
    )

    X = df[feats]
    y = df["cena"]
    search.fit(X, y)

    best_pipe: Pipeline = search.best_estimator_
    best_reg: xgb.XGBRegressor = best_pipe.named_steps["reg"]
    best_params = {k: best_reg.get_params().get(k) for k in [
        "n_estimators","max_depth","learning_rate","subsample","colsample_bytree","gamma","min_child_weight","reg_alpha","reg_lambda","tree_method","n_jobs"
    ]}
    # ensure two fixed params captured
    best_params["tree_method"] = best_reg.get_params().get("tree_method", "hist")
    best_params["n_jobs"] = best_reg.get_params().get("n_jobs", -1)

    joblib.dump(best_params, args.out_pkl)

    # build a simple report including ranges and best values
    ranges = {
        "n_estimators": "randint(200,1500)",
        "max_depth": "randint(3,12)",
        "learning_rate": "uniform(0.01,0.30)",
        "subsample": "uniform(0.5,1.0)",
        "colsample_bytree": "uniform(0.5,1.0)",
        "gamma": "uniform(0.0,0.5)",
        "min_child_weight": "randint(1,10)",
        "reg_alpha": "uniform(0.0,0.1)",
        "reg_lambda": "uniform(0.5,2.5)",
    }
    report = {
        "data_csv": args.data_csv,
        "features": feats,
        "categorical": cat_cols,
        "cv": args.cv,
        "n_iter": args.n_iter,
        "seed": args.seed,
        "best_score_neg_mae": float(search.best_score_),
        "best_params": best_params,
        "ranges": ranges,
    }
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # also a flat CSV table
    import csv
    table_path = os.path.splitext(args.out_report)[0] + "_table.csv"
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parametar", "raspon", "najbolja_vrednost"])
        for k, rng in ranges.items():
            w.writerow([k, rng, best_params.get(k)])

    print("Saved:")
    print(f" - best params -> {args.out_pkl}")
    print(f" - JSON report -> {args.out_report}")
    print(f" - Table CSV -> {table_path}")


if __name__ == "__main__":
    main()
