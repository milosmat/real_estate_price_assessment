import argparse
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(a), eps)
    return float(np.mean(np.abs(a - b) / denom) * 100.0)


def load_data(xgb_csv: str, img_csv: str) -> pd.DataFrame:
    dx = pd.read_csv(xgb_csv)
    di = pd.read_csv(img_csv)
    # normalize ids to string
    dx["id"] = dx["id"].astype(str)
    di["id"] = di["id"].astype(str)

    # expected columns
    if not {"id", "predicted_prices"}.issubset(dx.columns):
        # try alternative name
        if "predicted_prices" not in dx.columns and "predictions" in dx.columns:
            dx = dx.rename(columns={"predictions": "predicted_prices"})
    if not {"id", "predicted_prices"}.issubset(dx.columns):
        raise ValueError("xgb csv must contain columns: id, predicted_prices, and ideally actual_prices")

    if not {"id", "pred_price_eur"}.issubset(di.columns):
        raise ValueError("image csv must contain columns: id, pred_price_eur")

    # optional uncertainty columns
    if "pred_std_eur" not in di.columns:
        di["pred_std_eur"] = np.nan
    if "n_images" not in di.columns:
        di["n_images"] = np.nan

    # inner join on id
    df = pd.merge(dx, di[["id", "pred_price_eur", "pred_std_eur", "n_images"]], on="id", how="inner")
    # rename for clarity
    if "actual_prices" in df.columns:
        df = df.rename(columns={"actual_prices": "y"})
    df = df.rename(columns={"predicted_prices": "pred_xgb", "pred_price_eur": "pred_cnn"})

    # attempt to recover y if missing via filtered_property_data.csv
    if "y" not in df.columns:
        try:
            base = pd.read_csv("filtered_property_data.csv")
            base["id"] = base["id"].astype(str)
            df = df.merge(base[["id", "cena"]].rename(columns={"cena": "y"}), on="id", how="left")
        except Exception:
            pass

    # drop rows without target
    df = df.dropna(subset=["y", "pred_xgb", "pred_cnn"]).reset_index(drop=True)
    return df


def kfold_blend(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> Tuple[float, Dict[str, float]]:
    X1 = df["pred_xgb"].to_numpy()
    X2 = df["pred_cnn"].to_numpy()
    y = df["y"].to_numpy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    grid = np.linspace(0.0, 1.0, 101)  # weight for xgb
    fold_scores = {float(w): [] for w in grid}

    for tr, va in kf.split(X1):
        y_va = y[va]
        xgb_va = X1[va]
        cnn_va = X2[va]
        for w in grid:
            y_hat = w * xgb_va + (1.0 - w) * cnn_va
            fold_scores[float(w)].append(mae(y_va, y_hat))

    avg_scores = {w: float(np.mean(vals)) for w, vals in fold_scores.items()}
    best_w = min(avg_scores, key=avg_scores.get)

    # report metrics on full data using best_w (for reference)
    y_pred_full = best_w * X1 + (1.0 - best_w) * X2
    metrics = {
        "mae": mae(y, y_pred_full),
        "rmse": rmse(y, y_pred_full),
        "mape": mape(y, y_pred_full),
        "best_w": best_w,
    }
    return best_w, metrics


def kfold_linear(df: pd.DataFrame, n_splits: int = 5, seed: int = 42, ridge_alpha: float = None) -> Tuple[np.ndarray, Dict[str, float]]:
    X = df[["pred_xgb", "pred_cnn"]].to_numpy()
    y = df["y"].to_numpy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros_like(y, dtype=float)

    for tr, va in kf.split(X):
        X_tr, X_va = X[tr], X[va]
        y_tr = y[tr]
        if ridge_alpha is None:
            model = LinearRegression()
        else:
            model = Ridge(alpha=ridge_alpha)
        model.fit(X_tr, y_tr)
        oof_pred[va] = model.predict(X_va)

    metrics = {
        "mae": mae(y, oof_pred),
        "rmse": rmse(y, oof_pred),
        "mape": mape(y, oof_pred),
    }

    # Fit final model on all data for inference
    if ridge_alpha is None:
        final_model = LinearRegression()
    else:
        final_model = Ridge(alpha=ridge_alpha)
    final_model.fit(X, y)

    coefs = getattr(final_model, "coef_", np.array([np.nan, np.nan]))
    intercept = getattr(final_model, "intercept_", 0.0)

    metrics.update({"coef_xgb": float(coefs[0]), "coef_cnn": float(coefs[1]), "intercept": float(intercept)})

    y_pred_full = final_model.predict(X)
    metrics.update({
        "mae_full": mae(y, y_pred_full),
        "rmse_full": rmse(y, y_pred_full),
        "mape_full": mape(y, y_pred_full),
    })

    return y_pred_full, metrics


def uncertainty_blend(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
    """Inverse-variance-like blending using per-id std from CNN predictions.
    We treat XGB weight as a constant lambda tuned to minimize MAE on full data.
    """
    y = df["y"].to_numpy()
    px = df["pred_xgb"].to_numpy()
    pc = df["pred_cnn"].to_numpy()
    std = df.get("pred_std_eur", pd.Series(np.nan)).to_numpy()
    # fallback std if missing
    if np.all(np.isnan(std)):
        std = np.full_like(pc, fill_value=np.nanmedian(np.abs(pc - np.median(pc))) + 1e-6, dtype=float)
    std = np.where(np.isnan(std) | (std <= 0), np.nanmedian(std[~np.isnan(std)]) if np.any(~np.isnan(std)) else 1.0, std)
    inv_var_cnn = 1.0 / (std ** 2 + 1e-6)

    lambdas = np.logspace(-4, 2, 25)
    best = (float("inf"), 1.0, None)
    for lam in lambdas:
        w_xgb = lam
        w_cnn = inv_var_cnn
        y_pred = (w_xgb * px + w_cnn * pc) / (w_xgb + w_cnn)
        cur = mae(y, y_pred)
        if cur < best[0]:
            best = (cur, lam, y_pred)

    best_mae, best_lambda, best_pred = best
    metrics = {
        "mae": mae(y, best_pred),
        "rmse": rmse(y, best_pred),
        "mape": mape(y, best_pred),
        "lambda_xgb": float(best_lambda),
    }
    return best_pred, metrics


def main():
    ap = argparse.ArgumentParser(description="Fuse XGB and CNN predictions for housing price")
    ap.add_argument("--xgb_csv", type=str, default="xgb_predictions.csv")
    ap.add_argument("--img_csv", type=str, default="image_price_predictions.csv")
    ap.add_argument("--out_csv", type=str, default="fused_predictions.csv")
    ap.add_argument("--report_json", type=str, default="fusion_report.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_uncertainty", action="store_true", help="Use inverse-variance blending if image std available")
    ap.add_argument("--ridge_alpha", type=float, default=None, help="If set, use Ridge(alpha) for linear stacking; otherwise plain LinearRegression")
    args = ap.parse_args()

    df = load_data(args.xgb_csv, args.img_csv)
    if df.empty:
        raise SystemExit("No overlapping ids with targets across xgb and image predictions.")

    # 1) Weighted blend grid-search
    best_w, blend_metrics = kfold_blend(df, n_splits=args.folds, seed=args.seed)

    # 2) Linear stacking
    y_pred_linear, linear_metrics = kfold_linear(df, n_splits=args.folds, seed=args.seed, ridge_alpha=args.ridge_alpha)

    # 3) Uncertainty-aware blending (optional)
    if args.use_uncertainty:
        y_pred_unc, unc_metrics = uncertainty_blend(df)
    else:
        y_pred_unc, unc_metrics = None, {"mae": float("inf"), "rmse": float("inf"), "mape": float("inf")}

    # Select best by MAE on full data
    cand = [
        ("blend", best_w * df["pred_xgb"].to_numpy() + (1.0 - best_w) * df["pred_cnn"].to_numpy(), blend_metrics),
        ("linear", y_pred_linear, linear_metrics),
        ("unc_blend", y_pred_unc, unc_metrics),
    ]
    method, y_pred, _best_metrics = min(((m, p, mt) for m, p, mt in cand if p is not None), key=lambda t: t[2]["mae"])

    out = df.copy()
    out["pred_fused"] = y_pred
    out["abs_err_xgb"] = (out["y"] - out["pred_xgb"]).abs()
    out["abs_err_cnn"] = (out["y"] - out["pred_cnn"]).abs()
    out["abs_err_fused"] = (out["y"] - out["pred_fused"]).abs()

    # Final metrics on all
    final_metrics = {
        "mae": mae(out["y"].to_numpy(), out["pred_fused"].to_numpy()),
        "rmse": rmse(out["y"].to_numpy(), out["pred_fused"].to_numpy()),
        "mape": mape(out["y"].to_numpy(), out["pred_fused"].to_numpy()),
        "method": method,
    }

    report = {
        "blend": blend_metrics,
        "linear": linear_metrics,
        "unc_blend": unc_metrics,
        "final": final_metrics,
        "n_rows": int(len(out)),
    }

    out.to_csv(args.out_csv, index=False)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Fusion completed. Saved:")
    print(f" - {args.out_csv}")
    print(f" - {args.report_json}")
    print("Summary:")
    print(json.dumps(report["final"], indent=2))


if __name__ == "__main__":
    main()
