import json
import os
import argparse
import pandas as pd
import numpy as np


def metrics(truth_df: pd.DataFrame, pred_df: pd.DataFrame, col: str) -> dict:
    df = truth_df.merge(pred_df[["id", col]], on="id", how="inner")
    y = df["actual_prices"].to_numpy(dtype=float)
    p = df[col].to_numpy(dtype=float)
    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))
    mape = np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1e-8)) * 100.0
    return {"n": int(len(df)), "mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def main():
    ap = argparse.ArgumentParser(description="Compare metrics for image and fused predictions vs truth from xgb_predictions.csv")
    ap.add_argument("--img_csv", default="image_price_predictions_resnet50.csv", help="Path to image predictions CSV to evaluate")
    ap.add_argument("--fused_csv", default="fused_predictions_resnet50.csv", help="Path to fused predictions CSV to evaluate")
    ap.add_argument("--img_prev_csv", default="image_price_predictions.csv", help="Optional previous image predictions CSV for baseline comparison")
    ap.add_argument("--fused_prev_csv", default="fused_predictions.csv", help="Optional previous fused CSV for baseline comparison")
    args = ap.parse_args()
    out = {}

    xgb = pd.read_csv("xgb_predictions.csv")
    xgb["id"] = xgb["id"].astype(str)
    truth = xgb[["id", "actual_prices"]].dropna()

    # XGBoost baseline
    if {"predicted_prices"}.issubset(xgb.columns):
        out["xgb_baseline"] = metrics(truth, xgb.rename(columns={"predicted_prices": "pred"}).rename(columns={"pred": "predicted_prices"}), "predicted_prices")
    else:
        out["xgb_baseline"] = {"error": "xgb_predictions.csv is missing 'predicted_prices'"}

    # CNN ResNet-50 residual
    if os.path.exists(args.img_csv):
        img50 = pd.read_csv(args.img_csv)
        img50["id"] = img50["id"].astype(str)
        out["cnn_resnet50"] = metrics(truth, img50, "pred_price_eur")
    else:
        out["cnn_resnet50"] = {"error": f"{args.img_csv} not found"}

    # CNN previous (ResNet-18) if exists
    if os.path.exists(args.img_prev_csv):
        img18 = pd.read_csv(args.img_prev_csv)
        img18["id"] = img18["id"].astype(str)
        out["cnn_resnet18_prev"] = metrics(truth, img18, "pred_price_eur") if "pred_price_eur" in img18.columns else {"error": "pred_price_eur missing"}
    else:
        out["cnn_resnet18_prev"] = {"error": f"{args.img_prev_csv} not found"}

    # Fused new
    if os.path.exists(args.fused_csv):
        fused_new = pd.read_csv(args.fused_csv)
        col = "pred_fused" if "pred_fused" in fused_new.columns else ("fused_pred" if "fused_pred" in fused_new.columns else ("predicted_prices" if "predicted_prices" in fused_new.columns else fused_new.columns[-1]))
        out["fused_resnet50"] = metrics(truth, fused_new, col)
    else:
        out["fused_resnet50"] = {"error": f"{args.fused_csv} not found"}

    # Fused previous (if exists)
    if os.path.exists(args.fused_prev_csv):
        fused_old = pd.read_csv(args.fused_prev_csv)
        col_old = "pred_fused" if "pred_fused" in fused_old.columns else ("fused_pred" if "fused_pred" in fused_old.columns else ("predicted_prices" if "predicted_prices" in fused_old.columns else fused_old.columns[-1]))
        out["fused_prev"] = metrics(truth, fused_old, col_old)
    else:
        out["fused_prev"] = {"error": f"{args.fused_prev_csv} not found"}

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
