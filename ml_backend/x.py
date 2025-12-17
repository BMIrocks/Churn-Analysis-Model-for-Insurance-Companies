import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Run inference using saved churn model on a CSV.")
    p.add_argument("--input", required=True, help="Path to input CSV for inference")
    p.add_argument("--output", default=None, help="Path to save predictions CSV (default: <input>_inference.csv)")
    p.add_argument("--model", default="churn_model.pkl", help="Path to trained model joblib file")
    p.add_argument("--features", default="model_features.pkl", help="Path to saved features list")
    p.add_argument("--metadata", default="model_metadata.json", help="Path to training metadata JSON")
    p.add_argument("--target", default="Churn", help="Name of ground-truth target column if present")
    p.add_argument("--drop-cols", default="individual_id,address_id", help="Comma-separated columns to drop from input")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for predicted_churn (default: 0.5)")
    return p.parse_args()


def main():
    args = parse_args()

    inp = args.input
    out = args.output or os.path.splitext(inp)[0] + "_inference.csv"
    drop_cols = [c for c in args.drop_cols.split(",") if c]

    print(f"Loading model from {args.model} ...")
    model = joblib.load(args.model)
    print("Loading feature list ...")
    model_features = joblib.load(args.features)

    # Load metadata if available to override defaults
    metadata = {}
    if os.path.exists(args.metadata):
        with open(args.metadata, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        # prefer metadata target/drop_cols if present
        args.target = metadata.get("target", args.target)
        drop_cols = metadata.get("drop_cols", drop_cols)

    print(f"Reading input data from {inp} ...")
    df = pd.read_csv(inp)
    original_df = df.copy()

    # Drop identifiers and target for features
    X = df.drop(columns=[c for c in ([args.target] + drop_cols) if c in df.columns])

    # One-hot like training
    X = pd.get_dummies(X, drop_first=True)

    # Align columns to training features
    X = X.reindex(columns=list(model_features), fill_value=0)
    print(f"Aligned features shape: {X.shape}")

    # Predict probabilities and classes
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        # fallback
        try:
            scores = model.decision_function(X)
            s_min, s_max = np.min(scores), np.max(scores)
            prob = (scores - s_min) / (s_max - s_min + 1e-9)
        except Exception:
            prob = model.predict(X).astype(float)

    pred = (prob >= args.threshold).astype(int)

    # Save results
    out_df = original_df.copy()
    out_df["predicted_churn"] = pred
    out_df["predicted_churn_probability"] = prob
    out_df.to_csv(out, index=False)
    print(f"Predictions saved to {out}")

    # Optional metrics if ground truth present
    if args.target in original_df.columns:
        y_true = pd.to_numeric(original_df[args.target], errors="coerce")
        valid = ~y_true.isna()
        y_true = y_true[valid].astype(int).values
        y_pred = pred[valid]

        total = len(y_true)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        acc = (tp + tn) / total if total else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

        print("\nMetrics (on provided ground truth):")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print("  Confusion Matrix [TN FP; FN TP]:")
        print(f"  [[{tn} {fp}]\n   [{fn} {tp}]]")


if __name__ == "__main__":
    main()
