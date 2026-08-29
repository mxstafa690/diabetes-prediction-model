"""
train_final_model.py

Trains and saves the final diabetes-risk model with a decision threshold
tuned on a genuinely held-out validation split -- fixing the leak in the
original script's Section 13, where the threshold was tuned on the exact
same rows the model had just been fit on.

Usage:
    python diabetes_model_comparison.py      # once, to run the 50-run
                                              # benchmark and produce
                                              # undersampling_50_runs_model_ranking.csv
    python train_final_model.py              # trains + saves the final model
    
"""

import argparse

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from diabetes_model_comparison import (
    df,
    target,
    feature_cols,
    build_models,
    create_balanced_sample,
    find_best_threshold,
)

RANKING_CSV = "undersampling_50_runs_model_ranking.csv"


def get_best_model_name(default=None):
    """Read the winning model name from the ranking CSV produced by
    running diabetes_model_comparison.py directly. Falls back to
    `default` (or raises) if that file isn't there yet."""
    try:
        ranking = pd.read_csv(RANKING_CSV, index_col=0)
        return ranking.sort_values("Overall Score", ascending=False).index[0]
    except FileNotFoundError:
        if default is not None:
            print(
                f"Warning: {RANKING_CSV} not found -- run "
                "diabetes_model_comparison.py first for a proper model "
                f"selection. Falling back to default model: {default}"
            )
            return default
        raise RuntimeError(
            f"{RANKING_CSV} not found. Either run diabetes_model_comparison.py "
            "first (python diabetes_model_comparison.py), or pass --model explicitly."
        )


def train_final_model(
    model_name,
    seed=42,
    val_size=0.2,
    min_recall=0.72,
    refit_on_all_data=True,
):
    """
    Trains `model_name` on a balanced sample using a proper train/validation
    split: the model is fit on train only, and the decision threshold is
    tuned on the held-out validation set it never saw during fitting. This
    is the leak-free counterpart to the original script's Section 13, where
    the model was fit and threshold-tuned on the identical rows.
    """
    print(f"\nTraining final model: {model_name}")
    print(f"Validation split size: {val_size}, min recall for threshold: {min_recall}")

    # 1. Build one balanced sample for the final model
    balanced_df = create_balanced_sample(df, target, seed=seed)

    X = balanced_df[feature_cols].copy()
    y = balanced_df[target].copy()

    # 2. Split into train / validation so the threshold is tuned on data
    #    the model has never seen during fitting.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )

    # 3. Fit on TRAIN only
    model = build_models()[model_name]
    model.fit(X_train, y_train)

    # 4. Tune threshold on the held-out VALIDATION set
    y_val_prob = model.predict_proba(X_val)[:, 1]
    threshold = find_best_threshold(y_val, y_val_prob, minimum_recall=min_recall)

    # 5. Report honest validation metrics at that threshold
    y_val_pred = (y_val_prob >= threshold).astype(int)
    print("\nHeld-out validation metrics at chosen threshold:")
    print(f"  Threshold : {threshold:.3f}")
    print(f"  Accuracy  : {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"  Precision : {precision_score(y_val, y_val_pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_val, y_val_pred, zero_division=0):.4f}")
    print(f"  F1-Score  : {f1_score(y_val, y_val_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_val, y_val_prob):.4f}")

    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    print(f"  Confusion matrix -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

    final_model = model

    if refit_on_all_data:
        # Once the threshold is locked in from the held-out validation set,
        # refit the SAME model type on train+val combined so the deployed
        # model benefits from all available data. The threshold itself is
        # NOT re-tuned here -- it was already chosen on data the model
        # hadn't trained on, and re-tuning it on the refit model's own
        # training data would reintroduce the same leak we're fixing.
        print("\nRefitting on train+validation combined for the final artifact...")
        final_model = build_models()[model_name]
        final_model.fit(X, y)

    return final_model, threshold


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use, e.g. 'CatBoost'. If omitted, reads the "
        "winner from the ranking CSV produced by diabetes_model_comparison.py.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--min-recall", type=float, default=0.72)
    parser.add_argument(
        "--no-refit",
        action="store_true",
        help="Skip refitting on train+val; keep the model fit on train only.",
    )
    parser.add_argument("--model-out", default="best_diabetes_model_final.pkl")
    parser.add_argument("--threshold-out", default="best_threshold_final.pkl")
    args = parser.parse_args()

    model_name = args.model or get_best_model_name()

    final_model, threshold = train_final_model(
        model_name=model_name,
        seed=args.seed,
        val_size=args.val_size,
        min_recall=args.min_recall,
        refit_on_all_data=not args.no_refit,
    )

    joblib.dump(final_model, args.model_out)
    joblib.dump(threshold, args.threshold_out)

    print(f"\nSaved model     -> {args.model_out}")
    print(f"Saved threshold -> {args.threshold_out} (value: {threshold:.3f})")


if __name__ == "__main__":
    main()
