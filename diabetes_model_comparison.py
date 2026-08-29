import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix
)

# =========================================================
# 1. LOAD DATA
# =========================================================
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

x = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

df = pd.concat([x, y], axis=1).drop_duplicates().reset_index(drop=True)

target = 'Diabetes_binary'

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================
def bmi_to_category(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    else:
        return 3


df['BMI_Category'] = df['BMI'].apply(bmi_to_category)

df['TotalUnhealthyDays'] = df['MentHlth'] + df['PhysHlth']

df['CardioRiskScore'] = (
    df['HighBP'] +
    df['HighChol'] +
    df['HeartDiseaseorAttack'] +
    df['Stroke']
)

df['HealthyLifestyleScore'] = (
    df['PhysActivity'] +
    df['Fruits'] +
    df['Veggies'] +
    (1 - df['Smoker']) +
    (1 - df['HvyAlcoholConsump'])
)

df['MobilityHealthBurden'] = (
    df['DiffWalk'] +
    df['GenHlth'] +
    df['PhysHlth']
)

# =========================================================
# 3. REDUCED FEATURE SET
# =========================================================
drop_cols = [
    'AnyHealthcare',
    'NoDocbcCost',
    'HeartDiseaseorAttack',
    'Smoker',
    'PhysActivity',
    'Fruits',
    'Veggies'
]

feature_cols = [col for col in df.columns if col not in [target] + drop_cols]

# =========================================================
# 4. PREPROCESSING COLUMNS
# =========================================================

#listing non-binary features to normalize
all_scale_cols = [
    'BMI',
    'MentHlth',
    'PhysHlth',
    'GenHlth',
    'Age',
    'Education',
    'Income',
    'BMI_Category',
    'TotalUnhealthyDays',
    'CardioRiskScore',
    'HealthyLifestyleScore',
    'MobilityHealthBurden'
]

#safety net
scale_cols = [col for col in all_scale_cols if col in feature_cols]

# =========================================================
# 5. MODEL BUILDING FUNCTION
# =========================================================
def make_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), scale_cols)#normalize listed columns/features
        ],
        remainder='passthrough'
    )

    return preprocessor


def build_models():
    models = {
        "Logistic Regression": Pipeline(steps=[
            ('preprocessor', make_preprocessor()),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=1000
            ))
        ]),

        "CatBoost": Pipeline(steps=[
            ('preprocessor', make_preprocessor()),
            ('classifier', CatBoostClassifier(
                depth=8,
                iterations=400,
                learning_rate=0.05,
                l2_leaf_reg=5,
                loss_function='Logloss',
                eval_metric='AUC',
                random_seed=42,
                verbose=0
            ))
        ]),

        "XGBoost": Pipeline(steps=[
            ('preprocessor', make_preprocessor()),
            ('classifier', XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=42
            ))
        ]),

        "Random Forest": Pipeline(steps=[
            ('preprocessor', make_preprocessor()),
            ('classifier', RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "Decision Tree": Pipeline(steps=[
            ('preprocessor', make_preprocessor()),
            ('classifier', DecisionTreeClassifier(
                max_depth=8,
                random_state=42
            ))
        ]),

    }

    return models

# =========================================================
# 6. THRESHOLD TUNING FUNCTION
# =========================================================

#revisit
#understand in more detail
def find_best_threshold(y_true, y_prob, minimum_recall=0.72):
    thresholds = np.arange(0.10, 0.91, 0.01)

    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        rows.append({
            'Threshold': threshold,
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0)
        })

    threshold_df = pd.DataFrame(rows)

    valid_rows = threshold_df[threshold_df['Recall'] >= minimum_recall].copy()

    #if no threshold reaches a recall of 0.72 (safety net)
    if valid_rows.empty:
        best_row = threshold_df.loc[threshold_df['F1'].idxmax()]
    else:
        best_row = valid_rows.loc[valid_rows['F1'].idxmax()]

    return float(best_row['Threshold'])

# =========================================================
# 7. CREATE ONE BALANCED SAMPLE
# =========================================================
def create_balanced_sample(df, target, seed):
    df_pos = df[df[target] == 1].copy()
    df_neg = df[df[target] == 0].copy()

    df_neg_sampled = df_neg.sample(n=len(df_pos), random_state=seed)

    balanced_df = pd.concat([df_pos, df_neg_sampled], axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)#shuffles the new smapled data

    return balanced_df

# =========================================================
# EVERYTHING BELOW ONLY RUNS WHEN THIS FILE IS EXECUTED DIRECTLY
# =========================================================
if __name__ == "__main__":

    import joblib

    # =========================================================
    # 8. RUN 50 BALANCED UNDERSAMPLING EXPERIMENTS
    # =========================================================
    all_results = []
    roc_data = {}

    n_runs = 50
    mean_fpr = np.linspace(0, 1, 100)

    # Progress counter
    total_models = len(build_models())
    total_tasks = n_runs * total_models
    task_counter = 0

    for run in range(n_runs):
        print(f"\n===== RUN {run + 1}/{n_runs} =====")

        balanced_df = create_balanced_sample(df, target, seed=run)

        X = balanced_df[feature_cols].copy()
        y_bal = balanced_df[target].copy()

        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y_bal,
            test_size=0.15,
            random_state=run,
            stratify=y_bal
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.17647,
            random_state=run,
            stratify=y_temp
        )

        models = build_models()

        for model_name, model in models.items():
            task_counter += 1

            print(
                f"Progress: {task_counter}/{total_tasks} | "
                f"Run {run + 1}/{n_runs} | "
                f"Training {model_name}"
            )

            model.fit(X_train, y_train)

            y_val_prob = model.predict_proba(X_val)[:, 1]

            best_threshold = find_best_threshold(
                y_val,
                y_val_prob,
                minimum_recall=0.72
            )

            y_test_prob = model.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_prob >= best_threshold).astype(int)

            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, zero_division=0)
            recall = recall_score(y_test, y_test_pred, zero_division=0)
            f1 = f1_score(y_test, y_test_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_test_prob)

            tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

            result = {
                'Run': run + 1,
                'Model': model_name,
                'Threshold': best_threshold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'TP': tp
            }

            all_results.append(result)

            # Store ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0

            if model_name not in roc_data:
                roc_data[model_name] = []

            roc_data[model_name].append(interp_tpr)

    # =========================================================
    # 9. RESULTS DATAFRAME
    # =========================================================
    results_df = pd.DataFrame(all_results)

    # The detailed run-by-run table is NOT printed in the terminal.
    # It will still be saved later as a CSV file.

    # =========================================================
    # 10. SUMMARY TABLE WITH MEAN AND SD
    # =========================================================
    summary_df = results_df.groupby('Model').agg({
        'Accuracy': ['mean', 'std'],
        'Precision': ['mean', 'std'],
        'Recall': ['mean', 'std'],
        'F1-Score': ['mean', 'std'],
        'ROC-AUC': ['mean', 'std'],
        'Threshold': ['mean', 'std']
    }).round(4)

    print("\nAverage Results Over 50 Runs With Standard Deviation:")
    print(summary_df)

    clean_summary = pd.DataFrame()

    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Threshold']:
        clean_summary[metric] = (
            summary_df[(metric, 'mean')].astype(str) +
            " ± " +
            summary_df[(metric, 'std')].astype(str)
        )

    print("\nClean Summary Table: Mean ± SD")
    print(clean_summary)

    # =========================================================
    # 11. AVERAGE CONFUSION MATRIX FOR EACH MODEL
    # =========================================================
    confusion_summary = results_df.groupby('Model')[['TN', 'FP', 'FN', 'TP']].mean().round(2)

    print("\nAverage Confusion Matrix for Each Model Over 50 Runs:")

    for model_name, row in confusion_summary.iterrows():
        print(f"\n{model_name}")
        print("Average Confusion Matrix:")
        print(f"[[TN: {row['TN']}, FP: {row['FP']}],")
        print(f" [FN: {row['FN']}, TP: {row['TP']}]]")

    # =========================================================
    # 12. PICK BEST OVERALL MODEL
    # =========================================================
    model_means = results_df.groupby('Model').agg({
        'Accuracy': 'mean',
        'Precision': 'mean',
        'Recall': 'mean',
        'F1-Score': 'mean',
        'ROC-AUC': 'mean'
    }).round(4)

    # Overall score:
    # ROC-AUC has the highest weight because it measures overall ranking ability.
    # F1-score is important because it balances precision and recall.
    # Recall is important because in diabetes prediction we care about catching diabetic cases.
    # Accuracy has a smaller weight because it can be misleading in imbalanced datasets.

    model_means['Overall Score'] = (
        model_means['ROC-AUC'] * 0.40 +
        model_means['F1-Score'] * 0.30 +
        model_means['Recall'] * 0.20 +
        model_means['Accuracy'] * 0.10
    ).round(4)

    model_means = model_means.sort_values(by='Overall Score', ascending=False)

    best_model_name = model_means.index[0]
    best_model_score = model_means.iloc[0]['Overall Score']

    print("\nModel Ranking Based on Overall Score:")
    print(model_means)

    print("\nBest Overall Model:")
    print(f"{best_model_name} with Overall Score = {best_model_score}")

    print("\nBest Model Average Metrics:")
    print(model_means.loc[best_model_name])







    # =========================================================
    # 13. TRAIN & SAVE FINAL BEST MODEL (full balanced data)
    # =========================================================

    print(f"\nTraining final version of best model: {best_model_name}")

    final_balanced_df = create_balanced_sample(df, target, seed=42)

    X_final = final_balanced_df[feature_cols].copy()
    y_final = final_balanced_df[target].copy()

    final_model = build_models()[best_model_name]
    final_model.fit(X_final, y_final)

    y_final_prob = final_model.predict_proba(X_final)[:, 1]
    final_threshold = find_best_threshold(y_final, y_final_prob, minimum_recall=0.72)
    print(f"Final threshold for {best_model_name}: {final_threshold}")

    joblib.dump(final_model, "best_diabetes_model.pkl")
    joblib.dump(final_threshold, "best_threshold.pkl")  

    print("Saved: best_diabetes_model.pkl")
    print("Saved: best_threshold.pkl")  





    # =========================================================
    # 14. SAVE RESULTS
    # =========================================================
    results_df.to_csv("undersampling_50_runs_detailed_results.csv", index=False)

    summary_df_flat = summary_df.copy()
    summary_df_flat.columns = ['_'.join(col).strip() for col in summary_df_flat.columns.values]
    summary_df_flat.to_csv("undersampling_50_runs_summary_results.csv")

    clean_summary.to_csv("undersampling_50_runs_clean_summary_mean_sd.csv")

    confusion_summary.to_csv("undersampling_50_runs_average_confusion_matrices.csv")

    model_means.to_csv("undersampling_50_runs_model_ranking.csv")

    print("\nSaved:")
    print("- undersampling_50_runs_detailed_results.csv")
    print("- undersampling_50_runs_summary_results.csv")
    print("- undersampling_50_runs_clean_summary_mean_sd.csv")
    print("- undersampling_50_runs_average_confusion_matrices.csv")
    print("- undersampling_50_runs_model_ranking.csv")

    # =========================================================
    # 15. F1-SCORE DISTRIBUTION FOR EACH MODEL
    # =========================================================
    for model_name in results_df['Model'].unique():
        subset = results_df[results_df['Model'] == model_name]

        plt.figure(figsize=(8, 5))
        plt.hist(subset['F1-Score'], bins=10)
        plt.title(f"F1-Score Distribution Across 50 Runs - {model_name}")
        plt.xlabel("F1-Score")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    # =========================================================
    # 16. AVERAGE ROC CURVE GRAPH FOR ALL MODELS
    # =========================================================
    plt.figure(figsize=(10, 7))

    for model_name, tprs in roc_data.items():
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)

        mean_tpr[-1] = 1.0

        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = results_df[results_df['Model'] == model_name]['ROC-AUC'].std()

        plt.plot(
            mean_fpr,
            mean_tpr,
            label=f"{model_name} Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}"
        )

        plt.fill_between(
            mean_fpr,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            alpha=0.1
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        label='Random Classifier'
    )

    plt.title("Average ROC Curves Across 50 Runs - All Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

