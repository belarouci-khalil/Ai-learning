
# ============================================================
# CREDIT CARD FRAUD DETECTION
# Random Forest (6-Fold CV) vs Logistic Regression (80/20)
# Simple Professional Terminal Interface
# ============================================================

from pathlib import Path
import time
import kagglehub
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. TERMINAL DISPLAY FUNCTIONS
# ============================================================

def print_header():

    print("\n")
    print("=" * 85)
    print("                    CREDIT CARD FRAUD DETECTION")
    print("=" * 85)
    print("                  Random Forest vs Logistic Regression")
    print("=" * 85)


def print_section(title):

    print("\n")
    print("-" * 85)
    print(f" {title}")
    print("-" * 85)


def print_metrics(metrics, train_time=None, predict_time=None):

    print("\nEVALUATION METRICS")
    print("-" * 45)

    for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        print(f"{metric:<20}: {metrics[metric]:.4f}")

    if train_time is not None:
        print(f"{'Training Time':<20}: {train_time:.2f} ms")

    if predict_time is not None:
        print(f"{'Prediction Time':<20}: {predict_time:.2f} ms")


def print_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\nCONFUSION MATRIX")
    print("-" * 55)

    print(
        f"{'Actual / Predicted':<22}"
        f"{'Legitimate':>15}"
        f"{'Fraud':>12}"
    )

    print("-" * 55)

    print(
        f"{'Legitimate':<22}"
        f"{cm[0, 0]:>15}"
        f"{cm[0, 1]:>12}"
    )

    print(
        f"{'Fraud':<22}"
        f"{cm[1, 0]:>15}"
        f"{cm[1, 1]:>12}"
    )

    print("-" * 55)


def print_classification_report(y_true, y_pred):

    print("\nCLASSIFICATION REPORT")
    print("-" * 65)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )


def evaluate_model(name, y_true, y_pred, y_proba):

    return {
        "Model": name,

        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),

        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),

        "F1-Score": f1_score(
            y_true, y_pred, zero_division=0
        ),

        "ROC-AUC": roc_auc_score(y_true, y_proba)
    }


def print_fold_results(results_df, average):

    print("\nRANDOM FOREST - 6 FOLD RESULTS")
    print("=" * 90)

    print(
        f"{'Fold':<12}"
        f"{'Accuracy':>12}"
        f"{'F1-Score':>12}"
        f"{'Recall':>12}"
        f"{'Train (ms)':>16}"
        f"{'Predict (ms)':>16}"
    )

    print("-" * 90)

    for _, row in results_df.iterrows():

        print(
            f"{row['Fold']:<12}"
            f"{row['Accuracy']:>12.4f}"
            f"{row['F1-Score']:>12.4f}"
            f"{row['Recall']:>12.4f}"
            f"{row['Train (ms)']:>16.2f}"
            f"{row['Predict (ms)']:>16.2f}"
        )

    print("-" * 90)

    print(
        f"{'AVERAGE':<12}"
        f"{average['Accuracy']:>12.4f}"
        f"{average['F1-Score']:>12.4f}"
        f"{average['Recall']:>12.4f}"
        f"{average['Train (ms)']:>16.2f}"
        f"{average['Predict (ms)']:>16.2f}"
    )

    print("=" * 90)


def print_comparison(comparison):

    print("\n")
    print("=" * 115)
    print("                         FINAL MODEL COMPARISON")
    print("=" * 115)

    print(
        f"{'Model':<23}"
        f"{'Accuracy':>11}"
        f"{'Precision':>12}"
        f"{'Recall':>11}"
        f"{'F1-Score':>11}"
        f"{'ROC-AUC':>11}"
        f"{'Train(ms)':>13}"
        f"{'Predict(ms)':>14}"
    )

    print("-" * 115)

    for _, row in comparison.iterrows():

        print(
            f"{row['Model']:<23}"
            f"{row['Accuracy']:>11.4f}"
            f"{row['Precision']:>12.4f}"
            f"{row['Recall']:>11.4f}"
            f"{row['F1-Score']:>11.4f}"
            f"{row['ROC-AUC']:>11.4f}"
            f"{row['Train (ms)']:>13.2f}"
            f"{row['Predict (ms)']:>14.2f}"
        )

    print("=" * 115)


# ============================================================
# 2. DOWNLOAD AND LOAD DATASET
# ============================================================

print_header()

print_section("DATASET LOADING")

print("Downloading Credit Card Fraud Detection dataset...")

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

csv_files = list(Path(path).rglob("*.csv"))

if not csv_files:
    raise SystemExit("ERROR: No CSV file found.")

df = pd.read_csv(csv_files[0])

df = df.drop_duplicates().reset_index(drop=True)

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nDataset loaded successfully!")

print(f"Total transactions       : {len(df):,}")
print(f"Number of features       : {X.shape[1]}")
print(f"Legitimate transactions  : {(y == 0).sum():,}")
print(f"Fraudulent transactions   : {(y == 1).sum():,}")


# ============================================================
# 3. RANDOM FOREST - 6 FOLD CROSS VALIDATION
# ============================================================

print_section("RANDOM FOREST - 6 FOLD CROSS VALIDATION")

print("Configuration:")
print("  - Number of trees: 50")
print("  - Maximum samples per tree: 80%")
print("  - CPU parallelization: Enabled")
print("  - Cross-validation: 6 folds")

kf = StratifiedKFold(
    n_splits=6,
    shuffle=True,
    random_state=42
)

rf_predictions = np.zeros(len(y), dtype=int)
rf_probabilities = np.zeros(len(y), dtype=float)

rf_fold_results = []

rf_total_train = 0
rf_total_predict = 0

print("\nStarting Random Forest training...\n")

for fold, (train_index, test_index) in enumerate(
    kf.split(X, y),
    start=1
):

    print(f"\nProcessing Fold {fold}/6...")

    X_train_rf = X.iloc[train_index]
    X_test_rf = X.iloc[test_index]

    y_train_rf = y.iloc[train_index]
    y_test_rf = y.iloc[test_index]

    model = RandomForestClassifier(
        n_estimators=50,
        max_samples=0.8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # Training time

    start_train = time.perf_counter()

    model.fit(X_train_rf, y_train_rf)

    train_time = (
        time.perf_counter() - start_train
    ) * 1000

    # Prediction time

    start_predict = time.perf_counter()

    y_pred = model.predict(X_test_rf)

    y_proba = model.predict_proba(X_test_rf)[:, 1]

    predict_time = (
        time.perf_counter() - start_predict
    ) * 1000

    # Save predictions

    rf_predictions[test_index] = y_pred
    rf_probabilities[test_index] = y_proba

    rf_total_train += train_time
    rf_total_predict += predict_time

    # Fold metrics

    accuracy = accuracy_score(y_test_rf, y_pred)

    f1 = f1_score(
        y_test_rf, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test_rf, y_pred, zero_division=0
    )

    rf_fold_results.append({

        "Fold": f"Fold {fold}",

        "Accuracy": accuracy,

        "F1-Score": f1,

        "Recall": recall,

        "Train (ms)": train_time,

        "Predict (ms)": predict_time

    })

    print(f"Fold {fold} completed!")
    print(f"  Accuracy     : {accuracy:.4f}")
    print(f"  F1-Score     : {f1:.4f}")
    print(f"  Recall       : {recall:.4f}")
    print(f"  Training     : {train_time:.2f} ms")
    print(f"  Prediction   : {predict_time:.2f} ms")


# ============================================================
# 4. RANDOM FOREST RESULTS
# ============================================================

rf_results_df = pd.DataFrame(rf_fold_results)

rf_average = rf_results_df.drop(
    columns=["Fold"]
).mean()

rf_average_train = rf_total_train / 6
rf_average_predict = rf_total_predict / 6

# Overall out-of-fold evaluation

rf_metrics = evaluate_model(
    "Random Forest",
    y.to_numpy(),
    rf_predictions,
    rf_probabilities
)

print_section("RANDOM FOREST RESULTS")

print_fold_results(
    rf_results_df,
    rf_average
)

print_metrics(
    rf_metrics,
    rf_average_train,
    rf_average_predict
)

print_confusion_matrix(
    y.to_numpy(),
    rf_predictions
)

print_classification_report(
    y.to_numpy(),
    rf_predictions
)


# ============================================================
# 5. LOGISTIC REGRESSION - 80/20 SPLIT
# ============================================================

print_section("LOGISTIC REGRESSION - 80/20 SPLIT")

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training records: {len(X_train_lr):,}")
print(f"Testing records : {len(X_test_lr):,}")

# StandardScaler + Logistic Regression

lr_model = Pipeline([

    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))

])

# Training

print("\nTraining Logistic Regression...")

start_train = time.perf_counter()

lr_model.fit(X_train_lr, y_train_lr)

lr_train_time = (
    time.perf_counter() - start_train
) * 1000

# Prediction

start_predict = time.perf_counter()

lr_predictions = lr_model.predict(X_test_lr)

lr_probabilities = lr_model.predict_proba(X_test_lr)[:, 1]

lr_predict_time = (
    time.perf_counter() - start_predict
) * 1000

print("Logistic Regression training completed!")

# Evaluation

lr_metrics = evaluate_model(
    "Logistic Regression",
    y_test_lr,
    lr_predictions,
    lr_probabilities
)

print_metrics(
    lr_metrics,
    lr_train_time,
    lr_predict_time
)

print_confusion_matrix(
    y_test_lr,
    lr_predictions
)

print_classification_report(
    y_test_lr,
    lr_predictions
)


# ============================================================
# 6. FINAL MODEL COMPARISON
# ============================================================

print_section("FINAL MODEL COMPARISON")

comparison = pd.DataFrame([

    {
        "Model": "Random Forest",

        "Accuracy": rf_metrics["Accuracy"],

        "Precision": rf_metrics["Precision"],

        "Recall": rf_metrics["Recall"],

        "F1-Score": rf_metrics["F1-Score"],

        "ROC-AUC": rf_metrics["ROC-AUC"],

        "Train (ms)": rf_average_train,

        "Predict (ms)": rf_average_predict
    },

    {
        "Model": "Logistic Regression",

        "Accuracy": lr_metrics["Accuracy"],

        "Precision": lr_metrics["Precision"],

        "Recall": lr_metrics["Recall"],

        "F1-Score": lr_metrics["F1-Score"],

        "ROC-AUC": lr_metrics["ROC-AUC"],

        "Train (ms)": lr_train_time,

        "Predict (ms)": lr_predict_time
    }

])

print_comparison(comparison)


# ============================================================
# 7. BEST MODEL ACCORDING TO F1-SCORE
# ============================================================

best_model = comparison.loc[
    comparison["F1-Score"].idxmax()
]

print("\nBEST MODEL ACCORDING TO F1-SCORE")
print("-" * 55)

print(f"Model     : {best_model['Model']}")
print(f"F1-Score  : {best_model['F1-Score']:.4f}")
print(f"Precision : {best_model['Precision']:.4f}")
print(f"Recall    : {best_model['Recall']:.4f}")


# ============================================================
# 8. SAVE RESULTS
# ============================================================

rf_results_df.to_csv(
    "random_forest_6fold_results.csv",
    index=False
)

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print_section("EXPERIMENT COMPLETED")

print("Results saved successfully:")
print("  1. random_forest_6fold_results.csv")
print("  2. model_comparison.csv")

print("\nThank you for using the Credit Card Fraud Detection system!")

print("=" * 85)
