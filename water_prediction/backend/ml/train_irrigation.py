import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(BASE_DIR, "data", "waternet_dummy_data", "irrigation_water_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "backend", "ml", "saved_models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "backend", "ml", "evaluation_artifacts")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def engineer_features(df):
    """Computes agricultural irrigation indices (Kelly's ratio, Magnesium hazard, hardness, RSC) without leakage."""
    X = pd.DataFrame()
    X["ph"] = df["ph"]
    X["ec"] = df["ec"]
    X["tds"] = df["tds"]
    X["na_meq"] = df["na_meq"]
    X["ca_meq"] = df["ca_meq"]
    X["mg_meq"] = df["mg_meq"]
    X["chloride_meq"] = df["chloride_meq"]
    X["bicarbonate_meq"] = df["bicarbonate_meq"]
    X["sar"] = df["sar"]
    X["sodium_percent"] = df["sodium_percent"]

    # Engineered Agricultural Indices
    X["tds_ec_ratio"] = df["tds"] / (df["ec"] + 1e-6)
    # Total Hardness (mg/L CaCO3 equivalent)
    X["hardness"] = (df["ca_meq"] * 20.04 * 2.497) + (df["mg_meq"] * 12.15 * 4.118)
    # Kelly's Ratio: Na / (Ca + Mg)
    X["kellys_ratio"] = df["na_meq"] / (df["ca_meq"] + df["mg_meq"] + 1e-6)
    # Magnesium Hazard / Ratio: (Mg / (Ca + Mg)) * 100
    X["magnesium_hazard"] = (df["mg_meq"] / (df["ca_meq"] + df["mg_meq"] + 1e-6)) * 100.0
    # Residual Sodium Carbonate (RSC) = HCO3 - (Ca + Mg)
    X["rsc"] = df["bicarbonate_meq"] - (df["ca_meq"] + df["mg_meq"])
    return X

def train_irrigation_pipeline():
    print("=" * 60)
    print("TRAINING IRRIGATION WATER QUALITY MODELS")
    print("=" * 60)

    raw_df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(raw_df)} irrigation records.")

    # 1. Feature Extraction (No leakage of grade, suitable_irrigation)
    X = engineer_features(raw_df)
    y_bin = raw_df["suitable_irrigation"].values
    y_grade = raw_df["grade"].values

    feature_names = list(X.columns)
    print(f"Engineered Features ({len(feature_names)}): {feature_names}")

    # Holdout Test Split (20% untouched test set)
    X_train_raw, X_test_raw, y_train, y_test, y_grade_train, y_grade_test = train_test_split(
        X, y_bin, y_grade, test_size=0.20, random_state=42, stratify=y_bin
    )

    # 2. 5-Fold Stratified Cross-Validation on Train Split (No data leakage)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Baseline Linear SVM": SVC(kernel="linear", probability=True, random_state=42),
        "SVM RBF (Tuned - Best for Irrigation)": SVC(C=3.5, gamma="scale", kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
    }

    cv_results = []
    print("\n--- 5-Fold Stratified Cross-Validation (Training Folds Only) ---")
    
    for name, model in models.items():
        fold_accs, fold_precs, fold_recs, fold_f1s, fold_aucs = [], [], [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_raw, y_train)):
            X_tr, X_val = X_train_raw.iloc[train_idx].copy(), X_train_raw.iloc[val_idx].copy()
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Fit scaler ONLY on train fold
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)

            # Apply SMOTE ONLY on train fold
            smote = SMOTE(random_state=42)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_tr)

            # Fit model
            model.fit(X_tr_res, y_tr_res)
            preds = model.predict(X_val_scaled)
            probs = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, "predict_proba") else preds

            fold_accs.append(accuracy_score(y_val, preds))
            fold_precs.append(precision_score(y_val, preds, zero_division=0))
            fold_recs.append(recall_score(y_val, preds, zero_division=0))
            fold_f1s.append(f1_score(y_val, preds, zero_division=0))
            fold_aucs.append(roc_auc_score(y_val, probs))

        mean_acc = np.mean(fold_accs)
        mean_prec = np.mean(fold_precs)
        mean_rec = np.mean(fold_recs)
        mean_f1 = np.mean(fold_f1s)
        mean_auc = np.mean(fold_aucs)

        print(f"[{name}] CV Acc: {mean_acc:.4f} | Prec: {mean_prec:.4f} | Rec: {mean_rec:.4f} | F1: {mean_f1:.4f} | AUC: {mean_auc:.4f}")
        cv_results.append({
            "model": name,
            "domain": "Irrigation",
            "cv_accuracy": round(mean_acc, 4),
            "cv_precision": round(mean_prec, 4),
            "cv_recall": round(mean_rec, 4),
            "cv_f1": round(mean_f1, 4),
            "cv_roc_auc": round(mean_auc, 4)
        })

    # 3. Final Model Training on Full Train Split & Evaluation on Holdout Test Set
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train_raw)
    X_test_scaled = final_scaler.transform(X_test_raw)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # RFE Feature Selection using Logistic Regression / RF
    lr_for_rfe = LogisticRegression(max_iter=500, random_state=42)
    rfe = RFE(estimator=lr_for_rfe, n_features_to_select=7)
    rfe.fit(X_train_res, y_train_res)
    rfe_ranking = {f: int(r) for f, r in zip(feature_names, rfe.ranking_)}
    top_features = [f for f, s in zip(feature_names, rfe.support_) if s]
    print(f"\nRFE Selected Top Features for Irrigation: {top_features}")

    # Primary SVM RBF (Best for Irrigation as designated)
    primary_svm = SVC(C=3.5, gamma="scale", kernel="rbf", probability=True, random_state=42)
    primary_svm.fit(X_train_res, y_train_res)
    test_preds = primary_svm.predict(X_test_scaled)
    test_probs = primary_svm.predict_proba(X_test_scaled)[:, 1]

    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds)
    test_rec = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)

    print(f"\nFINAL TEST SET (Untouched Holdout) - SVM (RBF):")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"ROC-AUC:   {test_auc:.4f}")

    # Stacking / Voting Ensemble
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    xgb_clf = xgb.XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42)
    ensemble = VotingClassifier(
        estimators=[('svm', primary_svm), ('rf', rf_clf), ('xgb', xgb_clf)],
        voting='soft'
    )
    ensemble.fit(X_train_res, y_train_res)
    ens_preds = ensemble.predict(X_test_scaled)
    ens_acc = accuracy_score(y_test, ens_preds)
    print(f"Voting Ensemble Holdout Accuracy: {ens_acc:.4f}")

    # 4. Multiclass Grade Classifier
    grade_clf = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    grade_clf.fit(X_train_scaled, y_grade_train)
    grade_test_preds = grade_clf.predict(X_test_scaled)
    grade_acc = accuracy_score(y_grade_test, grade_test_preds)
    print(f"Multiclass Grade Prediction Holdout Accuracy: {grade_acc:.4f}")

    # 5. Precompute SHAP Explainer (TreeExplainer on RF for rapid instant explanations)
    rf_for_shap = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42)
    rf_for_shap.fit(X_train_scaled, y_train)
    explainer = shap.TreeExplainer(rf_for_shap)
    joblib.dump(explainer, os.path.join(MODELS_DIR, "irrigation_shap_explainer.joblib"))
    joblib.dump(rf_for_shap, os.path.join(MODELS_DIR, "irrigation_rf_shap_model.joblib"))

    # 6. Save Plots & Evaluation Artifacts
    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=["Unsuitable", "Suitable"], yticklabels=["Unsuitable", "Suitable"])
    plt.title("Irrigation Suitability Confusion Matrix (Holdout Test)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "irrigation_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"SVM RBF (AUC = {test_auc:.3f})", color="forestgreen", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Irrigation Suitability ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(ARTIFACTS_DIR, "irrigation_roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # Save Comparison CSV
    results_df = pd.DataFrame(cv_results)
    results_df["holdout_accuracy"] = [test_acc if "SVM RBF (Tuned" in m["model"] else None for m in cv_results]
    results_df.to_csv(os.path.join(ARTIFACTS_DIR, "irrigation_model_comparison.csv"), index=False)

    # 7. Save Models and Serialized Pipeline Components
    joblib.dump(primary_svm, os.path.join(MODELS_DIR, "irrigation_suitability_model.joblib"))
    joblib.dump(ensemble, os.path.join(MODELS_DIR, "irrigation_ensemble_model.joblib"))
    joblib.dump(grade_clf, os.path.join(MODELS_DIR, "irrigation_grade_model.joblib"))
    joblib.dump(final_scaler, os.path.join(MODELS_DIR, "irrigation_scaler.joblib"))
    joblib.dump(rfe, os.path.join(MODELS_DIR, "irrigation_rfe.joblib"))

    # Metadata & RFE Summary
    metadata = {
        "features": feature_names,
        "rfe_top_features": top_features,
        "rfe_ranking": rfe_ranking,
        "grade_classes": list(grade_clf.classes_),
        "holdout_test_metrics": {
            "accuracy": round(test_acc, 4),
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "f1": round(test_f1, 4),
            "roc_auc": round(test_auc, 4)
        }
    }
    with open(os.path.join(MODELS_DIR, "irrigation_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved all irrigation models, SHAP explainer, and artifacts successfully.")
    return metadata

if __name__ == "__main__":
    train_irrigation_pipeline()
