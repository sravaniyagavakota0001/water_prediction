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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, r2_score, mean_squared_error
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(BASE_DIR, "data", "waternet_dummy_data", "drinking_water_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "backend", "ml", "saved_models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "backend", "ml", "evaluation_artifacts")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def engineer_features(df):
    """Calculates TDS/EC ratio and relevant physical chemical indices without leakage."""
    X = pd.DataFrame()
    X["ph"] = df["ph"]
    X["turbidity"] = df["turbidity"]
    X["temperature"] = df["temperature"]
    X["do"] = df["do"]
    X["ec"] = df["ec"]
    X["tds"] = df["tds"]
    
    # Domain-specific Engineered features
    X["tds_ec_ratio"] = df["tds"] / (df["ec"] + 1e-6)
    X["ph_deviation"] = (df["ph"] - 7.0).abs()
    X["temp_do_factor"] = df["temperature"] * df["do"]
    X["turb_tds_index"] = np.sqrt(df["turbidity"] * df["tds"])
    return X

def train_drinking_pipeline():
    print("=" * 60)
    print("TRAINING DRINKING WATER QUALITY MODELS")
    print("=" * 60)

    raw_df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(raw_df)} records.")

    # 1. Feature Extraction (No leakage of wqi, grade, suitable_drinking)
    X = engineer_features(raw_df)
    y_bin = raw_df["suitable_drinking"].values
    y_grade = raw_df["grade"].values
    y_wqi = raw_df["wqi"].values

    feature_names = list(X.columns)
    print(f"Engineered Features ({len(feature_names)}): {feature_names}")

    # Holdout Test Split (20% untouched test set)
    X_train_raw, X_test_raw, y_train, y_test, y_wqi_train, y_wqi_test, y_grade_train, y_grade_test = train_test_split(
        X, y_bin, y_wqi, y_grade, test_size=0.20, random_state=42, stratify=y_bin
    )

    # 2. 5-Fold Stratified Cross-Validation on Train Split (No data leakage)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Baseline Logistic Regression (No Tuning)": LogisticRegression(max_iter=200, random_state=42),
        "Logistic Regression (Tuned - Best for Drinking)": LogisticRegression(C=2.5, solver="lbfgs", max_iter=1000, random_state=42),
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
            "domain": "Drinking",
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

    # RFE Feature Selection using Logistic Regression
    lr_for_rfe = LogisticRegression(C=2.5, solver="lbfgs", max_iter=1000, random_state=42)
    rfe = RFE(estimator=lr_for_rfe, n_features_to_select=6)
    rfe.fit(X_train_res, y_train_res)
    rfe_ranking = {f: int(r) for f, r in zip(feature_names, rfe.ranking_)}
    top_features = [f for f, s in zip(feature_names, rfe.support_) if s]
    print(f"\nRFE Selected Top Features for Drinking: {top_features}")

    # Primary Logistic Regression (Best for Drinking Water as designated)
    primary_lr = LogisticRegression(C=2.5, solver="lbfgs", max_iter=1000, random_state=42)
    primary_lr.fit(X_train_res, y_train_res)
    test_preds = primary_lr.predict(X_test_scaled)
    test_probs = primary_lr.predict_proba(X_test_scaled)[:, 1]

    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds)
    test_rec = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)

    print(f"\nFINAL TEST SET (Untouched Holdout) - Logistic Regression:")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"ROC-AUC:   {test_auc:.4f}")

    # Stacking / Voting Ensemble
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    xgb_clf = xgb.XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42)
    ensemble = VotingClassifier(
        estimators=[('lr', primary_lr), ('rf', rf_clf), ('xgb', xgb_clf)],
        voting='soft'
    )
    ensemble.fit(X_train_res, y_train_res)
    ens_preds = ensemble.predict(X_test_scaled)
    ens_acc = accuracy_score(y_test, ens_preds)
    print(f"Voting Ensemble Holdout Accuracy: {ens_acc:.4f}")

    # 4. Multiclass Grade Classifier
    grade_clf = LogisticRegression(C=3.0, max_iter=1000, random_state=42)
    grade_clf.fit(X_train_scaled, y_grade_train)
    grade_test_preds = grade_clf.predict(X_test_scaled)
    grade_acc = accuracy_score(y_grade_test, grade_test_preds)
    print(f"Multiclass Grade Prediction Holdout Accuracy: {grade_acc:.4f}")

    # 5. WQI Continuous Regressor
    wqi_regressor = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42)
    wqi_regressor.fit(X_train_scaled, y_wqi_train)
    wqi_test_preds = wqi_regressor.predict(X_test_scaled)
    wqi_r2 = r2_score(y_wqi_test, wqi_test_preds)
    wqi_rmse = np.sqrt(mean_squared_error(y_wqi_test, wqi_test_preds))
    print(f"WQI Regression R2: {wqi_r2:.4f}, RMSE: {wqi_rmse:.2f}")

    # 6. Precompute SHAP Explainer (TreeExplainer on RF for rapid instant explanations)
    rf_for_shap = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42)
    rf_for_shap.fit(X_train_scaled, y_train)
    explainer = shap.TreeExplainer(rf_for_shap)
    joblib.dump(explainer, os.path.join(MODELS_DIR, "drinking_shap_explainer.joblib"))
    joblib.dump(rf_for_shap, os.path.join(MODELS_DIR, "drinking_rf_shap_model.joblib"))

    # 7. Save Plots & Evaluation Artifacts
    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Unsuitable", "Suitable"], yticklabels=["Unsuitable", "Suitable"])
    plt.title("Drinking Suitability Confusion Matrix (Holdout Test)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "drinking_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"LR (AUC = {test_auc:.3f})", color="royalblue", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Drinking Suitability ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(ARTIFACTS_DIR, "drinking_roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # Save Comparison CSV
    results_df = pd.DataFrame(cv_results)
    results_df["holdout_accuracy"] = [test_acc if "Logistic Regression (Tuned" in m["model"] else None for m in cv_results]
    results_df.to_csv(os.path.join(ARTIFACTS_DIR, "drinking_model_comparison.csv"), index=False)

    # 8. Save Models and Serialized Pipeline Components
    joblib.dump(primary_lr, os.path.join(MODELS_DIR, "drinking_suitability_model.joblib"))
    joblib.dump(ensemble, os.path.join(MODELS_DIR, "drinking_ensemble_model.joblib"))
    joblib.dump(grade_clf, os.path.join(MODELS_DIR, "drinking_grade_model.joblib"))
    joblib.dump(wqi_regressor, os.path.join(MODELS_DIR, "drinking_wqi_regressor.joblib"))
    joblib.dump(final_scaler, os.path.join(MODELS_DIR, "drinking_scaler.joblib"))
    joblib.dump(rfe, os.path.join(MODELS_DIR, "drinking_rfe.joblib"))

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
        },
        "wqi_regression": {
            "r2": round(wqi_r2, 4),
            "rmse": round(wqi_rmse, 2)
        }
    }
    with open(os.path.join(MODELS_DIR, "drinking_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved all drinking models, SHAP explainer, and artifacts successfully.")
    return metadata

if __name__ == "__main__":
    train_drinking_pipeline()
