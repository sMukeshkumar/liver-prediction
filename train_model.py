"""
train_model.py - Train and save the Random Forest model
Run this ONCE before starting the app: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

print("=" * 60)
print("   LIVER CIRRHOSIS PREDICTION - MODEL TRAINING")
print("=" * 60)

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv("data/indian_liver_patient.csv")
print(f"\n✔  Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Pre-processing ─────────────────────────────────────────
# Encode Gender
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])          # Male=1, Female=0

# Fill missing values with median
df["Albumin_and_Globulin_Ratio"].fillna(df["Albumin_and_Globulin_Ratio"].median(), inplace=True)

# Remap target:  1 (liver patient) → 1,  2 (healthy) → 0
df["Dataset"] = df["Dataset"].map({1: 1, 2: 0})

print(f"✔  Pre-processing done  →  missing values filled, labels encoded")
print(f"   Class distribution  →  Liver Patient: {(df['Dataset']==1).sum()}  |  Healthy: {(df['Dataset']==0).sum()}")

# ── 3. Feature / Target split ──────────────────────────────────
FEATURES = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Protiens",
    "Albumin", "Albumin_and_Globulin_Ratio"
]
X = df[FEATURES]
y = df["Dataset"]

# ── 4. Train / Test split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Scale features ─────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 6. Train Random Forest ────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train_sc, y_train)
print(f"\n✔  Random Forest trained  →  200 trees, max_depth=10")

# ── 7. Evaluate ───────────────────────────────────────────────
y_pred = rf.predict(X_test_sc)
acc    = accuracy_score(y_test, y_pred)

cv_scores = cross_val_score(rf, scaler.transform(X), y, cv=5, scoring="accuracy")

print(f"\n{'─'*50}")
print(f"   Test Accuracy        : {acc*100:.2f}%")
print(f"   5-Fold CV Accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"{'─'*50}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Healthy','Liver Patient'])}")

# Feature importance
fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("Top Feature Importances:")
for feat, imp in fi.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# ── 8. Save model + scaler ────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(rf,     "model/rf_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(FEATURES, "model/features.pkl")

print(f"\n✔  Model saved  →  model/rf_model.pkl")
print(f"✔  Scaler saved →  model/scaler.pkl")
print("\n🎉  Training complete! You can now run  →  python app.py\n")
