# ╔══════════════════════════════════════════════════════════════════╗
# ║       LIVER CIRRHOSIS PREDICTION SYSTEM — COMPLETE SETUP GUIDE  ║
# ╚══════════════════════════════════════════════════════════════════╝

## PROJECT STRUCTURE
```
liver_cirrhosis_prediction/
├── app.py                  ← Flask web application (main server)
├── train_model.py          ← ML model training script (run ONCE)
├── requirements.txt        ← Python dependencies
├── data/
│   └── indian_liver_patient.csv    ← Dataset
├── model/                  ← Auto-created after training
│   ├── rf_model.pkl        ← Trained Random Forest model
│   ├── scaler.pkl          ← Feature scaler
│   └── features.pkl        ← Feature list
├── database/               ← Auto-created on first run
│   └── users.db            ← SQLite database
└── templates/
    ├── base.html           ← Shared layout
    ├── landing.html        ← Homepage
    ├── login.html          ← Login page
    ├── register.html       ← Register page
    ├── dashboard.html      ← User dashboard
    ├── predict.html        ← Prediction form
    └── result.html         ← Prediction result
```

---

## ✅ STEP-BY-STEP SETUP IN VS CODE

### STEP 1 — Install Python
- Download Python 3.10+ from https://python.org
- During install, ✅ CHECK "Add Python to PATH"
- Verify: open Terminal in VS Code → type: python --version

### STEP 2 — Open Project in VS Code
1. Move the `liver_cirrhosis_prediction/` folder to your preferred location
2. In VS Code: File → Open Folder → select `liver_cirrhosis_prediction`

### STEP 3 — Create Virtual Environment
Open VS Code Terminal (Ctrl + ` backtick) and run:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see (venv) at the start of your terminal prompt.

### STEP 4 — Install Dependencies
```bash
pip install -r requirements.txt
```
This installs: Flask, scikit-learn, pandas, numpy, joblib, Werkzeug

### STEP 5 — Train the Machine Learning Model
```bash
python train_model.py
```
This will:
- Load indian_liver_patient.csv
- Pre-process data (encode Gender, fill NaN)
- Train Random Forest (200 trees)
- Show Test Accuracy and 5-Fold CV Accuracy
- Save model files to model/ folder

Expected output:
  ✔ Dataset loaded  →  583 rows × 11 columns
  ✔ Pre-processing done
  ✔ Random Forest trained
  Test Accuracy : ~75–80%
  5-Fold CV     : ~74–78%
  ✔ Model saved

### STEP 6 — Run the Flask App
```bash
python app.py
```

You should see:
  🚀  Liver Cirrhosis Prediction System
      Running at  →  http://127.0.0.1:5000

### STEP 7 — Open in Browser
Go to: http://127.0.0.1:5000

---

## 🚀 HOW TO USE THE SYSTEM

1. **Landing Page** → http://127.0.0.1:5000
   - Beautiful homepage explaining the system

2. **Register** → Click "Get Started" or go to /register
   - Create your account (stored in SQLite with hashed password)

3. **Login** → /login
   - Sign in with your credentials

4. **Dashboard** → /dashboard
   - See prediction statistics, charts, and history

5. **Predict** → Click "New Prediction" or /predict
   - Enter patient parameters:
     - Age, Gender
     - Total & Direct Bilirubin
     - Alkaline Phosphotase, ALT, AST
     - Total Proteins, Albumin, A/G Ratio
   - Click "Analyze & Predict"

6. **Result Page** → Shows:
   - ✅ Cirrhosis Detected / No Cirrhosis Detected
   - Risk Level (Low / Medium / High)
   - Probability gauge chart
   - Top 5 contributing features
   - Clinical interpretation & next steps
   - Full input summary

---

## 🧪 SAMPLE TEST VALUES

### Sample 1 — Liver Patient (should detect Cirrhosis)
- Age: 65, Gender: Female
- Total Bilirubin: 10.9, Direct Bilirubin: 5.5
- Alkaline Phosphotase: 699, ALT: 64, AST: 100
- Total Proteins: 7.5, Albumin: 3.2, A/G Ratio: 0.74

### Sample 2 — Healthy Patient (should be No Cirrhosis)
- Age: 35, Gender: Male
- Total Bilirubin: 0.7, Direct Bilirubin: 0.1
- Alkaline Phosphotase: 120, ALT: 14, AST: 18
- Total Proteins: 7.0, Albumin: 4.2, A/G Ratio: 1.5

---

## 🗄️ DATABASE SCHEMA (SQLite)

### users table
| Column   | Type    | Description              |
|----------|---------|--------------------------|
| id       | INTEGER | Primary key              |
| name     | TEXT    | User's full name         |
| email    | TEXT    | Unique email             |
| password | TEXT    | Bcrypt hashed password   |
| created  | TEXT    | Account creation date    |

### predictions table
| Column       | Type    | Description              |
|--------------|---------|--------------------------|
| id           | INTEGER | Primary key              |
| user_id      | INTEGER | FK → users.id            |
| age, gender  | REAL/TEXT | Patient info           |
| total_bili…  | REAL    | Lab values               |
| result       | TEXT    | Prediction label         |
| probability  | REAL    | 0.0 to 1.0               |
| risk_level   | TEXT    | Low / Medium / High      |
| predicted_at | TEXT    | Timestamp                |

---

## 🎨 TECHNOLOGY STACK
| Layer     | Technology                          |
|-----------|-------------------------------------|
| Frontend  | HTML5, CSS3, JavaScript, Chart.js   |
| Backend   | Python Flask 3.0                    |
| Database  | SQLite3 (via Python built-in)       |
| ML Model  | Scikit-learn Random Forest          |
| Auth      | Werkzeug password hashing           |
| Styling   | Custom dark medical UI              |

---

## ⚠️  COMMON ISSUES

**"ModuleNotFoundError: No module named 'flask'"**
→ Make sure virtual environment is activated: venv\Scripts\activate

**"FileNotFoundError: model/rf_model.pkl"**
→ You need to run train_model.py first: python train_model.py

**"Address already in use"**
→ Port 5000 is busy. Change in app.py: port=5001

**Model already exists, retrain:**
→ Delete the model/ folder and re-run python train_model.py

---

END OF SETUP GUIDE
