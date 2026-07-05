"""
app.py  –  Liver Cirrhosis Prediction System
Flask + SQLite backend
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, joblib, numpy as np, os, json
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = "liver_cirrhosis_secret_2024"

DB_PATH    = "database/users.db"
MODEL_PATH = "model/rf_model.pkl"
SCALER_PATH= "model/scaler.pkl"

# ── Load ML model once at startup ────────────────────────────
rf_model  = joblib.load(MODEL_PATH)
scaler    = joblib.load(SCALER_PATH)
FEATURES  = joblib.load("model/features.pkl")

# ── Database helpers ──────────────────────────────────────────
def get_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT    NOT NULL,
            email     TEXT    NOT NULL UNIQUE,
            password  TEXT    NOT NULL,
            created   TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            patient_name  TEXT    DEFAULT '',
            age           REAL, gender TEXT,
            total_bili    REAL, direct_bili REAL,
            alk_phos      REAL, ala_amino   REAL,
            asp_amino     REAL, total_prot  REAL,
            albumin       REAL, ag_ratio    REAL,
            result        TEXT,
            probability   REAL,
            risk_level    TEXT,
            predicted_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()

# ── Auth decorator ────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("landing.html")


# ── Register ──────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if not all([name, email, password, confirm]):
            flash("All fields are required.", "error")
            return render_template("register.html")
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("register.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html")

        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, generate_password_hash(password))
            )
            conn.commit()
            conn.close()
            flash("Account created! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "error")
            return render_template("register.html")

    return render_template("register.html")


# ── Login ─────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"]   = user["id"]
            session["user_name"] = user["name"]
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")

    return render_template("login.html")


# ── Logout ────────────────────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ── Dashboard ─────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    preds = conn.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY predicted_at DESC",
        (session["user_id"],)
    ).fetchall()

    stats = conn.execute("""
        SELECT
          COUNT(*) as total,
          SUM(CASE WHEN result='Cirrhosis Detected' THEN 1 ELSE 0 END) as positive,
          SUM(CASE WHEN result='No Cirrhosis Detected' THEN 1 ELSE 0 END) as negative,
          ROUND(AVG(probability)*100,1) as avg_prob
        FROM predictions WHERE user_id=?
    """, (session["user_id"],)).fetchone()
    conn.close()

    return render_template("dashboard.html", predictions=preds, stats=stats)


# ── Predict ───────────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        try:
            patient_name = request.form.get("patient_name", "").strip()
            gender_val = 1 if request.form.get("gender") == "Male" else 0

            input_vals = [
                float(request.form["age"]),
                gender_val,
                float(request.form["total_bilirubin"]),
                float(request.form["direct_bilirubin"]),
                float(request.form["alkaline_phosphotase"]),
                float(request.form["alamine_aminotransferase"]),
                float(request.form["aspartate_aminotransferase"]),
                float(request.form["total_proteins"]),
                float(request.form["albumin"]),
                float(request.form["albumin_globulin_ratio"]),
            ]

            X = np.array(input_vals).reshape(1, -1)
            X_scaled = scaler.transform(X)

            prediction  = rf_model.predict(X_scaled)[0]
            proba       = rf_model.predict_proba(X_scaled)[0]
            prob_pos    = float(proba[1])        # probability of liver patient

            result     = "Cirrhosis Detected"     if prediction == 1 else "No Cirrhosis Detected"
            risk_level = (
                "High"   if prob_pos >= 0.70 else
                "Medium" if prob_pos >= 0.45 else
                "Low"
            )

            # Save to DB
            conn = get_db()
            conn.execute("""
                INSERT INTO predictions
                  (user_id, patient_name, age, gender, total_bili, direct_bili, alk_phos,
                   ala_amino, asp_amino, total_prot, albumin, ag_ratio,
                   result, probability, risk_level)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                session["user_id"], patient_name,
                input_vals[0], request.form["gender"],
                input_vals[2], input_vals[3], input_vals[4],
                input_vals[5], input_vals[6], input_vals[7],
                input_vals[8], input_vals[9],
                result, prob_pos, risk_level
            ))
            conn.commit()
            conn.close()

            # Feature contributions (simple approach)
            feat_imp = dict(zip(FEATURES, rf_model.feature_importances_))
            top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5]

            return render_template("result.html",
                result=result,
                probability=round(prob_pos * 100, 1),
                risk_level=risk_level,
                top_features=top_features,
                input_data=dict(zip(FEATURES, input_vals)),
                gender_label=request.form["gender"],
                patient_name=patient_name
            )

        except (ValueError, KeyError) as e:
            flash(f"Invalid input: {e}", "error")
            return redirect(url_for("predict"))

    return render_template("predict.html")


# ── History (AJAX) ────────────────────────────────────────────
@app.route("/api/history")
@login_required
def history_api():
    conn = get_db()
    preds = conn.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY predicted_at DESC",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return jsonify([dict(p) for p in preds])


# ── Stats for chart (AJAX) ────────────────────────────────────
@app.route("/api/stats")
@login_required
def stats_api():
    conn = get_db()
    rows = conn.execute(
        "SELECT result, COUNT(*) as cnt FROM predictions WHERE user_id=? GROUP BY result",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    init_db()
    print("\n🚀  Liver Cirrhosis Prediction System")
    print("    Running at  →  http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
