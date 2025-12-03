import os
import pandas as pd
import numpy as np
import pickle
import google.generativeai as genai
import warnings
import datetime  # Import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# -------------------------------------------------
# 0. SUPPRESS WARNINGS
# -------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
# Some platforms expect `application`
application = app

app.secret_key = "change_this_to_a_secure_random_key"

# Database
DB_NAME = "users.db"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, DB_NAME)}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

# -------------------------------------------------
# 2. AI & ML SETUP
# -------------------------------------------------
# A. ML Model
MODEL_PATH = os.path.join(BASE_DIR, "risk_model.pkl")
risk_model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            risk_model = pickle.load(f)
        print("✅ ML Model loaded.")
    except Exception as e:
        print(f"❌ ML Model Error: {e}")

# B. Gemini Chatbot
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
chat_model = None

if GENAI_API_KEY:
    try:
        genai.configure(api_key=GENAI_API_KEY)

        CANDIDATE_MODELS = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-pro",
        ]

        found_model_name = None
        print("🔍 Probing Gemini Models...")

        try:
            api_models = [
                m.name
                for m in genai.list_models()
                if "generateContent" in m.supported_generation_methods
            ]

            for candidate in CANDIDATE_MODELS:
                match = next((m for m in api_models if candidate in m), None)
                if match:
                    found_model_name = match  # e.g. "models/gemini-2.5-flash"
                    break
        except Exception as e:
            print(f"⚠️ Listing models failed: {e}")

        if not found_model_name:
            found_model_name = "gemini-2.5-flash"

        chat_model = genai.GenerativeModel(found_model_name)
        print(f"✅ Configured AI with model: {found_model_name}")
    except Exception as e:
        print(f"❌ Gemini Setup Failed: {e}")
else:
    print("⚠️ GEMINI_API_KEY not set. Chatbot will be offline.")

SYSTEM_CONTEXT = (
    "You are a Risk Assistant for XYZ Bank. Explain IFRS 9, ECL, PD, LGD. "
    "Keep it brief."
)

# -------------------------------------------------
# 3. MODELS & DATA
# -------------------------------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150))
    email = db.Column(db.String(150), unique=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=False
    )
    username = db.Column(db.String(150))
    action = db.Column(db.String(150))
    details = db.Column(db.String(500))
    timestamp = db.Column(
        db.DateTime, default=datetime.datetime.utcnow
    )


@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))


def log_audit(action, details=""):
    if current_user.is_authenticated:
        log = AuditLog(
            user_id=current_user.id,
            username=current_user.username,
            action=action,
            details=details,
        )
        db.session.add(log)
        db.session.commit()


# ✅ Create all tables when the app imports (works with Flask 3 + Gunicorn)
with app.app_context():
    db.create_all()

# Data
SCENARIO_FILE = os.path.join(BASE_DIR, "ECL_Results_ScenarioWeighted.csv")
BASE_FILE = os.path.join(BASE_DIR, "ECL_Results.csv")
DATA_PATH = SCENARIO_FILE if os.path.exists(SCENARIO_FILE) else BASE_FILE

df = pd.DataFrame()
ID_COL = "CUSTOMER_ID"


def load_data():
    global df, ID_COL
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            if "CUSTOMER_ID" in df.columns:
                ID_COL = "CUSTOMER_ID"
            elif "ACCOUNT_ID" in df.columns:
                ID_COL = "ACCOUNT_ID"
            else:
                ID_COL = df.columns[0]
            print(f"✅ Loaded Data: {len(df)} rows")
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame(
            {
                "CUSTOMER_ID": [1001],
                "PRED_PD": [0.05],
                "ECL_12m": [100],
                "ECL_weighted": [110],
                "Stage": ["Stage 1"],
                "SEGMENT": ["Retail"],
                "EAD": [50000],
                "LGD": [0.45],
            }
        )
        ID_COL = "CUSTOMER_ID"


load_data()

# -------------------------------------------------
# 4. ROUTES
# -------------------------------------------------
def build_portfolio_context():
    if df.empty:
        return {}
    ecl_col = "ECL_12m" if "ECL_12m" in df.columns else "ECL"
    pd_col = "PRED_PD" if "PRED_PD" in df.columns else "PD"

    total_accounts = len(df)
    avg_pd = df[pd_col].mean() if pd_col in df.columns else 0.0
    total_ecl_12m = df[ecl_col].sum() if ecl_col in df.columns else 0
    total_ecl_weighted = (
        df["ECL_weighted"].sum()
        if "ECL_weighted" in df.columns
        else total_ecl_12m
    )

    if pd_col in df.columns:
        pd_vals = df[pd_col].clip(0, 1) * 100
        counts = (
            pd.cut(
                pd_vals,
                bins=[0, 1, 3, 5, 10, 100],
                labels=["<1%", "1-3%", "3-5%", "5-10%", ">10%"],
                right=False,
            )
            .value_counts()
            .sort_index()
        )
        pd_hist = {
            "labels": counts.index.tolist(),
            "data": counts.values.tolist(),
        }
    else:
        pd_hist = {"labels": [], "data": []}

    stage_grp = (
        df.groupby("Stage")[ecl_col].sum().sort_index()
        if "Stage" in df.columns
        else pd.Series()
    )
    mix_grp = (
        df["Stage"].value_counts().sort_index()
        if "Stage" in df.columns
        else pd.Series()
    )
    seg_grp = (
        df.groupby("SEGMENT")[ecl_col]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        if "SEGMENT" in df.columns
        else pd.Series()
    )

    def get_top(stage_name):
        if "Stage" not in df.columns:
            return []
        mask = df["Stage"].astype(str).str.contains(stage_name, na=False)
        return (
            df[mask]
            .sort_values(ecl_col, ascending=False)
            .head(10)
            .to_dict(orient="records")
        )

    return dict(
        total_accounts=total_accounts,
        avg_pd=avg_pd,
        total_ecl_12m=total_ecl_12m,
        total_ecl_weighted=total_ecl_weighted,
        pd_hist=pd_hist,
        ecl_by_stage={
            "labels": stage_grp.index.tolist(),
            "data": stage_grp.values.tolist(),
        },
        stage_mix={
            "labels": mix_grp.index.tolist(),
            "data": mix_grp.values.tolist(),
        },
        ecl_by_segment={
            "labels": seg_grp.index.tolist(),
            "data": seg_grp.values.tolist(),
        },
        top_s3=get_top("Stage 3"),
        top_s2=get_top("Stage 2"),
        top_s1=get_top("Stage 1"),
    )


@app.route("/")
def home():
    return render_template("home.html", user=current_user)


@app.route("/dashboard")
@login_required
def index():
    return render_template(
        "index.html",
        summary=build_portfolio_context(),
        id_col=ID_COL,
        user=current_user,
    )


@app.route("/customer")
@login_required
def customer_view():
    cust_id = request.args.get("id", "").strip()
    benchmarks = {
        "avg_pd": df["PRED_PD"].mean()
        if "PRED_PD" in df.columns
        else 0.0,
        "avg_lgd": 0.45,
    }

    if not cust_id:
        return render_template(
            "customer.html",
            id_col=ID_COL,
            customer=None,
            benchmarks=benchmarks,
            user=current_user,
        )

    mask = df[ID_COL].astype(str) == cust_id
    if not mask.any():
        return render_template(
            "customer.html",
            id_col=ID_COL,
            customer=None,
            error="Not Found",
            benchmarks=benchmarks,
            user=current_user,
        )

    log_audit("Viewed Customer", f"ID: {cust_id}")

    row = df.loc[mask].iloc[0].to_dict()
    row["risk_band"] = (
        "High"
        if row.get("PRED_PD", 0) > 0.2
        else ("Medium" if row.get("PRED_PD", 0) > 0.05 else "Low")
    )
    return render_template(
        "customer.html",
        id_col=ID_COL,
        customer=row,
        benchmarks=benchmarks,
        user=current_user,
    )


@app.route("/calculator")
@login_required
def calculator():
    return render_template("calculator.html", user=current_user)


@app.route("/predict_api", methods=["POST"])
def predict_api():
    if not risk_model:
        return {"error": "Model not loaded"}, 500
    d = request.json

    if current_user.is_authenticated:
        log_audit("Ran ML Prediction", f"Inputs: {d}")

    try:
        feats = pd.DataFrame(
            [
                [
                    float(d["outstanding"]),
                    float(d["instalment"]),
                    float(d["eir"]),
                    float(d["term"]),
                ]
            ],
            columns=[
                "OUTSTANDING",
                "INSTALMENT",
                "EIR",
                "REMAINING_TERM",
            ],
        )
        pd_val = risk_model.predict_proba(feats)[0][1]
        lgd = 0.60 if pd_val > 0.1 else 0.45
        return {
            "pd": round(pd_val * 100, 2),
            "lgd": round(lgd * 100, 2),
            "ead": float(d["outstanding"]),
            "ecl": round(pd_val * lgd * float(d["outstanding"])),
        }
    except Exception as e:
        return {"error": str(e)}, 400


@app.route("/chat_api", methods=["POST"])
def chat_api():
    msg = request.json.get("message")
    if not msg:
        return {"response": "No message"}
    try:
        if chat_model:
            res = chat_model.generate_content(
                f"{SYSTEM_CONTEXT}\nUser: {msg}"
            )
            return {"response": res.text}
        else:
            return {
                "response": "AI Brain is currently offline (Model Config Error)."
            }
    except Exception as e:
        print(f"Chat Error: {e}")
        return {"response": f"AI Error: {str(e)}"}


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_data():
    if request.method == "POST":
        f = request.files.get("file")
        if f and f.filename.endswith(".csv"):
            f.save(os.path.join(BASE_DIR, "ECL_Results.csv"))
            load_data()
            log_audit("Uploaded Data", f"File: {f.filename}")
            return redirect(url_for("index"))
    return render_template("upload.html", user=current_user)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = User.query.filter_by(
            username=request.form.get("username")
        ).first()
        if u and check_password_hash(
            u.password, request.form.get("password")
        ):
            login_user(u)
            log_audit("Logged In")
            return redirect(url_for("index"))
        flash("Invalid Credentials", "danger")
    return render_template("login.html", user=current_user)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        try:
            u = User(
                full_name=request.form.get("full_name"),
                email=request.form.get("email"),
                username=request.form.get("username"),
                password=generate_password_hash(
                    request.form.get("password")
                ),
            )
            db.session.add(u)
            db.session.commit()
            return redirect(url_for("login"))
        except Exception:
            flash("User exists", "warning")
    return render_template("signup.html", user=current_user)


@app.route("/logout")
def logout():
    if current_user.is_authenticated:
        log_audit("Logged Out")
    logout_user()
    return redirect(url_for("login"))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    print(f"🚀 Server Running in: {BASE_DIR}")
    app.run(debug=True)
