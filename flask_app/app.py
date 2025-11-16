import os
import io
import time
import pickle
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from flask_cors import CORS
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from googleapiclient.discovery import build as google_build

# ----------------------------------------------------------------------
# PREPROCESSING IMPORT
# ----------------------------------------------------------------------
try:
    from src.model.preprocess import preprocess_comment
except Exception:
    def preprocess_comment(text):
        import re
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.strip()

# ----------------------------------------------------------------------
# FLASK SETUP
# ----------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")  
BEST_METRIC = os.environ.get("BEST_METRIC", "f1_score")  
METRIC_ORDER = os.environ.get("METRIC_ORDER", "max")  
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "../ml_models/lgbm_model.pkl")
LOCAL_VECT_PATH = os.environ.get("LOCAL_VECT_PATH", "../ml_models/tfidf_vectorizer.pkl")
MLFLOW_MODEL_CACHE_TTL = int(os.environ.get("MLFLOW_MODEL_CACHE_TTL", "300"))

_loaded_model = None
_loaded_vectorizer = None
_last_loaded_at = 0

# ----------------------------------------------------------------------
# MLflow Helpers
# ----------------------------------------------------------------------
def set_mlflow_uri():
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info("MLflow tracking URI set to %s", MLFLOW_TRACKING_URI)

def load_local_model_and_vectorizer():
    global _loaded_model, _loaded_vectorizer, _last_loaded_at
    logger.info("Loading LOCAL fallback model/vectorizer")

    if not (os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(LOCAL_VECT_PATH)):
        logger.error("Local model/vectorizer not found")
        return False

    _loaded_model = pickle.load(open(LOCAL_MODEL_PATH, "rb"))
    _loaded_vectorizer = pickle.load(open(LOCAL_VECT_PATH, "rb"))
    _last_loaded_at = time.time()
    logger.info("Loaded local model & vectorizer")
    return True

def autoload_best_mlflow_model():
    """
    Load best model from MLflow (ONLY pickled LGBM + vectorizer)
    NOT pyfunc.
    """

    global _loaded_model, _loaded_vectorizer, _last_loaded_at
    now = time.time()

    if _loaded_model and (now - _last_loaded_at < MLFLOW_MODEL_CACHE_TTL):
        return True

    if not MLFLOW_TRACKING_URI:
        logger.warning("No MLFLOW_TRACKING_URI; skipping MLflow autoload")
        return False

    set_mlflow_uri()
    client = MlflowClient()

    logger.info("Searching MLflow experiments for best model (metric=%s)", BEST_METRIC)

    best_score = None
    best_run = None

    # ───────────────────────────────────────────────────────────────
    # MLflow 3.x: Use search_experiments()
    # ───────────────────────────────────────────────────────────────
    for exp in client.search_experiments():
        runs = client.search_runs(exp.experiment_id, "", max_results=500)
        for r in runs:
            score = r.data.metrics.get(BEST_METRIC)
            if score is None:
                continue
            if best_score is None:
                best_score = score
                best_run = r
            else:
                if METRIC_ORDER == "max" and score > best_score:
                    best_score = score
                    best_run = r
                elif METRIC_ORDER == "min" and score < best_score:
                    best_score = score
                    best_run = r

    if not best_run:
        logger.warning("No MLflow run contains metric '%s'", BEST_METRIC)
        return False

    run_id = best_run.info.run_id
    logger.info("Best MLflow model = Run %s (score=%s)", run_id, best_score)

    # Download entire artifact folder for the run
    try:
        artifact_dir = client.download_artifacts(run_id, "", dst_path="/tmp/mlflow_best")
        model_path = os.path.join(artifact_dir, "lgbm_model.pkl")
        vect_path = os.path.join(artifact_dir, "tfidf_vectorizer.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError("lgbm_model.pkl missing from MLflow run")

        _loaded_model = pickle.load(open(model_path, "rb"))
        logger.info("Loaded model.pkl from MLflow")

        if os.path.exists(vect_path):
            _loaded_vectorizer = pickle.load(open(vect_path, "rb"))
            logger.info("Loaded vectorizer.pkl from MLflow")
        else:
            logger.warning("Vectorizer missing in MLflow; using local one if available")

        _last_loaded_at = time.time()
        return True

    except Exception as e:
        logger.error("MLflow load error → Falling back to local: %s", e)
        return load_local_model_and_vectorizer()

# Load model on startup
if not autoload_best_mlflow_model():
    load_local_model_and_vectorizer()

# ----------------------------------------------------------------------
# PREDICTION LOGIC
# ----------------------------------------------------------------------
def predict_comments(comments):
    """
    Predicts sentiment for list of strings.
    """

    if not comments:
        return []

    processed = [preprocess_comment(c) for c in comments]

    if _loaded_vectorizer is None:
        raise RuntimeError("Vectorizer not loaded")

    X = _loaded_vectorizer.transform(processed)

    if not hasattr(_loaded_model, "predict"):
        raise RuntimeError("Loaded model has no predict attribute")

    preds = _loaded_model.predict(X)
    return list(map(int, preds))

# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        if request.is_json:
            comments = request.json.get("comments", [])
        else:
            raw = request.form.get("comments", "")
            comments = [c.strip() for c in raw.splitlines() if c.strip()]

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preds = predict_comments(comments)
        label_map = {1:"Positive",0:"Neutral",-1:"Negative"}

        results = [
            {"comment": c, "label": label_map.get(p), "pred": p}
            for c,p in zip(comments, preds)
        ]

        if request.is_json:
            return jsonify({"results": results})

        return render_template("results.html", results=results)

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_csv():
    try:
        file = request.files.get("file")
        if not file:
            flash("No file uploaded")
            return redirect("/")

        df = pd.read_csv(file)
        col = "comment" if "comment" in df else ("text" if "text" in df else df.columns[0])
        comments = df[col].astype(str).tolist()

        preds = predict_comments(comments)
        label_map = {1:"Positive",0:"Neutral",-1:"Negative"}
        results = [{"comment": c, "label": label_map[p], "pred": p} for c,p in zip(comments, preds)]

        return render_template("results.html", results=results)

    except Exception as e:
        logger.exception("CSV upload failed")
        return jsonify({"error": str(e)}), 500

@app.route("/model_compare", methods=["GET"])
def model_compare():
    if not MLFLOW_TRACKING_URI:
        flash("MLFLOW_TRACKING_URI not set")
        return redirect("/")

    set_mlflow_uri()
    client = MlflowClient()
    experiments = client.search_experiments()

    rows = []
    for exp in experiments:
        runs = client.search_runs(
            exp.experiment_id,
            max_results=200,
            order_by=[f"metrics.{BEST_METRIC} DESC"]
        )
        for r in runs:
            rows.append({
                "experiment": exp.name,
                "run_id": r.info.run_id,
                "metric": r.data.metrics.get(BEST_METRIC),
                "params": r.data.params,
                "start_time": datetime.fromtimestamp(r.info.start_time/1000).strftime("%Y-%m-%d %H:%M:%S")
            })

    df = pd.DataFrame(rows)
    if df.empty:
        flash("No runs found in MLflow")
        return redirect("/")

    df = df.sort_values("metric", ascending=(METRIC_ORDER=="min"))
    table_html = df.to_html(classes="table table-striped", index=False)

    return render_template("model_compare.html", table_html=table_html)

# ----------------------------------------------------------------------
# VISUALIZATION ENDPOINTS
# ----------------------------------------------------------------------

@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.json or request.form
        comments = data.get("comments", [])

        if isinstance(comments, str):
            comments = [comments]

        processed = [preprocess_comment(c) for c in comments]
        text = " ".join(processed)

        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        img_io = io.BytesIO()
        wc.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        logger.exception("Wordcloud failed")
        return jsonify({"error": str(e)}), 500

@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.json or {}
        counts = data.get("sentiment_counts", {})

        sizes = [
            int(counts.get("1", 0)),
            int(counts.get("0", 0)),
            int(counts.get("-1", 0)),
        ]

        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(
            sizes,
            labels=["Positive", "Neutral", "Negative"],
            autopct="%1.1f%%",
            startangle=140,
            colors=["green","gray","red"]
        )
        ax.axis("equal")

        img_io = io.BytesIO()
        fig.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close(fig)

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        logger.exception("Chart failed")
        return jsonify({"error": str(e)}), 500

@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.json or {}
        entries = data.get("sentiment_data", [])

        df = pd.DataFrame(entries)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_pct = (monthly_counts.T / monthly_counts.sum(axis=1)).T * 100

        fig, ax = plt.subplots(figsize=(10,5))
        for col, color in zip(monthly_pct.columns, ["red","gray","green"]):
            ax.plot(monthly_pct.index, monthly_pct[col], label=str(col), color=color)

        ax.legend()
        img_io = io.BytesIO()
        fig.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close(fig)

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        logger.exception("Trend graph failed")
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------------------
# YOUTUBE INGESTION
# ----------------------------------------------------------------------

@app.route("/ingest_youtube", methods=["POST"])
def ingest_youtube():
    """
    Accepts:
    {
        "api_key": "...",
        "video_id": "...",
        "max_results": 50
    }
    """
    try:
        data = request.json
        api_key = data.get("api_key")
        video_id = data.get("video_id")
        max_results = int(data.get("max_results", 50))

        if not api_key or not video_id:
            return jsonify({"error": "Missing api_key or video_id"}), 400

        youtube = google_build("youtube", "v3", developerKey=api_key)
        comments = []

        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )

        while req and len(comments) < max_results:
            resp = req.execute()
            for item in resp.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
                if len(comments) >= max_results:
                    break
            req = youtube.commentThreads().list_next(req, resp)

        preds = predict_comments(comments)
        label_map = {1:"Positive",0:"Neutral",-1:"Negative"}

        results = [{"comment": c, "label": label_map[p], "pred": p} for c,p in zip(comments,preds)]

        return jsonify({"results": results})

    except Exception as e:
        logger.exception("YouTube ingestion failed")
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------------------
# VISUALIZATIONS PAGE
# ----------------------------------------------------------------------

@app.route("/visualizations")
def visualizations_page():
    return render_template("visualizations.html")

# ----------------------------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ----------------------------------------------------------------------
# RUN SERVER
# ----------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)