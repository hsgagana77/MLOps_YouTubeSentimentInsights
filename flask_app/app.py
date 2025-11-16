# app.py
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

# optional: YouTube client (used in /ingest_youtube)
try:
    from googleapiclient.discovery import build as google_build
except Exception:
    google_build = None  # route will complain if used without package

# ----------------------------------------------------------------------
# PREPROCESS (try import from your repo; fallback to a simple function)
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
# FLASK + LOGGING
# ----------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# CONFIG (env-driven)
# ----------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")  # e.g. http://ec2-44-213-77-221.compute-1.amazonaws.com:5000
BEST_METRIC = os.environ.get("BEST_METRIC", "f1_score")      # metric name to choose best run
METRIC_ORDER = os.environ.get("METRIC_ORDER", "max")        # "max" or "min"
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", os.path.abspath(os.path.join(os.getcwd(), "lgbm_model.pkl")))
LOCAL_VECT_PATH = os.environ.get("LOCAL_VECT_PATH", os.path.abspath(os.path.join(os.getcwd(), "tfidf_vectorizer.pkl")))
MLFLOW_MODEL_CACHE_TTL = int(os.environ.get("MLFLOW_MODEL_CACHE_TTL", "300"))  # seconds
MLFLOW_SEARCH_MAX_RUNS = int(os.environ.get("MLFLOW_SEARCH_MAX_RUNS", "1000"))

# ----------------------------------------------------------------------
# GLOBALS: loaded artifacts
# ----------------------------------------------------------------------
_loaded_model = None
_loaded_vectorizer = None
_last_loaded_at = 0

# ----------------------------------------------------------------------
# MLflow helpers
# ----------------------------------------------------------------------
def set_mlflow_uri():
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info("MLflow tracking URI set to %s", MLFLOW_TRACKING_URI)
    else:
        logger.warning("MLFLOW_TRACKING_URI not set; MLflow features will be disabled.")

def load_local_model_and_vectorizer():
    """
    Try to load model and vectorizer from configured local paths.
    """
    global _loaded_model, _loaded_vectorizer, _last_loaded_at
    logger.info("Attempting local fallback load: model=%s vectorizer=%s", LOCAL_MODEL_PATH, LOCAL_VECT_PATH)
    try:
        if os.path.exists(LOCAL_MODEL_PATH):
            with open(LOCAL_MODEL_PATH, "rb") as f:
                _loaded_model = pickle.load(f)
            logger.info("Loaded local model from %s", LOCAL_MODEL_PATH)
        else:
            logger.warning("Local model file not found at %s", LOCAL_MODEL_PATH)
            _loaded_model = None

        if os.path.exists(LOCAL_VECT_PATH):
            with open(LOCAL_VECT_PATH, "rb") as f:
                _loaded_vectorizer = pickle.load(f)
            logger.info("Loaded local vectorizer from %s", LOCAL_VECT_PATH)
        else:
            logger.warning("Local vectorizer file not found at %s", LOCAL_VECT_PATH)
            _loaded_vectorizer = None

        _last_loaded_at = time.time()
        return (_loaded_model is not None)
    except Exception as e:
        logger.exception("Failed to load local model/vectorizer: %s", e)
        return False

def _download_artifact_to(dst_dir, client, run_id, artifact_path):
    """
    Download artifact `artifact_path` from run to dst_dir.
    Returns local path to downloaded item (string) or None on failure.
    """
    try:
        downloaded = client.download_artifacts(run_id, artifact_path, dst_path=dst_dir)
        # client.download_artifacts may return a file path or dir path
        return downloaded
    except Exception as e:
        logger.debug("download_artifacts failed for %s: %s", artifact_path, e)
        return None

def autoload_best_mlflow_model():
    """
    Find best run across experiments by BEST_METRIC, download artifacts, and load.
    Primary target layout (Option A):
      run/artifacts/lgbm_model/model.pkl
      run/artifacts/tfidf_vectorizer.pkl
    Fallbacks:
      run/artifacts/lgbm_model.pkl
      run/artifacts/model.pkl
    """
    global _loaded_model, _loaded_vectorizer, _last_loaded_at

    now = time.time()
    if _loaded_model and (now - _last_loaded_at) < MLFLOW_MODEL_CACHE_TTL:
        logger.debug("Using cached model (age %s sec)", now - _last_loaded_at)
        return True

    if not MLFLOW_TRACKING_URI:
        logger.warning("MLFLOW_TRACKING_URI not configured; skipping MLflow autoload.")
        return False

    set_mlflow_uri()
    client = MlflowClient()

    logger.info("Searching MLflow runs for best model by metric: %s", BEST_METRIC)

    best_score = None
    best_run = None

    # iterate experiments and search runs
    try:
        experiments = client.search_experiments()
    except Exception as e:
        logger.exception("Failed to list MLflow experiments: %s", e)
        return False

    for exp in experiments:
        try:
            runs = client.search_runs(exp.experiment_id, filter_string="", max_results=MLFLOW_SEARCH_MAX_RUNS)
        except Exception as e:
            logger.debug("Failed to search runs for experiment %s: %s", exp.experiment_id, e)
            continue
        for r in runs:
            val = r.data.metrics.get(BEST_METRIC)
            if val is None:
                continue
            if best_score is None:
                best_score = val
                best_run = r
            else:
                if METRIC_ORDER == "max" and val > best_score:
                    best_score = val; best_run = r
                elif METRIC_ORDER == "min" and val < best_score:
                    best_score = val; best_run = r

    if not best_run:
        logger.warning("No runs with metric %s found in MLflow experiments.", BEST_METRIC)
        return False

    run_id = best_run.info.run_id
    logger.info("Selected best run %s (score=%s)", run_id, best_score)

    try:
        # create temp dir for artifacts
        tmp_base = "/tmp/mlflow_best_{}".format(run_id)
        os.makedirs(tmp_base, exist_ok=True)

        # 1) Preferred layout: 'lgbm_model' directory containing 'model.pkl'
        # try to download 'lgbm_model' folder
        downloaded_dir = _download_artifact_to(tmp_base, client, run_id, "lgbm_model")
        model_path = None
        vect_path = None

        if downloaded_dir:
            # downloaded_dir is a directory; look for model files inside it
            possible_model = os.path.join(downloaded_dir, "model.pkl")
            possible_model_alt = os.path.join(downloaded_dir, "model")  # some packaging variants
            if os.path.exists(possible_model):
                model_path = possible_model
            elif os.path.exists(possible_model_alt):
                model_path = possible_model_alt

        # 2) vectorizer usually at root artifacts "tfidf_vectorizer.pkl"
        vect_downloaded = _download_artifact_to(tmp_base, client, run_id, "tfidf_vectorizer.pkl")
        if vect_downloaded and os.path.exists(vect_downloaded):
            # when download_artifacts points to file, it may return the file path, OR a dir containing file
            if os.path.isdir(vect_downloaded):
                # find tfidf_vectorizer.pkl inside
                candidate = os.path.join(vect_downloaded, "tfidf_vectorizer.pkl")
                if os.path.exists(candidate):
                    vect_path = candidate
            else:
                vect_path = vect_downloaded

        # 3) fallback: try lgbm_model.pkl at root
        if not model_path:
            mroot = _download_artifact_to(tmp_base, client, run_id, "lgbm_model.pkl")
            if mroot and os.path.exists(mroot):
                model_path = mroot

        # 4) last fallback: try model.pkl at root artifacts
        if not model_path:
            mroot2 = _download_artifact_to(tmp_base, client, run_id, "model.pkl")
            if mroot2 and os.path.exists(mroot2):
                model_path = mroot2

        if not model_path:
            raise FileNotFoundError("Could not locate model artifact in run {} (checked lgbm_model/, lgbm_model.pkl, model.pkl)".format(run_id))

        # Load model
        logger.info("Loading model from %s", model_path)
        with open(model_path, "rb") as f:
            _loaded_model = pickle.load(f)

        # If vect_path was not found, try to search the tmp_base for any tfidf_vectorizer.pkl
        if not vect_path:
            for root, _, files in os.walk(tmp_base):
                if "tfidf_vectorizer.pkl" in files:
                    vect_path = os.path.join(root, "tfidf_vectorizer.pkl")
                    break

        if vect_path and os.path.exists(vect_path):
            logger.info("Loading vectorizer from %s", vect_path)
            with open(vect_path, "rb") as f:
                _loaded_vectorizer = pickle.load(f)
        else:
            logger.warning("Vectorizer artifact not found in run %s; _loaded_vectorizer remains None", run_id)
            _loaded_vectorizer = None

        _last_loaded_at = time.time()
        logger.info("Successfully loaded model (and vectorizer if present) from MLflow run %s", run_id)
        return True

    except Exception as e:
        logger.exception("Failed to load model from MLflow run %s: %s", run_id, e)
        # try local fallback
        return load_local_model_and_vectorizer()

# Attempt initial autoload
try:
    autoload_best_mlflow_model()
except Exception:
    logger.exception("Initial autoload attempt failed; will try local load.")
    load_local_model_and_vectorizer()

# ----------------------------------------------------------------------
# Prediction utilities
# ----------------------------------------------------------------------
def predict_comments(comments):
    """
    comments: list[str]
    returns: list[int] (pred labels)
    """
    if not comments:
        return []

    processed = [preprocess_comment(c) for c in comments]

    if _loaded_vectorizer is None:
        raise RuntimeError("Vectorizer not loaded")

    X = _loaded_vectorizer.transform(processed)

    if not hasattr(_loaded_model, "predict"):
        raise RuntimeError("Loaded model has no predict method")

    preds = _loaded_model.predict(X)
    try:
        return list(map(int, np.array(preds).astype(int)))
    except Exception:
        # Try returning as-is converted to python types
        return [int(p) if isinstance(p, (int, np.integer)) else int(np.round(p)) for p in preds]

# ----------------------------------------------------------------------
# Flask routes
# ----------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        # Accept JSON {"comments": [...]} or form textarea 'comments'
        if request.is_json:
            payload = request.get_json()
            comments = payload.get("comments") or payload.get("comment") or []
        else:
            raw = request.form.get("comments", "")
            comments = [c.strip() for c in raw.splitlines() if c.strip()]

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preds = predict_comments(comments)
        label_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
        results = [{"comment": c, "label": label_map.get(p, str(p)), "pred": int(p)} for c, p in zip(comments, preds)]

        if request.is_json:
            return jsonify({"results": results})
        else:
            return render_template("results.html", results=results)
    except Exception as e:
        logger.exception("Error in /predict")
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_csv():
    try:
        if "file" not in request.files:
            flash("No file part")
            return redirect(url_for("index"))
        f = request.files["file"]
        if f.filename == "":
            flash("No selected file")
            return redirect(url_for("index"))

        df = pd.read_csv(f)
        if "comment" in df.columns:
            comments = df["comment"].astype(str).tolist()
        elif "text" in df.columns:
            comments = df["text"].astype(str).tolist()
        else:
            comments = df.iloc[:, 0].astype(str).tolist()

        preds = predict_comments(comments)
        label_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
        results = [{"comment": c, "label": label_map.get(p, str(p)), "pred": int(p)} for c, p in zip(comments, preds)]
        return render_template("results.html", results=results)
    except Exception as e:
        logger.exception("CSV upload failed")
        return jsonify({"error": str(e)}), 500

@app.route("/model_compare", methods=["GET"])
def model_compare():
    if not MLFLOW_TRACKING_URI:
        flash("MLFLOW_TRACKING_URI not set; model comparison disabled")
        return redirect(url_for("index"))

    set_mlflow_uri()
    client = MlflowClient()
    experiments = client.search_experiments()

    rows = []
    for exp in experiments:
        runs = client.search_runs(exp.experiment_id, max_results=200, order_by=[f"metrics.{BEST_METRIC} DESC"])
        for r in runs:
            rows.append({
                "experiment": exp.name,
                "run_id": r.info.run_id,
                "start_time": datetime.fromtimestamp(r.info.start_time / 1000.0).strftime("%Y-%m-%d %H:%M:%S"),
                "metric": r.data.metrics.get(BEST_METRIC),
                "params": r.data.params
            })
    df = pd.DataFrame(rows)
    if df.empty:
        flash("No runs found in MLflow")
        return redirect(url_for("index"))
    df = df.sort_values("metric", ascending=(METRIC_ORDER == "min"))
    table_html = df.to_html(classes="table table-striped", index=False)
    return render_template("model_compare.html", table_html=table_html)

# ---------------- Visualizations (wordcloud / pie / trend) ----------------
@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json() if request.is_json else request.form
        comments = data.get("comments") or []
        if isinstance(comments, str):
            comments = [comments]
        if not comments:
            return jsonify({"error": "No comments provided"}), 400
        processed = [preprocess_comment(c) for c in comments]
        text = " ".join(processed)
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        img_io = io.BytesIO()
        wc.to_image().save(img_io, format="PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        logger.exception("Wordcloud generation failed")
        return jsonify({"error": str(e)}), 500

@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json() if request.is_json else {}
        sentiment_counts = data.get("sentiment_counts", {})
        if isinstance(sentiment_counts, str):
            import json
            sentiment_counts = json.loads(sentiment_counts)
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [int(sentiment_counts.get('1', 0)), int(sentiment_counts.get('0', 0)), int(sentiment_counts.get('-1', 0))]
        if sum(sizes) == 0:
            return jsonify({"error": "Sentiment counts sum to zero"}), 400
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        img_io = io.BytesIO()
        fig.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close(fig)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        logger.exception("Chart generation failed")
        return jsonify({"error": str(e)}), 500

@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json() if request.is_json else {}
        sentiment_data = data.get("sentiment_data")
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100
        fig, ax = plt.subplots(figsize=(12, 6))
        for col, color in zip(monthly_percentages.columns, ['red', 'gray', 'green']):
            ax.plot(monthly_percentages.index, monthly_percentages[col], marker='o', linestyle='-', label=str(col))
        ax.set_title('Monthly Sentiment Percentage Over Time')
        ax.legend()
        img_io = io.BytesIO()
        fig.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close(fig)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        logger.exception("Trend graph generation failed")
        return jsonify({"error": str(e)}), 500

# ---------------- YouTube ingestion ----------------
@app.route("/ingest_youtube", methods=["POST"])
def ingest_youtube():
    try:
        payload = request.get_json() or {}
        api_key = payload.get("api_key") or os.environ.get("YOUTUBE_API_KEY")
        video_id = payload.get("video_id")
        max_results = int(payload.get("max_results", 50))
        if not api_key or not video_id:
            return jsonify({"error": "api_key and video_id required"}), 400
        if google_build is None:
            return jsonify({"error": "google-api-python-client not installed on this environment"}), 500
        youtube = google_build("youtube", "v3", developerKey=api_key)
        comments = []
        req = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100, textFormat="plainText")
        while req and len(comments) < max_results:
            resp = req.execute()
            for item in resp.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(top)
                if len(comments) >= max_results:
                    break
            req = youtube.commentThreads().list_next(req, resp)
        preds = predict_comments(comments)
        label_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
        results = [{"comment": c, "label": label_map.get(p, str(p)), "pred": int(p)} for c, p in zip(comments, preds)]
        return jsonify({"results": results})
    except Exception as e:
        logger.exception("YouTube ingest error")
        return jsonify({"error": str(e)}), 500

@app.route("/visualizations")
def visualizations_page():
    return render_template("visualizations.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# --------------- Helper: force reload model (manual) ---------------
@app.route("/reload_model", methods=["POST"])
def reload_model_route():
    try:
        ok = autoload_best_mlflow_model()
        if not ok:
            ok = load_local_model_and_vectorizer()
        return jsonify({"loaded": bool(ok)})
    except Exception as e:
        logger.exception("reload_model failed")
        return jsonify({"error": str(e)}), 500

# ------------------- Run app -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=True)