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

# ---- Optional: import your preprocessing from your repo ----
# Adjust this import to your project structure. Example:
try:
    from src.model.preprocess import preprocess_comment
except Exception:
    # fallback simple clean if your module path differs
    def preprocess_comment(text):
        import re
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

# ----- App config -----
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- Environment-driven config -----
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")  # e.g. http://ec2-ip:5000
MODEL_NAME = os.environ.get("MODEL_NAME", "my_model")
BEST_METRIC = os.environ.get("BEST_METRIC", "f1_score")   # metric name to select best model by
METRIC_ORDER = os.environ.get("METRIC_ORDER", "max")      # "max" or "min"
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "../ml_models/lgbm_model.pkl")
LOCAL_VECT_PATH = os.environ.get("LOCAL_VECT_PATH", "../ml_models/tfidf_vectorizer.pkl")
MLFLOW_MODEL_CACHE_TTL = int(os.environ.get("MLFLOW_MODEL_CACHE_TTL", "300"))  # seconds

# ----- Globals (loaded models) -----
_loaded_model = None
_loaded_vectorizer = None
_last_loaded_at = 0

def set_mlflow_uri():
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info("MLflow tracking URI set to %s", MLFLOW_TRACKING_URI)

def load_local_model_and_vectorizer():
    global _loaded_model, _loaded_vectorizer, _last_loaded_at
    logger.info("Loading local model/vectorizer fallback from disk.")
    if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(LOCAL_VECT_PATH):
        _loaded_model = pickle.load(open(LOCAL_MODEL_PATH, "rb"))
        _loaded_vectorizer = pickle.load(open(LOCAL_VECT_PATH, "rb"))
        _last_loaded_at = time.time()
        return True
    logger.warning("Local model/vectorizer files not found at configured paths.")
    return False

def autoload_best_mlflow_model():
    """
    Load best model from MLflow runs based on BEST_METRIC (across all experiments).
    Caches the model for MLFLOW_MODEL_CACHE_TTL seconds.
    Returns True if loaded.
    """
    global _loaded_model, _loaded_vectorizer, _last_loaded_at
    now = time.time()
    if _loaded_model and (now - _last_loaded_at) < MLFLOW_MODEL_CACHE_TTL:
        return True

    if not MLFLOW_TRACKING_URI:
        logger.warning("MLFLOW_TRACKING_URI not configured; skipping MLflow auto-load.")
        return False

    set_mlflow_uri()
    client = MlflowClient()
    logger.info("Searching MLflow runs for best model by metric: %s", BEST_METRIC)

    # Aggregate all runs from all experiments (could be limited / filtered for performance)
    best_score = None
    best_run = None
    for exp in client.list_experiments():
        runs = client.search_runs(exp.experiment_id, filter_string="", max_results=1000)
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
        logger.warning("No runs with metric %s found in MLflow.", BEST_METRIC)
        return False

    run_id = best_run.info.run_id
    logger.info("Selected best run %s (score=%s)", run_id, best_score)

    # Attempt to load model artifact from the run (common model path: 'model' or 'artifact')
    model_uri = f"runs:/{run_id}/model"
    try:
        _loaded_model = mlflow.pyfunc.load_model(model_uri)
        # if vectorizer is stored as separate artifact, try to download and load it:
        # We'll attempt to load 'vectorizer.pkl' artifact
        try:
            client.download_artifacts(run_id, "tfidf_vectorizer.pkl", dst_path="/tmp")
            vect_path = os.path.join("/tmp", "tfidf_vectorizer.pkl")
            if os.path.exists(vect_path):
                _loaded_vectorizer = pickle.load(open(vect_path, "rb"))
        except Exception:
            logger.info("No vectorizer artifact found inside run; model may be a pyfunc that wraps preprocessing.")
        _last_loaded_at = time.time()
        logger.info("Loaded model from MLflow run %s", run_id)
        return True
    except Exception as e:
        logger.exception("Failed to load model from MLflow run (%s): %s", run_id, e)
        # fallback to local
        return load_local_model_and_vectorizer()

# At startup try to load model
if not autoload_best_mlflow_model():
    load_local_model_and_vectorizer()

# ------------------------
# Helper functions
# ------------------------
def predict_comments(comments):
    """
    comments: list[str]
    returns list of predictions and optionally probability if model supports it
    """
    if not comments or len(comments) == 0:
        return []
    # Preprocess
    processed = [preprocess_comment(c) for c in comments]
    # If model is a pyfunc (wrapped), it may accept raw text directly
    try:
        if hasattr(_loaded_model, "predict"):
            # If vectorizer exists, use it
            if _loaded_vectorizer is not None:
                X = _loaded_vectorizer.transform(processed)
                preds = _loaded_model.predict(X)
            else:
                # let pyfunc handle preprocessing
                preds = _loaded_model.predict(pd.DataFrame({"text": processed}))
        else:
            raise RuntimeError("Loaded model has no predict")
        return list(map(int, np.array(preds).astype(int)))
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        raise

# ------------------------
# Routes: Website & API
# ------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Accepts:
    - form POST from website: 'comments' textarea with newline separated comments
    - JSON POST: {"comments": ["a", "b"]}
    Returns:
    - HTML page with results if form
    - JSON list if JSON request
    """
    try:
        # JSON body?
        if request.is_json:
            payload = request.get_json()
            comments = payload.get("comments") or payload.get("comment") or []
        else:
            # from form: newline separated
            raw = request.form.get("comments", "")
            comments = [c.strip() for c in raw.splitlines() if c.strip()]

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preds = predict_comments(comments)

        # Map to labels
        label_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
        results = [{"comment": c, "label": label_map.get(p, str(p)), "pred": p} for c, p in zip(comments, preds)]

        if request.is_json:
            return jsonify({"results": results})
        else:
            return render_template("results.html", results=results)
    except Exception as e:
        logger.exception("Error in /predict")
        return jsonify({"error": str(e)}), 500

# Upload CSV (expects a column named 'comment' or 'text')
@app.route("/upload", methods=["POST"])
def upload_csv():
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
        # take first column
        comments = df.iloc[:, 0].astype(str).tolist()
    preds = predict_comments(comments)
    label_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
    results = [{"comment": c, "label": label_map.get(p, str(p)), "pred": p} for c, p in zip(comments, preds)]
    return render_template("results.html", results=results)

# Model comparison using MLflow
@app.route("/model_compare")
def model_compare():
    if not MLFLOW_TRACKING_URI:
        flash("MLFLOW_TRACKING_URI not configured; model comparison disabled")
        return redirect(url_for("index"))
    set_mlflow_uri()
    client = MlflowClient()
    experiments = client.list_experiments()
    all_runs = []
    for exp in experiments:
        runs = client.search_runs(exp.experiment_id, order_by=["metrics." + BEST_METRIC + " DESC"], max_results=200)
        for r in runs:
            all_runs.append({
                "experiment": exp.name,
                "run_id": r.info.run_id,
                "start_time": datetime.fromtimestamp(r.info.start_time/1000.0).strftime("%Y-%m-%d %H:%M:%S"),
                "metric": r.data.metrics.get(BEST_METRIC),
                "params": r.data.params
            })
    df = pd.DataFrame(all_runs)
    if df.empty:
        flash("No runs found in MLflow")
        return redirect(url_for("index"))
    df = df.sort_values("metric", ascending=(METRIC_ORDER=="min"))
    table_html = df.to_html(classes="table table-striped", index=False, escape=False)
    return render_template("model_compare.html", table_html=table_html)

# Visual endpoints (wordcloud / pie / trend)
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
        wc.to_image().save(img_io, "PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        logger.exception("Wordcloud generation failed")
        return jsonify({"error": str(e)}), 500

@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json() if request.is_json else request.form
        sentiment_counts = data.get("sentiment_counts")
        if isinstance(sentiment_counts, str):
            import json
            sentiment_counts = json.loads(sentiment_counts)
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [int(sentiment_counts.get('1', 0)), int(sentiment_counts.get('0', 0)), int(sentiment_counts.get('-1', 0))]
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#36A2EB','#C9CBCF','#FF6384'])
        ax.axis('equal')
        img_io = io.BytesIO()
        fig.savefig(img_io, format='png', transparent=True)
        img_io.seek(0)
        plt.close(fig)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        logger.exception("Chart generation failed")
        return jsonify({"error": str(e)}), 500

@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json() if request.is_json else request.form
        sentiment_data = data.get("sentiment_data")
        if not sentiment_data:
            return jsonify({"error": "No data"}), 400
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_pct = (monthly_counts.T / monthly_totals).T * 100
        fig, ax = plt.subplots(figsize=(10,5))
        for col, color in zip(monthly_pct.columns, ['red','gray','green']):
            ax.plot(monthly_pct.index, monthly_pct[col], label=str(col), color=color)
        ax.legend()
        img_io = io.BytesIO()
        fig.savefig(img_io, format='png')
        img_io.seek(0)
        plt.close(fig)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        logger.exception("Trend graph generation failed")
        return jsonify({"error": str(e)}), 500

# YouTube comment ingestion helper (simple polling one-off)
from googleapiclient.discovery import build as google_build

@app.route("/ingest_youtube", methods=["POST"])
def ingest_youtube():
    """
    Accepts JSON:
    {
      "api_key": "...",
      "video_id": "VIDEO_ID",
      "max_results": 50
    }
    Returns predicted sentiments for comments collected.
    """
    try:
        payload = request.get_json()
        api_key = payload.get("api_key") or os.environ.get("YOUTUBE_API_KEY")
        video_id = payload.get("video_id")
        max_results = int(payload.get("max_results", 50))

        if not api_key or not video_id:
            return jsonify({"error": "api_key and video_id required"}), 400

        youtube = google_build("youtube", "v3", developerKey=api_key)
        comments = []
        request_y = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )
        while request_y and len(comments) < max_results:
            resp = request_y.execute()
            for item in resp.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(top)
                if len(comments) >= max_results:
                    break
            request_y = youtube.commentThreads().list_next(request_y, resp)

        preds = predict_comments(comments)
        label_map = {1:"Positive",0:"Neutral",-1:"Negative"}
        results = [{"comment": c, "label": label_map.get(p, str(p)), "pred": p} for c,p in zip(comments,preds)]
        return jsonify({"results": results})
    except Exception as e:
        logger.exception("YouTube ingest error")
        return jsonify({"error": str(e)}), 500

# Health
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# Run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5002)), debug=True)
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
     # Set MLflow tracking URI to your server
     mlflow.set_tracking_uri("http://ec2-44-213-77-221.compute-1.amazonaws.com:5000/")  # Replace with your MLflow tracking URI
     client = MlflowClient()
     model_uri = f"models:/{model_name}/{model_version}"
     model = mlflow.pyfunc.load_model(model_uri)
     with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
   
     return model, vectorizer