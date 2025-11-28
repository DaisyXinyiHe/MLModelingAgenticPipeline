import os
import json
import traceback
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
ARTIFACTS_FOLDER = "artifacts"
ALLOWED_EXTENSIONS = {"csv", "txt"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ARTIFACTS_FOLDER"] = ARTIFACTS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Try to import ManagerAgent from your codebase. If not available, fallback to a toy pipeline.
try:
    from manager_agent import ManagerAgent  # expected to provide ManagerAgent(...) interface
    HAS_MANAGER = True
except Exception:
    HAS_MANAGER = False

    # Toy fallback manager that runs a minimal pipeline (EDA -> simple model -> eval -> report)
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

    class ManagerAgent:
        def __init__(self):
            pass

        def run_pipeline(self, csv_path, problem_statement, target_variable, variable_info_text=None):
            # Basic synchronous toy pipeline
            df = pd.read_csv(csv_path)
            # quick EDA
            eda = {
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "missing": df.isna().sum().to_dict()
            }

            # simple transform: drop rows with missing target, fill others with median/mode
            df = df.dropna(subset=[target_variable])
            for col in df.columns:
                if df[col].dtype.kind in "biufc":  # numeric-like
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna("missing")

            # simple train/test split
            X = df.drop(columns=[target_variable])
            # keep only numeric columns for the toy example
            X_num = X.select_dtypes(include=["number"])
            if X_num.shape[1] == 0:
                raise RuntimeError("Toy pipeline needs at least one numeric predictor column.")
            y = df[target_variable]
            X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=42, stratify=(y if y.nunique()>1 else None))

            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") and y.nunique() > 1 else model.predict(X_test)
            metrics = {}
            # attempt classification metrics if y is binary or categorical
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            except Exception:
                metrics["roc_auc"] = None
            try:
                y_pred = model.predict(X_test)
                metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
                metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
                metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
            except Exception:
                # regression fallback not implemented in toy
                pass

            # produce simple report
            report = f"Toy pipeline report\nRows: {eda['n_rows']}\nCols: {eda['n_cols']}\nMetrics: {json.dumps(metrics)}\n"

            deck = [
                {"slide_number": 1, "title": "Executive Summary", "bullets": [f"Dataset rows: {eda['n_rows']}", f"Columns: {eda['n_cols']}"], "speaker_notes": "Toy pipeline summary", "graphics_suggestion": "None"}
            ]

            return {
                "eda": eda,
                "model_summary": {"type": "RandomForest", "details": "Toy model"},
                "evaluation": metrics,
                "report_markdown": report,
                "deck_outline": deck
            }

# instantiate manager
manager = ManagerAgent()

@app.route("/")
def index():
    return render_template("index.html", has_manager=HAS_MANAGER)

@app.route("/run_pipeline", methods=["POST"])
def run_pipeline():
    """
    Expects multipart/form-data with:
      - csv_file: file
      - variable_info_file (optional): file
      - variable_info_text (optional): pasted text
      - target_variable: string
      - problem_statement: string
    """
    try:
        if "csv_file" not in request.files:
            return jsonify({"error": "No csv_file part in request"}), 400

        csv_file = request.files["csv_file"]
        if csv_file.filename == "":
            return jsonify({"error": "No selected CSV file"}), 400
        if not allowed_file(csv_file.filename):
            return jsonify({"error": "File type not allowed. Only csv/txt"}), 400

        filename = secure_filename(csv_file.filename)
        csv_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        csv_file.save(csv_path)

        # variable info text
        variable_info_text = request.form.get("variable_info_text", None)
        variable_info_file = request.files.get("variable_info_file", None)
        if variable_info_file and variable_info_file.filename != "":
            vfn = secure_filename(variable_info_file.filename)
            vpath = os.path.join(app.config["UPLOAD_FOLDER"], vfn)
            variable_info_file.save(vpath)
            with open(vpath, "r", encoding="utf-8") as f:
                variable_info_text = f.read()

        target_variable = request.form.get("target_variable", None)
        if not target_variable:
            return jsonify({"error": "target_variable is required"}), 400

        problem_statement = request.form.get("problem_statement", "")
        # Run pipeline synchronously (blocking)
        result = manager.run_pipeline(csv_path=csv_path,
                                      problem_statement=problem_statement,
                                      target_variable=target_variable,
                                      variable_info_text=variable_info_text)

        # Save output artifacts
        out_id = f"run_{len(os.listdir(ARTIFACTS_FOLDER))+1}"
        out_dir = os.path.join(ARTIFACTS_FOLDER, out_id)
        os.makedirs(out_dir, exist_ok=True)
        # save raw result
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        # provide a simple response
        return jsonify({
            "status": "success",
            "artifacts_dir": out_dir,
            "result_preview": {
                "eda": result.get("eda"),
                "model_summary": result.get("model_summary"),
                "evaluation": result.get("evaluation")
            }
        })

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"status": "error", "error": str(e), "traceback": tb}), 500

@app.route("/artifacts/<run_id>/<filename>")
def download_artifact(run_id, filename):
    # run_id is directory name under ARTIFACTS_FOLDER
    folder = os.path.join(ARTIFACTS_FOLDER, run_id)
    if not os.path.exists(folder):
        return "Not found", 404
    return send_from_directory(folder, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
