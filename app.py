import os
import json
import traceback
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import asyncio

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

# Try to import ManagerAgent from your codebase. If not available, fallback to a fallback manager pipeline.
try:
    from manager import ManagerAgent  # expected to provide ManagerAgent(...) interface
    HAS_MANAGER = True
except Exception:
    HAS_MANAGER = False
    from fallback_manager import ManagerAgent

# instantiate manager
manager = ManagerAgent()

@app.route("/")
def index():
    return render_template("index.html", has_manager=HAS_MANAGER)

@app.route("/run_pipeline", methods=["POST"])
async def run_pipeline():
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

        # variable info path
        variable_info_file = request.files["variable_info_file"]

        if variable_info_file.filename == "":
            variable_info_path=None
        else:
            filename = secure_filename(variable_info_file.filename)
            variable_info_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            variable_info_file.save(variable_info_path)
        # variable_info_text = request.form.get("variable_info_text", None)
        # variable_info_file = request.files.get("variable_info_file", None)
        # if variable_info_file and variable_info_file.filename != "":
        #     vfn = secure_filename(variable_info_file.filename)
        #     vpath = os.path.join(app.config["UPLOAD_FOLDER"], vfn)
        #     variable_info_file.save(vpath)
        #     with open(vpath, "r", encoding="utf-8") as f:
        #         variable_info_text = f.read()


        target_variable = request.form.get("target_variable", None)
        if not target_variable:
            return jsonify({"error": "target_variable is required"}), 400

        problem_statement = request.form.get("problem_statement", "")
        # Run pipeline synchronously (blocking)
        result = await manager.run_pipeline(csv_path=csv_path,
                                      user_instructions=problem_statement,
                                      user_defined_target=target_variable,
                                      varb_info_path=variable_info_path)

        # Save output artifacts
        out_id = f"run_{len(os.listdir(ARTIFACTS_FOLDER))+1}"
        out_dir = os.path.join(ARTIFACTS_FOLDER, out_id)
        os.makedirs(out_dir, exist_ok=True)
        # save raw result
        with open(os.path.join(out_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(result)
            # json.dump(result, f, indent=2, default=str)



        # provide a simple response
        return result
        # return jsonify({
        #     "status": "success",
        #     "artifacts_dir": out_dir,
        #     "result_preview": {
        #         "eda": result.get("eda"),
        #         "model_summary": result.get("model_summary"),
        #         "evaluation": result.get("evaluation")
        #     }
        # })

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
