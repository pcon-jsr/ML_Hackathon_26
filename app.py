from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    get_flashed_messages,
    send_file,
)
import os
import pandas as pd
import numpy as np
import re
from werkzeug.utils import secure_filename

# Control knobs
MAX_ATTEMPTS = 3
P1_EXPECTED_ROWS = 10000  # predictions only, excluding header if present
P2_EXPECTED_ROWS = 10000  # predictions only
P1_EVAL_LABELS_PATH = "classification/eval_labels.csv"
P2_EVAL_LABELS_PATH = "regression/eval_labels.csv"
P2_EVAL_DATA_PATH = "regression/eval.csv"
LEADERBOARD_FILE_PATH = "leaderboard.csv"
UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_ATTEMPTS"] = MAX_ATTEMPTS
app.config["P1_EVAL_LABELS"] = P1_EVAL_LABELS_PATH
app.config["P2_EVAL_LABELS"] = P2_EVAL_LABELS_PATH
app.config["P2_EVAL_DATA"] = P2_EVAL_DATA_PATH
app.config["LEADERBOARD_FILE"] = LEADERBOARD_FILE_PATH
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'  # Required for flash messages

# Ensure upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Load evaluation labels and data once
p1_eval_labels = None
p2_eval_labels = None
p2_eval_data = None
try:
    p1_eval_labels = pd.read_csv(app.config["P1_EVAL_LABELS"])
    p2_eval_labels = pd.read_csv(app.config["P2_EVAL_LABELS"])
    p2_eval_data = pd.read_csv(app.config["P2_EVAL_DATA"])
except FileNotFoundError as e:
    print(f"Error loading evaluation files: {e}")
    exit()


def get_leaderboard():
    if os.path.exists(app.config["LEADERBOARD_FILE"]):
        leaderboard = pd.read_csv(app.config["LEADERBOARD_FILE"])
        for col in [
            "p1_score",
            "p1_attempts",
            "p2_score",
            "p2_attempts",
            "total_score",
            "key",
        ]:
            if col not in leaderboard.columns:
                if col == "key":
                    leaderboard[col] = ""
                else:
                    leaderboard[col] = 0
        leaderboard["key"] = leaderboard["key"].astype(str)
        return leaderboard
    else:
        leaderboard = pd.DataFrame()
        leaderboard["roll_id"] = pd.Series(dtype="str")
        leaderboard["name"] = pd.Series(dtype="str")
        leaderboard["p1_score"] = pd.Series(dtype="float")
        leaderboard["p1_attempts"] = pd.Series(dtype="int")
        leaderboard["p2_score"] = pd.Series(dtype="float")
        leaderboard["p2_attempts"] = pd.Series(dtype="int")
        leaderboard["total_score"] = pd.Series(dtype="float")
        leaderboard["key"] = pd.Series(dtype="str")
        leaderboard.to_csv(app.config["LEADERBOARD_FILE"], index=False)
        return leaderboard


def save_leaderboard(leaderboard_df):
    leaderboard_df.to_csv(app.config["LEADERBOARD_FILE"], index=False)


def calculate_p1_score(submission_df):
    global p1_eval_labels
    if p1_eval_labels is None:
        print("DEBUG: p1_eval_labels is None in calculate_p1_score")
        return -1000

    # Extract predictions based on whether header was detected
    try:
        float(submission_df.iloc[0, 0])
        has_header = False
    except (ValueError, TypeError):
        has_header = True
    if has_header:
        predictions = submission_df.iloc[1:, 0]
    else:
        predictions = submission_df.iloc[:, 0]

    true_labels = p1_eval_labels.squeeze()

    if len(predictions) != len(true_labels):
        print(
            f"DEBUG: P1 submission length mismatch. Predicted: {len(predictions)}, True: {len(true_labels)}"
        )
        return -1000

    tp = ((predictions == 1) & (true_labels == 1)).sum()
    tn = ((predictions == 0) & (true_labels == 0)).sum()
    fp = ((predictions == 1) & (true_labels == 0)).sum()
    fn = ((predictions == 0) & (true_labels == 1)).sum()

    # score = (tp * 10) + (tn * 8) - (fp * 5) - (fn * 4)

    # f1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    score = f1 * 10000

    print(f"DEBUG: P1 calculated score: {score}")
    return score


def calculate_p2_score(submission_df):
    global p2_eval_labels, p2_eval_data
    if p2_eval_labels is None or p2_eval_data is None:
        print("DEBUG: p2_eval_labels or p2_eval_data is None in calculate_p2_score")
        return -1000

    predictions = submission_df.iloc[:, 0]
    predictions = pd.to_numeric(predictions, errors="coerce")

    true_labels = p2_eval_labels.squeeze()

    if "venue_capacity" not in p2_eval_data.columns:
        print(
            "DEBUG: 'venue_capacity' column not found in regression/eval.csv for P2 score calculation."
        )
        return -1000

    capacity = p2_eval_data["venue_capacity"]

    if len(predictions) != len(true_labels) or len(predictions) != len(capacity):
        print(
            f"DEBUG: P2 submission length or capacity mismatch. Predicted: {len(predictions)}, True: {len(true_labels)}, Capacity: {len(capacity)}"
        )
        return -1000

    tau = 0.1
    e = np.abs(predictions - true_labels) / capacity
    print(f"DEBUG: Number of correct predictions (e=0): {(e == 0).sum()}")
    # pcoins = 100 / (1 + (e / tau) ** 2)
    pcoins = 1 / (1 + (e / tau) ** 2)
    print(f"DEBUG: P2 calculated score: {pcoins.sum()}")
    return pcoins.sum()


def validate_p1_submission(submission_df):
    if submission_df.shape[1] != 1:
        flash("P1 submission must have exactly one column.", "error")
        return False
    predictions = submission_df.iloc[:, 0]
    if len(predictions) != P1_EXPECTED_ROWS:
        flash(
            f"P1 submission must have exactly {P1_EXPECTED_ROWS} predictions.", "error"
        )
        return False
    predictions_numeric = []
    for p in predictions:
        try:
            num = float(p)
            predictions_numeric.append(num)
        except (ValueError, TypeError):
            flash("Non-numeric value in P1 predictions.", "error")
            return False
    if not all(p in [0.0, 1.0] for p in predictions_numeric):
        flash("P1 predictions must be 0 or 1.", "error")
        return False
    return True


def validate_p2_submission(submission_df):
    if submission_df.shape[1] != 1:
        flash("P2 submission must have exactly one column.", "error")
        return False
    predictions = submission_df.iloc[:, 0]
    if len(predictions) != P2_EXPECTED_ROWS:
        flash(
            f"P2 submission must have exactly {P2_EXPECTED_ROWS} predictions.", "error"
        )
        return False
    predictions_numeric = []
    for p in predictions:
        try:
            num = float(p)
            predictions_numeric.append(num)
        except (ValueError, TypeError):
            flash("Non-numeric value in P2 predictions.", "error")
            return False
    if not all(p == int(p) for p in predictions_numeric):
        flash("P2 predictions must be integers.", "error")
        return False
    return True

@app.before_request
def lowercase_path():
    path = request.path
    if path != path.lower():
        return redirect(path.lower(), code=308)

@app.route("/favicon.ico")
def favicon():
    return app.send_static_file("favicon.ico")
@app.route("/dataset1.csv")
def dataset1():
    return app.send_static_file("dataset1.csv")
@app.route("/dataset2.csv")
def dataset2():
    return app.send_static_file("dataset2.csv")
@app.route("/eval1.csv")
def eval1():
    return app.send_static_file("eval1.csv")
@app.route("/eval2.csv")
def eval2():
    return app.send_static_file("eval2.csv")



@app.route("/cheatcode", methods=["GET"])
def hint():
    return "Get CJ full health, money and armour. Trust me! "


@app.route("/hesoyam", methods=["GET"])
@app.route("/cheatcode/hesoyam", methods=["GET"])
def answer():
    return send_file(
        "./cheatcodes/cheatcode2.txt",
        mimetype="text/plain",
        as_attachment=False
    )

# NOISE

@app.route("/2026", methods=["GET"])
def new_yeare():
    return "HAPPY NEW YEAR!"

@app.route("/ayush", methods=["GET"])
@app.route("/porceylain", methods=["GET"])
@app.route("/ayush_jayaswal", methods=["GET"])
def ayush():
    return "glazin me are you? >O<"

@app.route("/chandrima", methods=["GET"])
@app.route("/chandrimahazra", methods=["GET"])
@app.route("/chandrima_hazra", methods=["GET"])
def chandrima():
    return "you wont get her with a GET request!"


@app.route("/", methods=["GET", "POST"])
def index():
    print("DEBUG: Request received.")
    leaderboard = get_leaderboard()

    if request.method == "POST":
        print("DEBUG: POST request detected.")
        roll_id = request.form.get("roll", "").lower()
        name = request.form.get("name", "")
        key = request.form.get("key", "").strip()
        print(f"DEBUG: Roll ID: {roll_id}, Name: {name}, Key: {key}")

        if not key or len(key) >= 10:
            flash("Key must be 1-9 characters.", "error")
            return redirect(url_for("index"))

        if not roll_id or not name:
            flash("Roll ID and Name are required.", "error")
            print("DEBUG: Roll ID or Name missing.")
            return redirect(url_for("index"))

        # Validate name length
        if len(name) > 50:
            flash("Name must be 50 characters or less.", "error")
            return redirect(url_for("index"))

        # Validate roll_id format
        roll_pattern = r"^2024(ug|pg)[a-z]{2}\d{3}$"
        if not re.match(roll_pattern, roll_id):
            flash(
                "Invalid Roll no.",
                "error",
            )
            return redirect(url_for("index"))
        # Check the 3 digits are between 000 and 150
        digits = int(roll_id[8:11])
        if digits > 150:
            flash("Roll ID number must be between 000 and 150.", "error")
            return redirect(url_for("index"))

        user_entry = leaderboard[leaderboard["roll_id"] == roll_id]
        if user_entry.empty:
            print(f"DEBUG: Creating new leaderboard entry for {roll_id}")
            new_entry_data = {
                "roll_id": roll_id,
                "name": name,
                "key": key,
                "p1_score": 0,
                "p1_attempts": 0,
                "p2_score": 0,
                "p2_attempts": 0,
                "total_score": 0,
            }
            new_entry_df = pd.DataFrame([new_entry_data])
            leaderboard = pd.concat([leaderboard, new_entry_df], ignore_index=True)
            user_index = leaderboard[leaderboard["roll_id"] == roll_id].index[0]
        else:
            user_index = user_entry.index[0]
            stored_key = leaderboard.loc[user_index, "key"]
            if stored_key != key:
                flash("Invalid key.", "error")
                return redirect(url_for("index"))
            leaderboard.loc[user_index, "name"] = name
            print(f"DEBUG: Updating existing leaderboard entry for {roll_id}")

        p1_file = request.files.get("p1_file")
        p2_file = request.files.get("p2_file")

        if p1_file and p1_file.filename and p1_file.filename != "":
            print(f"DEBUG: P1 file detected: {p1_file.filename}")
            if leaderboard.loc[user_index, "p1_attempts"] < app.config["MAX_ATTEMPTS"]:
                try:
                    filename = secure_filename(p1_file.filename)
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    print(f"DEBUG: Saving P1 file to {filepath}")
                    p1_file.save(filepath)
                    print(f"DEBUG: Reading P1 file from {filepath}")
                    p1_df = pd.read_csv(filepath, header=None)
                    os.remove(filepath)
                    print(f"DEBUG: P1 file {filepath} removed.")

                    if not validate_p1_submission(p1_df):
                        flash("P1 submission CSV is invalid.", "error")
                        print("DEBUG: P1 submission CSV is invalid.")
                        return redirect(url_for("index"))

                    score = calculate_p1_score(p1_df)

                    if score > leaderboard.loc[user_index, "p1_score"]:
                        leaderboard.loc[user_index, "p1_score"] = score

                    leaderboard.loc[user_index, "total_score"] = (
                        leaderboard.loc[user_index, "p1_score"]
                        + leaderboard.loc[user_index, "p2_score"]
                    )
                    leaderboard.loc[user_index, "p1_attempts"] += 1
                    save_leaderboard(leaderboard)
                    flash("P1 submission successful!", "success")
                    print("DEBUG: P1 submission processed.")
                except Exception as e:
                    flash(f"Error processing P1 file: {e}", "error")
                    print(f"DEBUG: Error in P1 submission: {e}")
            else:
                flash("P1: No attempts left.", "error")
                print("DEBUG: P1 no attempts left.")

        if p2_file and p2_file.filename and p2_file.filename != "":
            print(f"DEBUG: P2 file detected: {p2_file.filename}")
            if leaderboard.loc[user_index, "p2_attempts"] < app.config["MAX_ATTEMPTS"]:
                try:
                    filename = secure_filename(p2_file.filename)
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    print(f"DEBUG: Saving P2 file to {filepath}")
                    p2_file.save(filepath)
                    print(f"DEBUG: Reading P2 file from {filepath}")
                    p2_df = pd.read_csv(filepath, header=None)
                    os.remove(filepath)
                    print(f"DEBUG: P2 file {filepath} removed.")

                    if not validate_p2_submission(p2_df):
                        flash("P2 submission CSV is invalid.", "error")
                        print("DEBUG: P2 submission CSV is invalid.")
                        return redirect(url_for("index"))

                    score = calculate_p2_score(p2_df)

                    if score > leaderboard.loc[user_index, "p2_score"]:
                        leaderboard.loc[user_index, "p2_score"] = score

                    leaderboard.loc[user_index, "total_score"] = (
                        leaderboard.loc[user_index, "p1_score"]
                        + leaderboard.loc[user_index, "p2_score"]
                    )
                    leaderboard.loc[user_index, "p2_attempts"] += 1
                    save_leaderboard(leaderboard)

                    flash("P2 submission successful!", "success")
                    print("DEBUG: P2 submission processed.")
                except Exception as e:
                    flash(f"Error processing P2 file: {e}", "error")
                    print(f"DEBUG: Error in P2 submission: {e}")
            else:
                flash("P2: No attempts left.", "error")
                print("DEBUG: P2 no attempts left.")

        print("DEBUG: Redirecting after POST.")
        return redirect(url_for("index"))

    messages = get_flashed_messages(with_categories=True)
    print(f"DEBUG: Flashed messages: {messages}")

    display_leaderboard = leaderboard.sort_values(
        by=["total_score", "p1_score", "p2_score"], ascending=[False, False, False]
    )

    print("DEBUG: Rendering template.")
    return render_template(
        "index.html",
        leaderboard=display_leaderboard.to_dict("records"),
        messages=messages,
    )

@app.errorhandler(404)
def reflect_unknown(_):
    return request.path.lstrip("/")

if __name__ == "__main__":
    app.run(port=2026, debug=True)
