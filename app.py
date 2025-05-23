from pathlib import Path
import os, sys, json, subprocess
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────
# 1. Load environment variables
# ──────────────────────────────────────────────────────────────
# `.env` should contain at least:
#   DATA_ROOT=absolute/path/to/GraphRAG
#   FLASK_SECRET=your-secret-string
#   OPENAI_API_KEY=sk-xxx               (if GraphRAG needs it)
load_dotenv()

DATA_ROOT = Path(os.getenv("DATA_ROOT", "data")).resolve()

# ──────────────────────────────────────────────────────────────
# 2. Helper: call GraphRAG via CLI
# ──────────────────────────────────────────────────────────────
def graphrag_cli_query(
    dataset_path: Path,
    question: str,
    method: str = "global",
) -> str:
    """
    Run: python -m graphrag query --root <dir> --method <method> --query <q>
    Returns the LLM answer as plain text (stdout).
    Using `python -m` avoids PATH issues on Windows/virtual-env.
    """
    cmd = [
        sys.executable,
        "-m",
        "graphrag",
        "query",
        "--root",
        str(dataset_path),
        "--method",
        method,
        "--query",
        question,
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,         # raises CalledProcessError on non-zero exit
    )
    return completed.stdout.strip()


# ──────────────────────────────────────────────────────────────
# 3. Helper: load dataset metadata from listing.json
# ──────────────────────────────────────────────────────────────
def load_datasets() -> list[dict]:
    """
    Read DATA_ROOT / listing.json.
    Returns an empty list if the file is missing or invalid.
    """
    listing = DATA_ROOT / "listing.json"
    if not listing.exists():
        print("[load_datasets] listing.json not found")
        return []

    try:
        with listing.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[load_datasets] Invalid JSON: {exc}")
        return []


# ──────────────────────────────────────────────────────────────
# 4. Flask app
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret")  # fallback for dev

@app.get("/")
def index():
    """Render the home page with dataset dropdown."""
    datasets = load_datasets()
    return render_template("index.html", datasets=datasets)


@app.post("/ask")
def ask():
    """Handle form submission, run GraphRAG, return the answer page."""
    question = request.form.get("question", "").strip()
    ds_key   = request.form.get("dataset")
    method   = request.form.get("method", "global")

    datasets_dict = {d["key"]: d for d in load_datasets()}

    # Basic validation
    if not question or ds_key not in datasets_dict:
        flash("Please enter a question and choose a dataset.", "danger")
        return redirect(url_for("index"))

    dataset_path = DATA_ROOT / datasets_dict[ds_key]["path"]

    # Call GraphRAG and capture its output
    try:
        answer = graphrag_cli_query(dataset_path, question, method)
    except subprocess.CalledProcessError as e:
        # Show only the first 500 chars of stderr to avoid leaking details
        answer = f"GraphRAG query failed:\n{e.stderr[:500]}..."

    return render_template(
        "index.html",
        datasets=datasets_dict.values(),
        answer=answer,
        ask=question,
        chosen=ds_key,
        method=method,
    )


if __name__ == "__main__":
    # Enable auto-reload in development; disable debug in production
    app.run(debug=True)

