import os
import re
import json
import joblib
import numpy as np
import logging
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from ml_training.feature_engineering import extract_features

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB upload limit

SIMILARITY_THRESHOLD = 0.30
MAX_LLM_CALLS = 10
llm_cache = {}

# --------------------------------------------------
# LOAD ML MODELS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, "ml_training")

try:
    model_path = os.path.join(ML_DIR, "issue_model.pkl")
    vectorizer_path = os.path.join(ML_DIR, "tfidf_vectorizer.pkl")

    ML_MODELS = joblib.load(model_path)
    TFIDF = joblib.load(vectorizer_path)

    logging.info("ML Models Loaded Successfully")

except Exception as e:
    logging.error(f"Model loading failed: {e}")
    ML_MODELS = None
    TFIDF = None

# --------------------------------------------------
# GEMINI SUGGESTION FUNCTION (NEW SDK)
# --------------------------------------------------

def get_llm_suggestion(requirement_text, issues):

    cache_key = requirement_text + str(sorted(issues))

    if cache_key in llm_cache:
        return llm_cache[cache_key]

    if not OPENROUTER_API_KEY:
        return {
            "explanation": "OpenRouter API key not configured.",
            "improved": requirement_text
        }

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
Improve the following requirement. Issues detected: {", ".join(issues)}

Requirement:
"{requirement_text}"

Respond ONLY in JSON format:
{{"explanation":"short reason","improved":"rewritten requirement"}}
"""
        
        payload = {
            "model": "google/gemma-3n-e4b-it:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        response.raise_for_status()
        result_json = response.json()
        
        text = result_json["choices"][0]["message"]["content"].strip()

        # Clean markdown wrapping if present
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(text)

        llm_cache[cache_key] = parsed
        return parsed

    except Exception as e:
        logging.error(f"OpenRouter API Error: {e}")
        return {
            "explanation": "OpenRouter API error or timeout.",
            "improved": requirement_text
        }

# --------------------------------------------------
# CONFLICT & INCONSISTENCY CHECK
# --------------------------------------------------

OPPOSING_PAIRS = [
    ("must", "must not"),
    ("shall", "shall not"),
    ("allow", "disallow"),
    ("enable", "disable"),
]

def check_conflicts_inconsistencies(reqs):

    if len(reqs) < 2:
        return

    texts = [r["text"] for r in reqs]

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(texts)
    sim_matrix = cosine_similarity(matrix)

    for i in range(len(reqs)):
        for j in range(i + 1, len(reqs)):

            if sim_matrix[i][j] > SIMILARITY_THRESHOLD:

                t1 = reqs[i]["text"].lower()
                t2 = reqs[j]["text"].lower()

                # Conflict Detection
                for pos, neg in OPPOSING_PAIRS:
                    if (pos in t1 and neg in t2) or (neg in t1 and pos in t2):
                        reqs[i]["issues"].append("conflict")
                        reqs[j]["issues"].append("conflict")

                # Inconsistency Detection
                nums1 = re.findall(r"\b\d+\b", t1)
                nums2 = re.findall(r"\b\d+\b", t2)

                if nums1 and nums2 and nums1 != nums2:
                    reqs[i]["issues"].append("inconsistency")
                    reqs[j]["issues"].append("inconsistency")

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "running"})

@app.route("/analyze", methods=["POST"])
def analyze():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    # Extract text
    if file.filename.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    else:
        text = file.read().decode("utf-8")

    sentences = [
        s.strip()
        for s in re.split(r'(?<=[.!?;\n])\s+', text)
        if len(s.strip()) > 5
    ]

    results = []

    for i, sent in enumerate(sentences, 1):

        req = {
            "id": f"REQ-{i}",
            "text": sent,
            "issues": [],
            "suggestion": None
        }

        # ML inference
        if ML_MODELS and TFIDF:
            # 1. Get TF-IDF Vector (1000 features)
            tfidf_vec = TFIDF.transform([sent]).toarray()
            
            # 2. Get Heuristic Features (5 features)
            heuristics = extract_features(sent)
            heuristics_vec = np.array([[
                heuristics['modal_count'],
                heuristics['has_numeric'],
                heuristics['has_vague'],
                heuristics['has_incomplete_marker'],
                heuristics['sentence_length']
            ]])
            
            # 3. Combine them to standard 1005 feature structure
            combined_vec = np.hstack((tfidf_vec, heuristics_vec))

            for issue, model in ML_MODELS.items():
                pred = model.predict(combined_vec)
                if pred[0] == 1:
                    req["issues"].append(issue)

        results.append(req)

    # Relational detection
    check_conflicts_inconsistencies(results)

    # Stats + LLM
    stats = {
        "ambiguity": 0,
        "verifiability": 0,
        "incompleteness": 0,
        "conflict": 0,
        "inconsistency": 0,
        "total": len(results)
    }

    # Calculate stats
    for r in results:
        r["issues"] = list(set(r["issues"]))
        for issue in r["issues"]:
            if issue in stats:
                stats[issue] += 1

    # Select up to 15 issues for LLM improvement, diversifying across issue types
    MAX_LLM_CALLS = 15
    llm_calls = 0
    issue_queues = {k: [] for k in ["ambiguity", "verifiability", "incompleteness", "conflict", "inconsistency"]}
    
    for r in results:
        if r["issues"]:
            # Group by the first listed issue
            issue_type = r["issues"][0]
            if issue_type in issue_queues:
                issue_queues[issue_type].append(r)
            
    selected_for_llm = []
    # Pop one from each issue category round-robin until we fill our quota
    while llm_calls < MAX_LLM_CALLS:
        added = False
        for k in issue_queues.keys():
            if issue_queues[k] and llm_calls < MAX_LLM_CALLS:
                selected_for_llm.append(issue_queues[k].pop(0))
                llm_calls += 1
                added = True
        if not added:
            # All queues empty
            break
            
    for r in selected_for_llm:
        r["suggestion"] = get_llm_suggestion(r["text"], r["issues"])

    # Quality Score
    penalty = (
        2 * stats["ambiguity"] +
        3 * stats["inconsistency"] +
        4 * stats["conflict"] +
        2 * stats["incompleteness"] +
        3 * stats["verifiability"]
    )

    if stats["total"] > 0:
        score = max(0, min(100, 100 - int((penalty / stats["total"]) * 10)))
    else:
        score = 0

    stats["score"] = score

    return jsonify({
        "requirements": results,
        "stats": stats
    })


# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)