import os
import uuid
import numpy as np
import faiss
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import PyPDF2

app = Flask(__name__)

BASE_UPLOAD_FOLDER = "uploads"
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------- PDF / TXT READER ----------------
def read_resume(file_path: str) -> str:
    if file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if file_path.lower().endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text

    return ""


# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None

    if request.method == "POST":
        job_description = request.form.get("job_description", "").strip()
        files = request.files.getlist("resumes")

        if not job_description or not files:
            error = "Please provide a job description and upload a resume folder."
            return render_template("index.html", error=error, results=None)

        # Create a unique folder for this request (prevents Windows file-lock errors)
        session_id = str(uuid.uuid4())
        session_folder = os.path.join(BASE_UPLOAD_FOLDER, session_id)
        os.makedirs(session_folder, exist_ok=True)

        resume_texts = []
        resume_names = []

        for file in files:
            if file.filename == "":
                continue

            filename = os.path.basename(file.filename)
            save_path = os.path.join(session_folder, filename)
            file.save(save_path)

            text = read_resume(save_path)
            if text.strip():
                resume_texts.append(text)
                resume_names.append(filename)

        if not resume_texts:
            error = "No readable resumes found (only PDF or TXT supported)."
            return render_template("index.html", error=error, results=None)

        # --------- VECTORIZE RESUMES ---------
        resume_embeddings = model.encode(resume_texts)
        resume_embeddings = np.array(resume_embeddings).astype("float32")
        faiss.normalize_L2(resume_embeddings)

        dimension = resume_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(resume_embeddings)

        # --------- VECTORIZE JOB DESCRIPTION ---------
        job_embedding = model.encode([job_description])
        job_embedding = np.array(job_embedding).astype("float32")
        faiss.normalize_L2(job_embedding)

        # --------- SEARCH ---------
        top_k = min(5, len(resume_texts))
        distances, indices = index.search(job_embedding, top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            score = float(distances[0][rank]) * 100
            results.append({
                "rank": rank + 1,
                "filename": resume_names[idx],
                "score": round(score, 2),
                "preview": resume_texts[idx][:500]
            })

    return render_template("index.html", results=results, error=error)


if __name__ == "__main__":
    app.run(debug=True)
