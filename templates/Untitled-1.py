"""
Roadmap AI — Flask + Ollama Backend
=====================================
Run:  python app.py
Then: open http://localhost:5000 in your browser

Requirements:
  pip install flask requests
  Ollama must be running:  ollama serve
  Model must be pulled:    ollama pull llama3.2
"""

from flask import Flask, request, jsonify, render_template
import requests
import json
import pypdf

app = Flask(__name__)

# ── Ollama config ──────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.2"          # Change to "llama3", "mistral", etc. if you prefer

# ── Career Roadmap Knowledge Base ─────────────────────────
CAREER_DATA = [
    {
        "career": "Software Developer",
        "roadmap": [
            "Learn Java or Python",
            "Learn Data Structures and Algorithms",
            "Build real-world projects",
            "Apply for internships"
        ],
        "salary": "4-12 LPA"
    },
    {
        "career": "Data Scientist",
        "roadmap": [
            "Learn Python",
            "Learn Statistics and Math",
            "Learn Pandas and NumPy",
            "Learn Machine Learning"
        ],
        "salary": "6-15 LPA"
    },
    {
        "career": "Cyber Security",
        "roadmap": [
            "Learn Networking basics",
            "Learn Linux",
            "Understand Ethical Hacking",
            "Prepare for security certifications"
        ],
        "salary": "5-14 LPA"
    }
]

# Pre-format career data as a readable string for the system prompt
def format_career_knowledge():
    lines = []
    for item in CAREER_DATA:
        lines.append(f"Career: {item['career']}")
        lines.append(f"  Salary: {item['salary']}")
        lines.append(f"  Roadmap steps:")
        for i, step in enumerate(item['roadmap'], 1):
            lines.append(f"    {i}. {step}")
        lines.append("")
    return "\n".join(lines)

CAREER_KNOWLEDGE = format_career_knowledge()

# ── System Prompt ──────────────────────────────────────────
SYSTEM_PROMPT = f"""You are Roadmap AI — a helpful career guidance chatbot specializing in tech career roadmaps.

You have knowledge about the following career paths:

{CAREER_KNOWLEDGE}

Rules you must follow:
1. Answer questions about career roadmaps, skills needed, salaries, and learning paths.
2. Give clear, structured, and helpful answers.
3. Keep responses concise (under 200 words) unless a detailed roadmap is requested.
4. When listing steps, use numbered lists.
5. If someone asks about a career not in your knowledge base, give a helpful general answer.
6. Be friendly, encouraging, and professional.
7. Do NOT go off-topic. Stay focused on careers, roadmaps, and tech skills.
8. Do NOT repeat the question back. Answer directly.
"""

# ── Routes ────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the main chat interface."""
    return render_template("index.html")


@app.route("/careers", methods=["GET"])
def get_careers():
    """Return the career data as JSON (used by frontend suggestion cards)."""
    return jsonify(CAREER_DATA)


def call_ollama(prompt, max_tokens=250):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            },
            timeout=60
        )
        r.raise_for_status()
        reply = r.json().get("response", "").strip()
        return reply if reply else "I couldn't generate a response. Please try again."
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to Ollama. Make sure Ollama is running: run `ollama serve` in your terminal."
    except requests.exceptions.Timeout:
        return "⚠️ The AI took too long to respond. Please try again."
    except Exception as e:
        print(f"[ERROR] Ollama call: {e}")
        return "⚠️ Something went wrong on the server. Please try again."

@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.
    Accepts: { "message": "user text" }
    Returns: { "response": "AI reply" }
    """
    data         = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please type a message."}), 400

    # Build the full prompt with system context
    prompt = f"""{SYSTEM_PROMPT}

User: {user_message}
Roadmap AI:"""

    reply = call_ollama(prompt)
    return jsonify({"response": reply})

@app.route("/analyze-resume", methods=["POST"])
def analyze_resume():
    if 'resume' not in request.files:
        return jsonify({"response": "No resume file provided."}), 400
    file = request.files['resume']
    role = request.form.get("role", "Software Developer")
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({"response": "Invalid file. Please upload a PDF."}), 400
    
    try:
        reader = pypdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Limit text if it's crazily long
        text = text[:4000]
        prompt = f"""{SYSTEM_PROMPT}

The user wants to become a {role}. Here is the parsed text of their resume:
{text}

Evaluate this resume for the {role} role. Give an estimated % readiness, list missing skills, and provide brief actionable advice.
Roadmap AI:"""
        
        reply = call_ollama(prompt, max_tokens=300)
        return jsonify({"response": reply})
    except Exception as e:
        print(f"[ERROR] Resume analysis: {e}")
        return jsonify({"response": "⚠️ Failed to parse or analyze the PDF. Make sure pypdf is installed."}), 500

@app.route("/check-skill-gap", methods=["POST"])
def check_skill_gap():
    data = request.get_json(silent=True) or {}
    role = data.get("role", "Software Developer")
    skills = data.get("skills", "").strip()
    if not skills:
        return jsonify({"response": "Please provide your current skills."}), 400
    
    prompt = f"""{SYSTEM_PROMPT}

The user wants to become a {role}. Their current skills are: {skills}.

Based on typical industry requirements for {role}, identify exact skills they are missing, focusing on high-priority tools and frameworks.
Roadmap AI:"""
    
    reply = call_ollama(prompt, max_tokens=250)
    return jsonify({"response": reply})

@app.route("/estimate-time", methods=["POST"])
def estimate_time():
    data = request.get_json(silent=True) or {}
    missing_skills = data.get("missing_skills", "").strip()
    hours = data.get("hours", 0)
    try:
        hours = int(hours)
    except ValueError:
        hours = 2
    if not missing_skills or hours <= 0:
        return jsonify({"response": "Please provide missing skills and a valid number of daily hours."}), 400
    
    prompt = f"""{SYSTEM_PROMPT}

The user needs to learn these skills: {missing_skills}. They plan to study {hours} hours per day.

Estimate a realistic timeline (in weeks or months) for an average learner to become job-ready, breaking down the time roughly by skill chunk. Explain the estimation briefly.
Roadmap AI:"""
    
    reply = call_ollama(prompt, max_tokens=250)
    return jsonify({"response": reply})


# ── Health check endpoint ──────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Check if Ollama is reachable."""
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        return jsonify({"status": "ok", "ollama": "reachable", "model": MODEL})
    except Exception:
        return jsonify({"status": "error", "ollama": "unreachable", "model": MODEL}), 503


if __name__ == "__main__":
    print("=" * 50)
    print("  Roadmap AI — Flask Server")
    print("  URL:   http://localhost:5000")
    print("  Model: " + MODEL)
    print("  Make sure `ollama serve` is running!")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)