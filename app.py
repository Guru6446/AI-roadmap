"""
Roadmap AI — Flask + Ollama Backend
========================================
Features:
  1. /chat          — Main AI career chatbot
  2. /analyze-resume — Resume PDF analyzer
  3. /skill-gap     — Skill gap checker
  4. /time-estimate — Learning time estimator
  5. /careers       — Career data JSON
  6. /health        — Ollama connection check

Run:
  pip install flask requests PyMuPDF
  ollama pull llama3.2
  ollama serve
  python app.py
  Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import requests
import re
import fitz   # PyMuPDF — for reading PDF resumes

app = Flask(__name__)

# ── Ollama config ──────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.2"   # Change to "phi" if llama3.2 not available

# ── Career Knowledge Base ──────────────────────────────────
CAREER_DATA = [
    {
        "career": "Software Developer",
        "skills": ["Python or Java", "Data Structures", "Algorithms", "Git",
                   "HTML/CSS", "JavaScript", "SQL", "REST APIs", "Problem Solving"],
        "roadmap": [
            "Learn Python or Java basics",
            "Learn Data Structures and Algorithms",
            "Learn Git and version control",
            "Build 3-5 real-world projects",
            "Learn a web framework (Django/Spring)",
            "Apply for internships or jobs"
        ],
        "salary": "4-12 LPA",
        "hours_to_ready": 800   # total study hours to be job-ready
    },
    {
        "career": "Data Scientist",
        "skills": ["Python", "Statistics", "Mathematics", "Pandas", "NumPy",
                   "Matplotlib", "Machine Learning", "Scikit-learn", "SQL",
                   "Deep Learning", "Data Visualization", "Feature Engineering"],
        "roadmap": [
            "Learn Python programming",
            "Learn Statistics and Mathematics",
            "Learn Pandas, NumPy, Matplotlib",
            "Learn Machine Learning with Scikit-learn",
            "Work on Kaggle datasets and projects",
            "Learn Deep Learning basics",
            "Apply for data science roles"
        ],
        "salary": "6-15 LPA",
        "hours_to_ready": 1000
    },
    {
        "career": "Cyber Security",
        "skills": ["Networking", "Linux", "Python", "Ethical Hacking",
                   "Firewall", "Cryptography", "Penetration Testing",
                   "OWASP", "Security Certifications (CEH/CISSP)", "Incident Response"],
        "roadmap": [
            "Learn Networking basics (TCP/IP, DNS, HTTP)",
            "Learn Linux command line",
            "Learn Python for scripting",
            "Understand ethical hacking concepts",
            "Practice on platforms like TryHackMe / HackTheBox",
            "Prepare for CEH or CompTIA Security+ certification",
            "Apply for security analyst roles"
        ],
        "salary": "5-14 LPA",
        "hours_to_ready": 900
    },
    {
        "career": "Web Developer",
        "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js",
                   "REST APIs", "SQL", "Git", "Responsive Design", "TypeScript"],
        "roadmap": [
            "Learn HTML and CSS fundamentals",
            "Learn JavaScript thoroughly",
            "Learn React or Vue.js for frontend",
            "Learn Node.js and Express for backend",
            "Learn database basics (MySQL/MongoDB)",
            "Build and deploy full-stack projects",
            "Apply for web developer roles"
        ],
        "salary": "4-10 LPA",
        "hours_to_ready": 700
    },
    {
        "career": "AI/ML Engineer",
        "skills": ["Python", "Mathematics", "Machine Learning", "Deep Learning",
                   "TensorFlow", "PyTorch", "NLP", "Computer Vision",
                   "Data Processing", "Cloud Platforms (AWS/GCP)", "MLOps"],
        "roadmap": [
            "Master Python and Mathematics",
            "Learn Machine Learning fundamentals",
            "Learn Deep Learning with TensorFlow/PyTorch",
            "Study NLP and Computer Vision",
            "Work on AI projects and research papers",
            "Learn MLOps and model deployment",
            "Apply for AI/ML roles"
        ],
        "salary": "8-25 LPA",
        "hours_to_ready": 1200
    }
]

def format_career_knowledge():
    lines = []
    for item in CAREER_DATA:
        lines.append(f"Career: {item['career']}")
        lines.append(f"  Salary Range: {item['salary']}")
        lines.append(f"  Required Skills: {', '.join(item['skills'])}")
        lines.append(f"  Roadmap:")
        for i, step in enumerate(item['roadmap'], 1):
            lines.append(f"    Step {i}: {step}")
        lines.append("")
    return "\n".join(lines)

CAREER_KNOWLEDGE = format_career_knowledge()

# ── System Prompt ──────────────────────────────────────────
SYSTEM_PROMPT = f"""You are Roadmap AI — a smart, friendly career guidance assistant like ChatGPT, specialized in tech careers.

CAREER DATABASE:
{CAREER_KNOWLEDGE}

YOUR STYLE:
- Friendly, clear, structured like a knowledgeable mentor
- Always use numbered steps, bullet points, headings
- Be specific: tools, technologies, timelines, resources
- Never give one-line vague answers

HOW TO ANSWER:
1. Career questions → Full roadmap + salary + tips
2. Technology questions → What it is + learning path + resources
3. Timeline questions → Realistic months with study hours
4. Salary questions → LPA ranges + top hiring companies
5. All tech topics → Detailed helpful answer always

RULES:
- NEVER refuse a tech/career question
- NEVER be vague — always be specific
- End with encouragement or offer to help more
- Redirect off-topic questions gently back to careers
"""

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def call_ollama(prompt, max_tokens=700):
    """Send prompt to Ollama and return response text."""
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model":  MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature":    0.6,
                "num_predict":    max_tokens,
                "top_p":          0.9,
                "repeat_penalty": 1.1,
                "top_k":          40,
            }
        },
        timeout=180
    )
    resp.raise_for_status()
    return clean_text(resp.json().get("response", ""))

# ═══════════════════════════════════════════════════════════
# ROUTE 1 — Home
# ═══════════════════════════════════════════════════════════
@app.route("/")
def home():
    return render_template("index.html")

# ═══════════════════════════════════════════════════════════
# ROUTE 2 — Career Data
# ═══════════════════════════════════════════════════════════
@app.route("/careers", methods=["GET"])
def get_careers():
    return jsonify(CAREER_DATA)

# ═══════════════════════════════════════════════════════════
# ROUTE 3 — Main Chat
# ═══════════════════════════════════════════════════════════
@app.route("/chat", methods=["POST"])
def chat():
    data         = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please type a message."}), 400

    prompt = f"""{SYSTEM_PROMPT}

User: {user_message}

Roadmap AI:"""

    try:
        reply = call_ollama(prompt, max_tokens=700)
        if not reply:
            reply = "I couldn't generate a response. Please try again."
        return jsonify({"response": reply})

    except requests.exceptions.ConnectionError:
        return jsonify({"response": "⚠️ Cannot connect to Ollama. Run `ollama serve` in your terminal."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"response": "⚠️ Response timed out. Try again or restart Ollama."}), 504
    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        return jsonify({"response": "⚠️ Something went wrong. Please try again."}), 500

# ═══════════════════════════════════════════════════════════
# ROUTE 4 — Resume Analyzer
# ═══════════════════════════════════════════════════════════
@app.route("/analyze-resume", methods=["POST"])
def analyze_resume():
    """
    Accepts: multipart form with 'resume' PDF + optional 'career' field
    Returns: AI analysis of resume vs career requirements
    """
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded."}), 400

    career_goal = request.form.get("career", "Software Developer").strip()
    pdf_file    = request.files["resume"]

    # ── Extract text from PDF using PyMuPDF ──
    try:
        pdf_bytes = pdf_file.read()
        doc       = fitz.open(stream=pdf_bytes, filetype="pdf")
        resume_text = ""
        for page in doc:
            resume_text += page.get_text()
        doc.close()

        if not resume_text.strip():
            return jsonify({"error": "Could not extract text from PDF. Make sure it's a text-based PDF, not scanned."}), 400

        # Limit to 3000 chars to avoid prompt overflow
        resume_text = resume_text[:3000]

    except Exception as e:
        print(f"[PDF ERROR] {e}")
        return jsonify({"error": "Failed to read PDF file. Make sure it's a valid PDF."}), 400

    # ── Find career data ──
    career_info = next((c for c in CAREER_DATA if c["career"].lower() == career_goal.lower()), CAREER_DATA[0])
    required_skills = ", ".join(career_info["skills"])

    # ── Build analysis prompt ──
    prompt = f"""You are a professional resume analyzer and career counselor.

TARGET CAREER: {career_goal}
REQUIRED SKILLS FOR THIS ROLE: {required_skills}
SALARY RANGE: {career_info['salary']}

CANDIDATE'S RESUME:
{resume_text}

Analyze this resume carefully and provide:

**1. READINESS SCORE**
Calculate a percentage (0-100%) of how ready this candidate is for a {career_goal} role.
Be honest and realistic.

**2. SKILLS FOUND IN RESUME**
List all relevant skills you found in the resume that match the {career_goal} requirements.

**3. MISSING SKILLS**
List the important skills NOT found in the resume that are required for {career_goal}.

**4. WHAT TO IMPROVE**
Give 3-5 specific, actionable recommendations to improve this resume for a {career_goal} role.

**5. ROADMAP TO GET THE JOB**
Give a step-by-step plan to fill the skill gaps, with realistic timeframes.

**6. ENCOURAGING SUMMARY**
End with a brief motivational message about their chances and next steps.

Be specific, honest, and helpful. Format clearly with the headings above."""

    try:
        reply = call_ollama(prompt, max_tokens=900)
        if not reply:
            reply = "Could not analyze the resume. Please try again."
        return jsonify({"response": reply, "career": career_goal})

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to Ollama. Run `ollama serve`."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Analysis timed out. Try again."}), 504
    except Exception as e:
        print(f"[RESUME ERROR] {e}")
        return jsonify({"error": "Analysis failed. Please try again."}), 500

# ═══════════════════════════════════════════════════════════
# ROUTE 5 — Skill Gap Checker
# ═══════════════════════════════════════════════════════════
@app.route("/skill-gap", methods=["POST"])
def skill_gap():
    """
    Accepts: { "career": "Data Scientist", "current_skills": "Python, Excel, SQL" }
    Returns: AI skill gap analysis with learning plan
    """
    data           = request.get_json(silent=True) or {}
    career_goal    = data.get("career", "").strip()
    current_skills = data.get("current_skills", "").strip()

    if not career_goal or not current_skills:
        return jsonify({"error": "Please provide both career goal and current skills."}), 400

    # Find career info
    career_info = next(
        (c for c in CAREER_DATA if c["career"].lower() == career_goal.lower()),
        None
    )

    if career_info:
        required_skills = ", ".join(career_info["skills"])
        salary          = career_info["salary"]
        roadmap_steps   = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(career_info["roadmap"])])
    else:
        required_skills = "varies based on role"
        salary          = "depends on experience"
        roadmap_steps   = "Research the specific role requirements"

    prompt = f"""You are a career skills advisor helping a person identify what they need to learn.

THEIR GOAL: Become a {career_goal}
THEIR CURRENT SKILLS: {current_skills}
REQUIRED SKILLS FOR {career_goal}: {required_skills}
SALARY RANGE: {salary}

OFFICIAL ROADMAP:
{roadmap_steps}

Analyze the gap between their current skills and what's needed. Provide:

**1. SKILLS YOU ALREADY HAVE ✅**
From their list, identify which skills are relevant and useful for {career_goal}.

**2. MISSING SKILLS ❌**
List the important skills they are missing for {career_goal}.
For each missing skill, give:
  - Skill name
  - Why it matters for this role
  - Best resource to learn it (YouTube channel / free course)
  - Estimated time to learn (weeks)

**3. SKILL MATCH PERCENTAGE**
Give an honest percentage like: "You are currently X% ready for {career_goal}"

**4. YOUR PERSONAL LEARNING ROADMAP**
Based on their current skills, create a custom step-by-step plan:
  - What to learn first (based on what they already know)
  - What to learn second, third, etc.
  - Estimated total time in weeks

**5. QUICK WINS**
List 2-3 things they can start learning TODAY to make fast progress.

Be specific, encouraging, and practical. Use clear headings and bullet points."""

    try:
        reply = call_ollama(prompt, max_tokens=900)
        if not reply:
            reply = "Could not analyze skill gap. Please try again."
        return jsonify({"response": reply, "career": career_goal})

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to Ollama. Run `ollama serve`."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Analysis timed out. Try again."}), 504
    except Exception as e:
        print(f"[SKILL GAP ERROR] {e}")
        return jsonify({"error": "Analysis failed. Please try again."}), 500

# ═══════════════════════════════════════════════════════════
# ROUTE 6 — Learning Time Estimator
# ═══════════════════════════════════════════════════════════
@app.route("/time-estimate", methods=["POST"])
def time_estimate():
    """
    Accepts: { "career": "Data Scientist", "hours_per_day": 2, "current_skills": "Python basics" }
    Returns: AI-calculated timeline to job readiness
    """
    data           = request.get_json(silent=True) or {}
    career_goal    = data.get("career", "").strip()
    hours_per_day  = float(data.get("hours_per_day", 2))
    current_skills = data.get("current_skills", "none").strip()

    if not career_goal:
        return jsonify({"error": "Please provide a career goal."}), 400

    if hours_per_day <= 0 or hours_per_day > 24:
        return jsonify({"error": "Please enter a valid number of study hours (1-12)."}), 400

    # Find career info for base hours
    career_info = next(
        (c for c in CAREER_DATA if c["career"].lower() == career_goal.lower()),
        None
    )

    base_hours    = career_info["hours_to_ready"] if career_info else 800
    required_skills = ", ".join(career_info["skills"]) if career_info else "varies"
    salary        = career_info["salary"] if career_info else "depends on experience"

    # Quick estimate calculation
    hours_per_week = hours_per_day * 7
    estimated_weeks = base_hours / hours_per_week
    estimated_months = round(estimated_weeks / 4.3, 1)

    prompt = f"""You are a learning timeline expert helping someone plan their career preparation.

THEIR GOAL: Become a {career_goal} (Salary: {salary})
STUDY TIME AVAILABLE: {hours_per_day} hours per day ({hours_per_day * 7:.0f} hours per week)
CURRENT SKILLS: {current_skills}
REQUIRED SKILLS: {required_skills}
BASE ESTIMATE: approximately {estimated_months} months

Create a detailed, personalized learning schedule:

**1. YOUR TIMELINE SUMMARY** 📅
State clearly: "Based on {hours_per_day} hours/day, you will be job-ready in approximately X months"
(Adjust the estimate based on their current skills — if they already know some skills, reduce time)

**2. MONTH-BY-MONTH PLAN** 🗓️
Break down what to learn each month:
  - Month 1: [topic] — [specific tasks to do daily]
  - Month 2: [topic] — [specific tasks to do daily]
  - Continue for all months...

**3. WEEKLY SCHEDULE TEMPLATE** 📋
Show a sample week:
  - Monday-Wednesday: [what to study]
  - Thursday-Friday: [practice/projects]
  - Weekend: [revision/mini projects]

**4. MILESTONES & CHECKPOINTS** 🎯
List 3-5 key milestones to track progress:
  - Milestone 1 (Week X): [achievement]
  - Milestone 2 (Week X): [achievement]
  etc.

**5. PRODUCTIVITY TIPS** ⚡
Give 3 tips to make the most of {hours_per_day} hours/day.

**6. WHEN YOU'LL BE READY** 🏆
Give a realistic date-based summary of when they can start applying for jobs.

Be specific, encouraging, and realistic. Show exact numbers and dates."""

    try:
        reply = call_ollama(prompt, max_tokens=900)
        if not reply:
            reply = "Could not calculate timeline. Please try again."

        return jsonify({
            "response":         reply,
            "career":           career_goal,
            "hours_per_day":    hours_per_day,
            "estimated_months": estimated_months,
            "salary":           salary
        })

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to Ollama. Run `ollama serve`."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Calculation timed out. Try again."}), 504
    except Exception as e:
        print(f"[TIME ERROR] {e}")
        return jsonify({"error": "Calculation failed. Please try again."}), 500

# ═══════════════════════════════════════════════════════════
# ROUTE 7 — Health Check
# ═══════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    try:
        requests.get("http://localhost:11434", timeout=3)
        return jsonify({"status": "ok", "ollama": "reachable", "model": MODEL})
    except Exception:
        return jsonify({"status": "error", "ollama": "unreachable", "model": MODEL}), 503


if __name__ == "__main__":
    print("=" * 60)
    print("  Roadmap AI - Enhanced Flask Server")
    print(f"  Model  : {MODEL}")
    print("  URL    : http://localhost:5000")
    print("  Features:")
    print("    /chat           -> AI Career Chatbot")
    print("    /analyze-resume -> Resume PDF Analyzer")
    print("    /skill-gap      -> Skill Gap Checker")
    print("    /time-estimate  -> Learning Time Estimator")
    print("  Run 'ollama serve' in another terminal first!")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)