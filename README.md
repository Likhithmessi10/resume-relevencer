# 📊 AI-Powered Resume Relevance Analyzer

This project evaluates and ranks resumes based on how well they match a given job description (JD). It combines NLP techniques, semantic similarity models, and Groq API (LLaMA 3.1) to provide a comprehensive evaluation with personalized AI feedback.

---

## 🚀 Features

- ✅ Upload and parse multiple resumes (PDF/DOCX)  
- ✅ Upload a job description (PDF/DOCX)  
- ✅ Extract must-have and good-to-have skills from the JD  
- ✅ Compute two scores: **Hard Skill Score** and **Semantic Similarity**  
- ✅ Generate AI-powered feedback using Groq’s LLaMA 3.1  
- ✅ Store and display results with **MongoDB Atlas**  
- ✅ Interactive Streamlit dashboard with search, filter, and feedback view  

---

## ⚙️ Tech Stack

| Category             | Tools & Libraries                                        |
|----------------------|----------------------------------------------------------|
| **Frontend**         | Streamlit                                                |
| **NLP**              | Sentence Transformers (`all-mpnet-base-v2`), RapidFuzz   |
| **LLM Feedback**     | Groq API (LLaMA 3.1)                                     |
| **Backend Database** | MongoDB Atlas                                            |
| **PDF/DOCX Parsing** | `pdfplumber`, `python-docx`                              |
| **Similarity Metric**| Cosine similarity, Fuzzy Matching                        |

---

## 🧠 Logic Breakdown

### 1. Resume & JD Parsing
- Supports `.pdf` and `.docx` formats.  
- Uses `pdfplumber` and `python-docx` to extract clean text.  

### 2. Skill Extraction
- Parses JD to extract **must-have** and **good-to-have** skills.  
- Uses heuristics based on sections like *Skills* and *Requirements*.  

### 3. Scoring
- **Hard Skill Score** → Fuzzy matching of skills against resume.  
- **Semantic Similarity** → SentenceTransformer comparison of resume & JD meaning.  
- **Final Score** = 60% Hard Skills + 40% Semantic Similarity.  

### 4. AI Feedback
- Groq’s LLaMA 3.1 (API key rotation with up to 4 keys).  
- Generates one-sentence personalized summary highlighting **strengths & missing skills**.  

### 5. Storage & Dashboard
- Results stored in MongoDB Atlas.  
- Streamlit dashboard with:
  - Search & filter  
  - Indexed results (starting from 1)  
  - Detailed feedback view  

---

## 🛠️ Local Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/resume-relevance-analyzer.git
cd resume-relevance-analyzer
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Setup Secrets
Create `.streamlit/secrets.toml`:

```toml
[mongodb]
uri = "your_mongodb_connection_string"

[groq]
api_keys = [
  "your_first_groq_api_key",
  "your_second_groq_api_key",
  "your_third_groq_api_key",
  "your_fourth_groq_api_key"
]
```

🔐 **Tip**: Never commit API keys to version control.

### 5. Run the App
```bash
streamlit run main_app.py
```

---

## 📁 Directory Structure
```
resume-relevance-analyzer/
│
├── main_app.py                # Main Streamlit app
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── secrets.toml           # API Keys and MongoDB URI
├── resumes/                   # Temporary storage for uploads
└── README.md                  # Documentation
```

---

## 🌐 Deployment Instructions

### Deploy on **Streamlit Cloud**
1. Push your code to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and link your repo.
3. Add `secrets.toml` content manually in the Streamlit secret manager.
4. Click **Deploy** → App goes live 🚀

---

## 🎨 Theme Customization

Add `.streamlit/config.toml`:

```toml
[theme]
base="dark"
primaryColor="#4B8BBE"
backgroundColor="#1E1E1E"
secondaryBackgroundColor="#2C2C2C"
textColor="#FFFFFF"
```

---

## 🧹 Optional: Clear MongoDB Collection
Run this to clear stored evaluations:

```python
from pymongo import MongoClient

client = MongoClient("your_mongo_uri")
db = client.get_database("resume_relevance_db")
db.evaluations.delete_many({})
```

---

## 🔐 Security & Best Practices
- Use `.gitignore` to avoid uploading sensitive files.
- Store secrets only in `secrets.toml` or Streamlit Cloud secrets.
- Use virtual environments for isolated dependencies.

---

## 📌 Future Enhancements
- 🔍 Support for image-based PDFs (OCR with Tesseract)
- 🌍 Multi-language support
- 🏷️ Auto skill tagging via Named Entity Recognition (NER)
- 📊 Export results to CSV/Excel
- 📧 Email feedback to candidates

---
