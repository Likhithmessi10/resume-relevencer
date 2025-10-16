import streamlit as st
import pdfplumber
from docx import Document
import re
import json
import os
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import pandas as pd
from pymongo import MongoClient
from groq import Groq
import itertools

st.set_page_config(page_title="Resume Relevancer", layout="wide")
# ======================
# Helpers
# ======================
def extract_text_from_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(path):
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text)
    return text

def clean_text(text):
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()

def compute_hard_score(resume_text, must_have, good_to_have):
    resume_text = resume_text.lower()
    must_matches = [fuzz.partial_ratio(skill.lower(), resume_text)/100 for skill in must_have]
    good_matches = [fuzz.partial_ratio(skill.lower(), resume_text)/100 for skill in good_to_have]

    must_score = sum(must_matches)/len(must_matches) if must_matches else 1.0
    good_score = sum(good_matches)/len(good_matches) if good_matches else 1.0

    combined = 0.75*must_score + 0.25*good_score
    return round(combined * 100, 2)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

def semantic_score(model, resume_text, jd_text):
    r_vec = model.encode(resume_text, convert_to_numpy=True)
    jd_vec = model.encode(jd_text, convert_to_numpy=True)
    cos_sim = np.dot(r_vec, jd_vec) / (norm(r_vec)*norm(jd_vec))
    score = ((cos_sim + 1)/2) * 100
    return round(float(score), 2)

def final_score(hard, semantic):
    return round(0.6*hard + 0.4*semantic, 2)

def verdict(score):
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

def parse_jd_skills(jd_text):
    jd_text = jd_text.lower()
    must_have = []
    good_to_have = []

    skill_sections_pattern = r"(skills|requirements|key skills|experience|responsibilities|qualifications|will need|must have|ideal candidate|about you|what you'll bring)\s*[:\n]"
    
    skill_section_match = re.search(skill_sections_pattern, jd_text)
    if not skill_section_match:
        return [], []
        
    start_pos = skill_section_match.end()
    end_match = re.search(skill_sections_pattern, jd_text[start_pos:])
    if end_match:
        end_pos = start_pos + end_match.start()
        skills_text = jd_text[start_pos:end_pos].strip()
    else:
        skills_text = jd_text[start_pos:].strip()
        
    skills_list = re.split(r'[,•\-*]', skills_text)
    skills_list = [s.strip() for s in skills_list if s.strip()]

    for skill_phrase in skills_list:
        if any(keyword in skill_phrase for keyword in ['advantageous', 'preferred', 'nice to have']):
            clean_skill = re.sub(r'\(?advantageous\)?|\(preferred\)|\(nice to have\)|knowledge of', '', skill_phrase).strip()
            if clean_skill:
                good_to_have.append(clean_skill)
        else:
            if skill_phrase:
                must_have.append(skill_phrase)
    
    return must_have, good_to_have

# ======================
# MongoDB Atlas Functions
# ======================
def get_db_client():
    connection_string = st.secrets["mongodb"]["uri"]
    client = MongoClient(connection_string)
    db = client.get_database("resume_relevance_db")
    return db

def save_result(result):
    db = get_db_client()
    collection = db.get_collection("evaluations")
    collection.replace_one({"Resume": result["Resume"]}, result, upsert=True)

def get_all_results():
    db = get_db_client()
    collection = db.get_collection("evaluations")
    results = list(collection.find())
    df = pd.DataFrame(results)
    
    if '_id' in df.columns:
        df['_id'] = df['_id'].astype(str)

    if 'Job Title' in df.columns:
        df['Job Title'] = df['Job Title'].fillna('')
    else:
        df['Job Title'] = ''
        
    return df

@st.cache_resource
def get_api_key_cycle():
    """Returns a cycle iterator of Groq API keys from Streamlit secrets."""
    try:
        groq_secrets = st.secrets.get("groq", {})
        api_keys = groq_secrets.get("api_keys") or groq_secrets.get("api_key")

        if not api_keys or not isinstance(api_keys, list):
            st.error("❌ Groq API keys not found or improperly formatted in secrets.toml.")
            return None

        #st.success(f"✅ Loaded {len(api_keys)} Groq API key(s).")
        return itertools.cycle(api_keys)

    except Exception as e:
        st.error(f"❌ Error loading Groq API keys: {e}")
        return None



def generate_feedback_groq(resume_text, jd_text, missing_skills):
    api_key_cycle = get_api_key_cycle()
    if api_key_cycle is None:
        return "Feedback could not be generated due to missing API keys."

    max_retries = 4
    attempt = 0
    last_error = None

    for attempt in range(max_retries):
        try:
            api_key = next(api_key_cycle)
        except StopIteration:
            st.error("No more Groq API keys available.")
            break

        client = Groq(api_key=api_key)

        prompt = f"""
You are a career counselor. Based on the resume and job description, provide a single, concise sentence that summarizes the resume's fit. Highlight their strengths and any key missing skills.

Job Description:
---
{jd_text}
---

Resume:
---
{resume_text}
---

Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}

One-sentence summary:
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=100,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            st.warning(f"[Attempt {attempt + 1}] API key failed: {e}. Trying next key...")

    st.error("All Groq API keys failed. Feedback could not be generated.")
    return "Feedback could not be generated at this time."


# ======================
# Streamlit UI
# ======================
st.title("Resume Relevance Dashboard")

st.sidebar.header("Job Description")
uploaded_jd = st.sidebar.file_uploader("Upload Job Description", type=["pdf", "docx"])
uploaded_resumes = st.file_uploader("Upload Multiple Resumes", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_jd and uploaded_resumes:
    jd_title = uploaded_jd.name.split('.')[0]
    
    jd_temp_path = os.path.join("resumes", uploaded_jd.name)
    os.makedirs("resumes", exist_ok=True)
    with open(jd_temp_path, "wb") as f:
        f.write(uploaded_jd.getbuffer())

    jd_text = extract_text_from_pdf(jd_temp_path) if uploaded_jd.name.lower().endswith(".pdf") else extract_text_from_docx(jd_temp_path)
    jd_text = clean_text(jd_text)

    must_list, good_list = parse_jd_skills(jd_text)
    
    if not must_list and not good_list:
        st.warning("Could not find relevant skills in the Job Description. Please check its formatting.")
    else:
        st.sidebar.success(f"Parsed {len(must_list)} must-have skills and {len(good_list)} good-to-have skills.")
        
    model = load_model()

    results = []
    for resume in uploaded_resumes:
        temp_path = os.path.join("resumes", resume.name)
        with open(temp_path, "wb") as f:
            f.write(resume.getbuffer())

        resume_text = extract_text_from_pdf(temp_path) if resume.name.lower().endswith(".pdf") else extract_text_from_docx(temp_path)
        resume_text = clean_text(resume_text)

        hard = compute_hard_score(resume_text, must_list, good_list)
        sem = semantic_score(model, resume_text, jd_text)
        score = final_score(hard, sem)
        v = verdict(score)

        missing = [s for s in must_list if max(fuzz.partial_ratio(s.lower(), resume_text.lower()) for _ in [0]) < 70]
        
        with st.spinner(f"Generating personalized feedback for {resume.name}..."):
            feedback = generate_feedback_groq(resume_text, jd_text, missing)

        result = {
            "Resume": resume.name,
            "Job Title": jd_title,
            "Hard Score": hard,
            "Semantic Score": sem,
            "Final Score": score,
            "Verdict": v,
            "Missing Skills": ", ".join(missing) if missing else "None",
            "Feedback": feedback
        }
        
        save_result(result)
        results.append(result)

    st.success("Resumes evaluated!!!")

# Show table from database
df = get_all_results()

if not df.empty:
    st.subheader("Results Table")

    search_query = st.text_input("Search by Resume or Feedback Keywords", "")
    job_titles = ["All"] + sorted(df["Job Title"].unique().tolist())
    job_title_filter = st.selectbox("Filter by Job Title", job_titles)

    filtered_df = df
    if job_title_filter != "All":
        filtered_df = filtered_df[filtered_df["Job Title"] == job_title_filter]

    if search_query:
        filtered_df = filtered_df[
            filtered_df["Resume"].str.contains(search_query, case=False, na=False) |
            filtered_df["Feedback"].str.contains(search_query, case=False, na=False)
        ]

    df_display = filtered_df.drop(columns=['_id'], errors='ignore')
    # Reset index and start from 1 instead of 0
    df_display = df_display.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
    df_display.index = df_display.index + 1  # Start index from 1

    st.dataframe(df_display, width='stretch')


    for index, res in filtered_df.iterrows():
        with st.expander(f"**Feedback for {res['Resume']}**"):
            st.markdown(res['Feedback'])
    
    st.subheader("Other Filters")
    min_score = st.slider("Minimum Final Score", 0, 100, 0)
    verdict_filter = st.selectbox("Verdict Filter", ["All", "High", "Medium", "Low"])

    filtered_df2 = filtered_df[filtered_df["Final Score"] >= min_score]
    if verdict_filter != "All":
        filtered_df2 = filtered_df2[filtered_df2["Verdict"] == verdict_filter]

    # Drop _id and keep only columns you want to display
    filtered_df2 = filtered_df2.drop(columns=['_id'], errors='ignore')
    columns_to_show = ["Resume", "Job Title", "Hard Score", "Semantic Score", "Final Score", "Verdict", "Missing Skills", "Feedback"]
    filtered_df2 = filtered_df2[[col for col in columns_to_show if col in filtered_df2.columns]]

    filtered_df2 = filtered_df2.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
    filtered_df2.index = filtered_df2.index + 1

    st.dataframe(filtered_df2, use_container_width=True)


