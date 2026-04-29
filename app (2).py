import streamlit as st
import easyocr
import json
import os
import time
from google import genai
from google.genai.errors import ClientError, ServerError

st.set_page_config(page_title="Document Intelligence Analyzer", page_icon="📄", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0a0a0a; color: #f0f0f0; }
.title { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; }
.subtitle { font-family: 'DM Mono', monospace; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; color: #555; margin-bottom: 2rem; }
.tag { display: inline-block; background: #e8ff00; color: #000; font-family: 'DM Mono', monospace; font-size: 0.65rem; font-weight: 600; padding: 2px 8px; letter-spacing: 0.08em; margin-right: 4px; margin-bottom: 0.8rem; }
.field-key { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.08em; }
.field-val { font-size: 0.95rem; color: #f0f0f0; margin-bottom: 0.8rem; }
.stButton > button { background: #e8ff00 !important; color: #000 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; border: none !important; border-radius: 2px !important; text-transform: uppercase !important; width: 100%; }
.stButton > button:hover { background: #ff6b35 !important; color: #fff !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown('<span class="tag">AI</span><span class="tag">OCR</span>', unsafe_allow_html=True)
st.markdown('<div class="title">Document Intelligence<br>Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">EasyOCR + Gemini · Extract · Analyze · Export</div>', unsafe_allow_html=True)

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

@st.cache_resource
def load_genai():
    key = os.environ.get("GEMINI_KEY") or st.secrets.get("GEMINI_KEY")
    return genai.Client(api_key=key)

MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

def analyze_document(text, client):
    prompt = f"""Analyze the following OCR-extracted text and return a JSON object with:
- document_type: (e.g. ID Card, Invoice, Certificate, Receipt)
- fields: all key-value pairs found in the document
- summary: one line description of the document
- confidence: high / medium / low
Return ONLY valid JSON, no extra text or markdown.
OCR Text:
{text}"""
    for model in MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except ServerError as e:
                if hasattr(e, 'code') and e.code == 503:
                    time.sleep(10 * (attempt + 1))
                else:
                    raise
            except ClientError as e:
                if hasattr(e, 'code') and e.code == 429:
                    break
                raise
    return None

uploaded_file = st.file_uploader("Upload Document Image", type=["png", "jpg", "jpeg", "webp", "bmp"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Document", use_container_width=True)
    if st.button("Analyze Document"):
        with open("temp_doc.png", "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("Extracting text with OCR..."):
            reader = load_ocr()
            result = reader.readtext("temp_doc.png", detail=0)
            extracted = "\n".join(result)
        if not extracted.strip():
            st.error("No text found in the image. Try a clearer document photo.")
        else:
            with st.expander("📝 Extracted Text (OCR)", expanded=False):
                st.code(extracted, language=None)
            with st.spinner("Analyzing with Gemini AI..."):
                client = load_genai()
                raw = analyze_document(extracted, client)
            if raw:
                clean = raw.strip().replace("```json", "").replace("```", "").strip()
                try:
                    data = json.loads(clean)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Document Type", data.get("document_type", "Unknown"))
                    col2.metric("Confidence", data.get("confidence", "—").capitalize())
                    col3.metric("Fields Found", len(data.get("fields", {})))
                    st.markdown(f"**Summary:** {data.get('summary', '')}")
                    st.divider()
                    st.markdown("#### Extracted Fields")
                    for key, value in data.get("fields", {}).items():
                        st.markdown(f'<div class="field-key">{key}</div><div class="field-val">{value}</div>', unsafe_allow_html=True)
                    st.divider()
                    st.download_button(label="⬇ Download JSON", data=clean, file_name="result.json", mime="application/json")
                except json.JSONDecodeError:
                    st.error("Could not parse AI response.")
                    st.code(raw)
            else:
                st.error("Analysis failed. Check your GEMINI_KEY secret.")
else:
    st.info("👆 Upload a document image to get started")

st.markdown("---")
st.markdown('<div style="text-align:center; font-family: monospace; font-size: 0.7rem; color: #333; letter-spacing: 0.1em;">BUILT WITH EASYOCR + GEMINI API</div>', unsafe_allow_html=True)
