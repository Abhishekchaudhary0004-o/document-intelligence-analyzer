import gradio as gr
import easyocr
import json
from google import genai
from google.genai.errors import ClientError, ServerError
import time
import os

# Initialize
reader = easyocr.Reader(['en'])
client = genai.Client(api_key=os.environ.get("GEMINI_KEY"))

MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]


def extract_text(image_path):
    result = reader.readtext(image_path, detail=0)
    return "\n".join(result)


def analyze_document(extracted_text):
    prompt = f"""You are a smart document analyzer.
Analyze the following OCR-extracted text and return a JSON object with:
- document_type: (e.g. ID Card, Invoice, Certificate, Receipt)
- fields: all key-value pairs found in the document
- summary: one line description of the document
- confidence: high / medium / low

Return ONLY valid JSON, no extra text.

OCR Text:
{extracted_text}"""

    for model in MODELS:
        for attempt in range(5):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except ServerError as e:
                code = e.code if hasattr(e, 'code') else None
                if code == 503:
                    wait = 15 * (attempt + 1)
                    time.sleep(wait)
                else:
                    raise
            except ClientError as e:
                code = e.code if hasattr(e, 'code') else None
                if code == 429:
                    break
                raise
    return None


def process_document(image):
    if image is None:
        return "Please upload an image.", "", []

    # OCR
    extracted = extract_text(image)
    if not extracted.strip():
        return "No text found in image.", "", []

    # Analyze
    raw = analyze_document(extracted)
    if not raw:
        return extracted, "Analysis failed. Try again.", []

    clean = raw.strip().replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        return extracted, "Could not parse response.", []

    doc_type = data.get("document_type", "Unknown")
    summary = data.get("summary", "")
    confidence = data.get("confidence", "")
    fields = data.get("fields", {})

    info = f"**Type:** {doc_type}  \n**Summary:** {summary}  \n**Confidence:** {confidence}"
    table = [[k, v] for k, v in fields.items()]

    return extracted, info, table


# ── UI ──────────────────────────────────────────────────────────────────────

css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg: #0a0a0a;
    --surface: #111111;
    --border: #222222;
    --accent: #e8ff00;
    --accent2: #ff6b35;
    --text: #f0f0f0;
    --muted: #666;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    color: var(--text) !important;
    margin-bottom: 0 !important;
}

.subtitle {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.gr-panel, .gr-box, .gr-form {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
}

.gr-button-primary {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 2px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

.gr-button-primary:hover {
    background: var(--accent2) !important;
    color: #fff !important;
}

label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--muted) !important;
}

textarea, input {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    border-color: var(--border) !important;
}

.tag {
    display: inline-block;
    background: var(--accent);
    color: #000;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 2px 8px;
    letter-spacing: 0.08em;
    margin-right: 6px;
}
"""

with gr.Blocks(css=css, title="DIA — Document Intelligence Analyzer") as demo:

    gr.HTML("""
        <div style="padding: 2rem 0 1rem">
            <span class="tag">AI</span><span class="tag">OCR</span>
            <h1>Document Intelligence<br>Analyzer</h1>
            <p class="subtitle">EasyOCR + Gemini · Extract · Analyze · Export</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Upload Document Image")
            analyze_btn = gr.Button("Analyze Document", variant="primary", size="lg")

        with gr.Column(scale=1):
            ocr_output = gr.Textbox(label="Extracted Text (OCR)", lines=6)
            info_output = gr.Markdown(label="Analysis")
            fields_output = gr.Dataframe(
                headers=["Field", "Value"],
                label="Extracted Fields",
                wrap=True
            )

    analyze_btn.click(
        fn=process_document,
        inputs=[image_input],
        outputs=[ocr_output, info_output, fields_output]
    )

    gr.HTML("""
        <div style="text-align:center; padding: 2rem 0 1rem; font-family: 'DM Mono', monospace;
                    font-size: 0.7rem; color: #444; letter-spacing: 0.1em; text-transform: uppercase;">
            Built with EasyOCR + Gemini API
        </div>
    """)

demo.launch()
