import io
import numpy as np
import streamlit as st
import torch
from matplotlib import cm
from PIL import Image
from torchvision import transforms as T
from model import HydraNet, IMG, MEAN, STD
from gradcam import gradcam
from file_check import file_evidence, verdict_label

st.set_page_config(page_title="Hydra | AI Image Forensics", page_icon="🐍", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#183044; --muted:#637b8a; --line:#d9e8e5; --panel:#ffffff; --mint:#168c77; --blue:#4a83c6; }
.stApp { background:radial-gradient(ellipse at 5% 0%,#e3f8f0 0,transparent 35%),radial-gradient(ellipse at 100% 9%,#e9f1ff 0,transparent 32%),#f5f8f7; color:var(--ink); font-family:'DM Sans',sans-serif; }
.block-container { max-width:1250px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; color:#17364a; }
.hero { border:1px solid #d6ebe4; border-radius:25px; padding:32px 36px; background:linear-gradient(115deg,rgba(255,255,255,.96),rgba(237,249,245,.97) 58%,rgba(239,244,255,.97)); box-shadow:0 14px 45px rgba(42,91,80,.09); margin-bottom:25px; }
.eyebrow { color:#168c77; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
.hero-title { font:700 clamp(2rem,4vw,3.15rem)/1.1 'Space Grotesk',sans-serif; color:#17364a; margin:.45rem 0 .7rem; }
.hero-copy { color:#536d7b; font-size:1rem; margin:0; max-width:800px; }
.chips { display:flex; flex-wrap:wrap; gap:9px; margin-top:20px; }
.chip { padding:7px 11px; border:1px solid #d6e8e2; background:rgba(255,255,255,.82); border-radius:99px; color:#315366; font-size:.8rem; }
.section-label { color:#193b50; font:600 1.3rem 'Space Grotesk',sans-serif; margin:.2rem 0 .3rem; }
.section-hint { color:var(--muted); margin-bottom:1rem; }
.flow-card { min-height:94px; padding:15px 17px; border:1px solid #dceae6; border-radius:16px; background:rgba(255,255,255,.84); box-shadow:0 5px 18px rgba(27,75,67,.04); }
.flow-number { color:#168c77; font:700 .72rem 'Space Grotesk',sans-serif; letter-spacing:.12em; }
.flow-title { color:#1b3b50; font:600 1rem 'Space Grotesk',sans-serif; margin:5px 0 3px; }
.flow-copy { color:#657d89; font-size:.82rem; }
.signal-note { border:1px solid #d9eae4; background:linear-gradient(145deg,#fff,#f4faf8); border-radius:15px; padding:14px 16px; margin-bottom:10px; color:#385666; font-size:.86rem; }
.signal-note strong { color:#183b4f; }
div[data-testid="stFileUploader"] { border:1px dashed #75bbaa; border-radius:18px; background:rgba(255,255,255,.84); padding:12px; }
div[data-testid="stFileUploader"] section { background:transparent; }
div[data-testid="stFileUploader"] button { background:#168c77 !important; color:#fff !important; border:0 !important; font-weight:700 !important; }
div[data-testid="stFileUploader"] button:hover { background:#117460 !important; color:#fff !important; }
header[data-testid="stHeader"] { background:rgba(245,248,247,.92) !important; }
[data-testid="stToolbar"] { color:#183044 !important; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:14px 17px; border-radius:15px; box-shadow:0 4px 16px rgba(28,70,70,.04); }
div[data-testid="stMetricLabel"] { color:#607a88; }
div[data-testid="stProgress"] > div { background:#e2eeeb; border-radius:99px; }
div[data-testid="stProgress"] div[role="progressbar"] { background:linear-gradient(90deg,#58c7a8,#6e9ce0); border-radius:99px; }
div[data-testid="stAlert"] { border-radius:14px; }
.result-card { background:linear-gradient(145deg,#fff,#f0faf7); border:1px solid #d9eae4; border-radius:18px; padding:18px 20px; box-shadow:0 8px 24px rgba(37,96,80,.06); }
div[data-testid="stImage"] img { border-radius:14px; }
hr { border-color:#dce9e5; }
.footer { color:#718692; font-size:.78rem; padding-top:12px; }
[data-testid="stTable"] { background:#fff; border-radius:12px; overflow:hidden; }
</style>
<div class="hero">
  <div class="eyebrow">HYDRA · IMAGE FORENSICS</div>
  <div class="hero-title">Is this image real?</div>
  <p class="hero-copy">An explainable AI-image screening prototype combining visual patterns, frequency signals and file metadata in one review.</p>
  <div class="chips"><span class="chip">◉ Pixel analysis · CNN</span><span class="chip">⌁ Frequency analysis · FFT</span><span class="chip">⌕ Metadata · EXIF / possible C2PA/JUMBF marker</span></div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load():
    m = HydraNet(pretrained=False)
    m.load_state_dict(torch.load("hydra.pt", map_location="cpu"))
    return m.eval()

try:
    model = load()
except FileNotFoundError:
    st.error("Model weights (hydra.pt) were not found. Keep the file in the same folder as app.py.")
    st.stop()

st.markdown('<div class="section-label">Your image scan</div><div class="section-hint">A quick, explainable review in three steps.</div>', unsafe_allow_html=True)
flow = st.columns(3, gap="medium")
steps = [("01", "Upload", "Choose a JPG, PNG or WEBP image."), ("02", "Analyze", "Two image-model streams plus file metadata review the image."), ("03", "Explore", "Review the prediction and visual evidence.")]
for col, (number, title, copy) in zip(flow, steps):
    with col:
        st.markdown(f'<div class="flow-card"><div class="flow-number">STEP {number}</div><div class="flow-title">{title}</div><div class="flow-copy">{copy}</div></div>', unsafe_allow_html=True)
st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
with st.expander("Model blueprint · how Hydra reaches a result"):
    a, b, c = st.columns(3, gap="medium")
    a.markdown("**01 · Spatial vision**

ResNet-50 learns visual patterns from the image.")
    b.markdown("**02 · Frequency vision**

FFT converts image texture into a frequency map for a compact CNN.")
    c.markdown("**03 · Evidence & explanation**

Grad-CAM highlights influential regions; file metadata adds context.")
    st.markdown("Pixel and frequency features are fused for the model score. Metadata is supporting evidence, not a verdict.")
    st.caption("Small demo validation snapshot · 16 held-out images")
    m1, m2, m3 = st.columns(3)
    m1.metric("Clean accuracy", "68.8%")
    m2.metric("Clean AUC", "0.89")
    m3.metric("Validation size", "16 images")
    st.caption("These early results are from a tiny sample and should not be read as real-world performance.")
up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

if up:
    try:
        raw = up.getvalue()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        ev = file_evidence(io.BytesIO(raw))
        x = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(MEAN, STD)])(img).unsqueeze(0)
        with st.status("Hydra scan in progress", expanded=True) as scan:
            scan.write("Preparing image pixels and file evidence")
            cam, p_fake = gradcam(model, x)
            scan.write("Analyzing spatial and frequency patterns")
            ps, pf = model.head_probs(x)
            scan.update(label="Hydra scan complete", state="complete", expanded=False)
        side, level, conflict = verdict_label(p_fake, ev["hint"])
        confidence = max(p_fake, 1 - p_fake)
        base = np.array(img.resize((IMG, IMG))) / 255.0
        overlay = 0.55 * base + 0.45 * cm.jet(cam)[..., :3]
        file_size_kb = len(raw) / 1024
        image_format = Image.open(io.BytesIO(raw)).format or "Unknown"
        width, height = img.size

        st.markdown("---")
        st.markdown('<div class="section-label">Analysis report</div><div class="section-hint">Review the prediction, visual explanation and file signals together.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1.05], gap="large")
        c1.image(img, caption="Uploaded image", use_container_width=True)
        c2.image(overlay, caption="Grad-CAM · regions influencing the model", use_container_width=True)
        with c3:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.caption("HYDRA MODEL SCREENING")
            st.subheader(f"{side} · {level}")
            real_col, fake_col = st.columns(2, gap="small")
            real_col.metric("Real score", f"{(1 - p_fake)*100:.1f}%")
            fake_col.metric("AI / Fake score", f"{p_fake*100:.1f}%")
            st.progress(float(p_fake), text="AI / Fake score scale")
            st.caption("Real and AI/Fake scores are complementary and add up to 100%. They are uncalibrated model scores, not proof of authenticity.")
            st.markdown('</div>', unsafe_allow_html=True)
            if level == "Uncertain":
                st.warning("Low confidence: treat this image as unverified.")
            if conflict:
                st.warning("File metadata conflicts with the model prediction.")

        st.markdown("#### Image & file details")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Dimensions", f"{width} × {height}")
        d2.metric("Format", image_format)
        d3.metric("File size", f"{file_size_kb:.1f} KB")
        d4.metric("Evidence views", "3")

        st.markdown("#### Evidence breakdown")
        s1, s2, s3 = st.columns(3, gap="medium")
        with s1:
            st.markdown('<div class="signal-note"><strong>01 · Pixel patterns</strong><br>Visual features learned by the CNN.</div>', unsafe_allow_html=True)
            px_real, px_fake = st.columns(2, gap="small")
            px_real.metric("Real score", f"{(1-ps.item())*100:.1f}%", help="Diagnostic view from the shared fusion layer with frequency features masked. Not a standalone classifier.")
            px_fake.metric("AI / Fake score", f"{ps.item()*100:.1f}%", help="Diagnostic view from the shared fusion layer with frequency features masked. Not a standalone classifier.")
            st.progress(float(ps.item()), text="AI / Fake diagnostic score")
        with s2:
            st.markdown('<div class="signal-note"><strong>02 · Frequency patterns</strong><br>Image texture analyzed with FFT.</div>', unsafe_allow_html=True)
            fq_real, fq_fake = st.columns(2, gap="small")
            fq_real.metric("Real score", f"{(1-pf.item())*100:.1f}%", help="Diagnostic view from the shared fusion layer with pixel features masked. Not a standalone classifier.")
            fq_fake.metric("AI / Fake score", f"{pf.item()*100:.1f}%", help="Diagnostic view from the shared fusion layer with pixel features masked. Not a standalone classifier.")
            st.progress(float(pf.item()), text="AI / Fake diagnostic score")
        with s3:
            st.markdown('<div class="signal-note"><strong>03 · File metadata</strong><br>Rule-based evidence, not a probability score.</div>', unsafe_allow_html=True)
            for note in ev["notes"]:
                st.caption("• " + note)
        st.caption("Pixel and frequency values are diagnostic ablations from the shared fusion model, not separately trained classifiers. Missing metadata or a missing marker does not mean an image is fake.")
        report = (
            "HYDRA — AI IMAGE SCREENING REPORT
"
            f"File: {up.name}
Format: {image_format}
Dimensions: {width} x {height}
Size: {file_size_kb:.1f} KB

"
            f"Screening result: {side} · {level}
Real score: {(1-p_fake)*100:.1f}%
AI/Fake score: {p_fake*100:.1f}%
Model score confidence (not calibrated): {confidence*100:.1f}%

"
            f"Pixel diagnostic scores: real {(1-ps.item())*100:.1f}% / AI-fake {ps.item()*100:.1f}%
Frequency diagnostic scores: real {(1-pf.item())*100:.1f}% / AI-fake {pf.item()*100:.1f}%
"
            f"File signals: {'; '.join(ev['notes']) or 'No readable metadata signals'}

"
            "Prototype limitation: trained on a small demo sample. This report is not proof of authenticity. "
            "Metadata can be removed or forged; possible C2PA/JUMBF marker detection is not cryptographic verification.
"
        )
        st.download_button("Download analysis summary", report, file_name="hydra_analysis_report.txt", mime="text/plain")
        st.info("Prototype limitation: this model was trained on a small demo sample. Treat scores as a project demonstration, not proof of authenticity. EXIF metadata can be removed or forged; possible C2PA/JUMBF marker detection here is not cryptographic verification.")
    except Exception as e:
        st.error(f"Could not analyze this image: {e}")
else:
    st.markdown("<div class='footer'>Upload a clear JPG or PNG to view the model prediction, Grad-CAM heatmap, image details and signal breakdown.</div>", unsafe_allow_html=True)

st.markdown('<div class="footer">HYDRA · Explainable AI-image screening prototype</div>', unsafe_allow_html=True)
