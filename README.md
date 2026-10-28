# Hydra — Explainable AI Image Forensics Prototype

Team: Hydra  
Competition: Hack in Hills, powered by TinyFish

Hydra is a prototype for screening images for possible AI generation or manipulation. It combines spatial image features, frequency analysis, Grad-CAM visualization, and rule-based file metadata evidence.

## What it includes

- ResNet-50 spatial feature stream.
- FFT frequency stream with a compact CNN.
- Fused real-vs-AI/fake model score.
- Grad-CAM heatmap for model explanation.
- EXIF, AI-tool metadata, and possible C2PA/JUMBF marker checks. Marker detection is heuristic, not cryptographic verification.
- Streamlit interface with image details and downloadable report.

The real and AI/fake scores are uncalibrated model outputs, not proof of authenticity. Pixel/frequency views are diagnostic ablations of the shared fusion classifier, not separate trained classifiers. Metadata checks are supporting evidence only.

## Run locally

1. Install Python 3.10 or later.
2. In this folder, create and activate a virtual environment.
3. Install dependencies with: pip install -r requirements.txt
4. Place the trained checkpoint file hydra.pt in this folder.
5. Start the app with: streamlit run app.py
6. Open the local address printed by Streamlit and upload a JPG, PNG, or WEBP image.

The app requires hydra.pt beside app.py. The checkpoint is about 50 MB and is not uploaded to this repository yet. The source files are here; the model file must be added before the demo can run.

## Training and evaluation

train.py expects an ImageFolder-style dataset with train/real, train/fake, and valid/real, valid/fake folders (valid may instead be val or test). Example:

python train.py --data path/to/real_vs_fake --epochs 3 --n_train 8000 --n_val 2000

Training may download pretrained ResNet weights and needs dataset images. Do not commit private credentials or datasets unless their licenses allow it.

## Current demo metrics

results.json records results on a very small validation sample of 16 images: 68.75% clean accuracy and 0.890625 clean AUC. This is a prototype result and does not show real-world reliability. The model was developed around aligned face images and may perform poorly on other image types or current generators.

## Limitations

- Tiny training and validation samples; scores are not calibrated.
- Pixel and frequency values are shared-model diagnostic ablations.
- EXIF can be removed or forged; marker checks do not verify signatures.
- Broader data and independent testing are needed before real use.

## Repository files

- app.py — Streamlit application
- model.py — Hydra neural network
- gradcam.py and file_check.py — explanations and metadata evidence
- train.py — training/evaluation routine
- requirements.txt — Python dependencies
- results.json — saved prototype metrics
- .streamlit/config.toml — interface theme settings
- hydra.pt — required trained checkpoint; pending upload

TinyFish integration is not implemented in this prototype. Mentioning the event does not qualify as an integration for any separate prize.
