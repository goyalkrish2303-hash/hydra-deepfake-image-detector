# Hydra — Explainable AI Image Forensics Prototype

Team: Hydra  
Competition: Hack in Hills, powered by TinyFish

Hydra screens images for possible AI generation or manipulation using spatial image features, frequency analysis, Grad-CAM visualization, and rule-based file metadata evidence.

## Features

- ResNet-50 spatial image stream and a compact CNN over FFT frequency maps.
- Fused real-vs-AI/fake model score with Grad-CAM visualization.
- EXIF and known AI-tool metadata checks, plus possible C2PA/JUMBF marker detection. Marker detection is not cryptographic verification.
- Streamlit interface with image details and a downloadable report.

Scores are uncalibrated model outputs, not proof of authenticity. Pixel/frequency views are diagnostic ablations of the shared classifier, not independent classifiers. Metadata is supporting evidence only.

## Run locally

1. Install Python 3.10 or later.
2. Create and activate a virtual environment.
3. Install dependencies: pip install -r requirements.txt
4. Download the [Hydra model weights (hydra.pt)](https://github.com/goyalkrish2303-hash/hydra-deepfake-image-detector/releases/download/v0.1.0-hydra-demo/hydra.pt) into this project folder, beside app.py.
5. Run: streamlit run app.py
6. Open the local address shown by Streamlit and upload a JPG, PNG, or WEBP image.

## Training and evaluation

train.py expects ImageFolder-style folders: train/real, train/fake and valid/real, valid/fake (valid may instead be val or test). Example:

    python train.py --data path/to/real_vs_fake --epochs 3 --n_train 8000 --n_val 2000

Training needs dataset images and may download pretrained ResNet weights. Do not commit private credentials or datasets unless their licenses allow it.

## Demo metrics and limitations

results.json records results on only 16 held-out images: 68.75% clean accuracy and 0.890625 clean AUC. These small-sample results do not show real-world reliability. The model may perform poorly on non-face images and current generators. EXIF can be removed or forged; possible C2PA/JUMBF marker detection here does not verify signatures.

The model weights are attached to the [Hydra demo pre-release](https://github.com/goyalkrish2303-hash/hydra-deepfake-image-detector/releases/tag/v0.1.0-hydra-demo). TinyFish functionality is not implemented; naming the event is not a TinyFish integration.
