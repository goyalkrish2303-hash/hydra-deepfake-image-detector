"""Lightweight file metadata signals. These support the model; they never prove authenticity."""
from PIL import Image

AI_WORDS = ["stable diffusion", "midjourney", "dall-e", "dalle", "firefly", "comfyui", "automatic1111", "novelai", "leonardo", "imagen", "sdxl", "flux"]


def file_evidence(path_or_file):
    notes = []
    hint = "neutral"
    raw = path_or_file.read() if hasattr(path_or_file, "read") else open(path_or_file, "rb").read()
    if hasattr(path_or_file, "seek"):
        path_or_file.seek(0)
    raw_lower = raw.lower()
    c2pa_found = b"c2pa" in raw_lower or b"jumbf" in raw_lower
    notes.append("Possible C2PA/JUMBF marker found (not cryptographically verified)" if c2pa_found else "No C2PA/JUMBF marker detected")
    try:
        img = Image.open(path_or_file)
        exif = img.getexif()
        make, camera_model, software = exif.get(271), exif.get(272), exif.get(305)
        camera = " ".join(str(v) for v in (make, camera_model) if v)
        if camera:
            notes.append(f"Camera EXIF found ({camera})")
            hint = "camera"
        else:
            notes.append("No camera make/model in EXIF")
        metadata_values = []
        for value in img.info.values():
            if isinstance(value, bytes):
                metadata_values.append(value.decode("utf-8", errors="ignore"))
            elif isinstance(value, str):
                metadata_values.append(value)
        metadata_values.extend(str(v) for v in exif.values())
        if software:
            metadata_values.append(str(software))
        metadata_text = " ".join(metadata_values).lower()
        hits = sorted({word for word in AI_WORDS if word in metadata_text})
        if hits:
            notes.append("Known AI-tool metadata found (" + ", ".join(hits) + ")")
            hint = "ai"
        else:
            notes.append("No known AI-tool name found in readable metadata")
        if not exif and not any(isinstance(v, (str, bytes)) for v in img.info.values()) and not c2pa_found:
            notes.append("No readable provenance metadata; this is inconclusive")
    except Exception as e:
        notes.append(f"Image metadata could not be read ({e})")
    return {"hint": hint, "notes": notes}


def verdict_label(p_fake, hint="neutral"):
    """Map the model score to plain language. Conflicting metadata lowers a strong signal."""
    conf = max(p_fake, 1 - p_fake)
    side = "Fake" if p_fake >= 0.5 else "Real"
    level = "Strong model signal" if conf >= 0.90 else "Moderate model signal" if conf >= 0.70 else "Uncertain"
    conflict = (side == "Fake" and hint == "camera") or (side == "Real" and hint == "ai")
    if conflict and level == "Strong model signal":
        level = "Moderate model signal"
    return side, level, conflict
