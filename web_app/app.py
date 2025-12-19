"""
SHARP Web Interface - Single Image to 3D Gaussian Splat
A beautiful web UI for Apple's ml-sharp model.
"""

import io
import logging
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request, send_file

from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.utils import io as sharp_io
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max upload

# Global model instance (loaded once)
MODEL = None
DEVICE = None
OUTPUT_DIR = Path(tempfile.gettempdir()) / "sharp_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_device():
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    """Load the SHARP model (downloads if needed)."""
    global MODEL, DEVICE
    
    if MODEL is not None:
        return MODEL
    
    DEVICE = get_device()
    LOGGER.info(f"Loading SHARP model on device: {DEVICE}")
    
    # Download or load checkpoint
    DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    
    MODEL = create_predictor(PredictorParams())
    MODEL.load_state_dict(state_dict)
    MODEL.eval()
    MODEL.to(DEVICE)
    
    LOGGER.info("Model loaded successfully!")
    return MODEL


@torch.no_grad()
def predict_image(image: np.ndarray, f_px: float) -> tuple[Gaussians3D, float, tuple[int, int]]:
    """Predict Gaussians from an image."""
    model = load_model()
    internal_shape = (1536, 1536)
    
    image_pt = torch.from_numpy(image.copy()).float().to(DEVICE).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(DEVICE)
    
    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )
    
    # Predict Gaussians in NDC space
    gaussians_ndc = model(image_resized_pt, disparity_factor)
    
    # Build intrinsics
    intrinsics = (
        torch.tensor([
            [f_px, 0, width / 2, 0],
            [0, f_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        .float()
        .to(DEVICE)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height
    
    # Convert Gaussians to metric space
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(DEVICE), intrinsics_resized, internal_shape
    )
    
    return gaussians, f_px, (height, width)


def load_image_from_upload(file_storage) -> tuple[np.ndarray, float]:
    """Load an image from Flask file upload."""
    # Save to temp file to use sharp's loader (handles EXIF, HEIC, etc.)
    with tempfile.NamedTemporaryFile(suffix=Path(file_storage.filename).suffix, delete=False) as tmp:
        file_storage.save(tmp.name)
        tmp_path = Path(tmp.name)
    
    try:
        image, _, f_px = sharp_io.load_rgb(tmp_path)
        return image, f_px
    finally:
        tmp_path.unlink()


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """Handle image upload and return 3D Gaussians."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    try:
        LOGGER.info(f"Processing image: {file.filename}")
        
        # Load image
        image, f_px = load_image_from_upload(file)
        LOGGER.info(f"Image loaded: {image.shape}, focal length: {f_px:.2f}px")
        
        # Run prediction
        gaussians, f_px, (height, width) = predict_image(image, f_px)
        
        # Save to PLY
        output_id = str(uuid.uuid4())[:8]
        output_filename = f"gaussians_{output_id}.ply"
        output_path = OUTPUT_DIR / output_filename
        
        save_ply(gaussians, f_px, (height, width), output_path)
        LOGGER.info(f"Saved output to: {output_path}")
        
        return jsonify({
            "success": True,
            "filename": output_filename,
            "download_url": f"/api/download/{output_filename}",
            "device": DEVICE,
            "image_size": {"width": width, "height": height},
            "focal_length_px": round(f_px, 2),
        })
        
    except Exception as e:
        LOGGER.exception("Error processing image")
        return jsonify({"error": str(e)}), 500


@app.route("/api/download/<filename>")
def download(filename):
    """Download a generated PLY file."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    
    return send_file(
        file_path,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/api/status")
def status():
    """Check API status and device info."""
    device = get_device()
    model_loaded = MODEL is not None
    
    return jsonify({
        "status": "ready",
        "device": device,
        "model_loaded": model_loaded,
        "supported_formats": sharp_io.get_supported_image_extensions(),
    })


if __name__ == "__main__":
    # Pre-load model on startup
    print("\n" + "=" * 60)
    print("  SHARP - Single Image to 3D Gaussian Splat")
    print("  by Apple Machine Learning Research")
    print("=" * 60 + "\n")
    
    print("Loading model (this may take a moment on first run)...")
    load_model()
    print(f"\nModel ready on {DEVICE}!")
    print("\nStarting web server...")
    print("Open http://localhost:8080 in your browser\n")
    
    app.run(host="0.0.0.0", port=8080, debug=False)

