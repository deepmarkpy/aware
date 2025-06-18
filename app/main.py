from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
import numpy as np
import librosa
import soundfile as sf
import torch

from .models import WatermarkResponse, DetectionResponse
from .watermark.embedding import WatermarkEmbedder
from .watermark.detection import WatermarkDetector
from .utils.audio_utils import load_audio, save_audio

app = FastAPI(
    title="Swift-Mark Watermarking API",
    description="Fast and robust watermarking scheme with optimization-based embedding and neural detection",
    version="1.0.0"
)

# Initialize watermark components
embedder = WatermarkEmbedder()
detector = WatermarkDetector()

@app.get("/")
async def root():
    return {"message": "Swift-Mark Watermarking API", "version": "1.0.0"}

@app.post("/embed", response_model=WatermarkResponse)
async def embed_watermark(
    audio: UploadFile = File(...),
    watermark_strength: float = 1.0,
    secret_key: str = "default_key",
    sample_rate: int = 44100
):
    """Embed watermark into audio using optimization procedure"""
    try:
        # Load and validate audio
        audio_data = await audio.read()
        audio_array, sr = load_audio(io.BytesIO(audio_data), target_sr=sample_rate)
        
        # Embed watermark
        watermarked_array = embedder.embed(
            audio_array, 
            sample_rate=sr,
            secret_key=secret_key,
            strength=watermark_strength
        )
        
        # Save to bytes
        audio_byte_arr = io.BytesIO()
        save_audio(watermarked_array, sr, audio_byte_arr, format='wav')
        audio_byte_arr.seek(0)
        
        return StreamingResponse(
            io.BytesIO(audio_byte_arr.read()),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=watermarked.wav"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error embedding watermark: {str(e)}")

@app.post("/detect", response_model=DetectionResponse)
async def detect_watermark(
    audio: UploadFile = File(...),
    secret_key: str = "default_key",
    sample_rate: int = 44100
):
    """Detect watermark in audio using neural network"""
    try:
        # Load audio
        audio_data = await audio.read()
        audio_array, sr = load_audio(io.BytesIO(audio_data), target_sr=sample_rate)
        
        # Detect watermark
        detection_result = detector.detect(audio_array, sample_rate=sr, secret_key=secret_key)
        
        return DetectionResponse(
            watermark_detected=detection_result["detected"],
            confidence=detection_result["confidence"],
            secret_key_match=detection_result["key_match"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error detecting watermark: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "embedder": "ready", "detector": "ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 