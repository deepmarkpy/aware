from pydantic import BaseModel
from typing import Optional, List

class EmbeddingRequest(BaseModel):
    audio: List[float]
    sampling_rate: int
    watermark: bytes

class DetectionRequest(BaseModel):
    audio: List[float]
    sampling_rate: int
    watermark_length: int = None
    expected_watermark: bytes = None

class EmbeddingResponse(BaseModel):
    success: bool
    watermarked_audio: List[float] = None
    error_message: str = None
    metadata: dict = None

class DetectionResponse(BaseModel):
    success: bool
    decision: bool = None
    extracted_watermark: list[int] = None
    confidence: float = None
    confidence_stats: dict = None
    error_message: str = None
    metadata: dict = None
