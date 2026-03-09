from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureExtractorConfig:
    device: str = "cuda"
    cache_dir: str = "ckpt"
    radio_model: str = "radio_v2.5-b"
    radio_lang: str = "siglip"
    radio_input_size: int = 336
    radio_sigma: float = 7.0
