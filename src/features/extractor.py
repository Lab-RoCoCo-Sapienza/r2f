"""
Dense patch-level feature extractor (RADIO + SigLIP backend).

Loads RADIO (AM-RADIO, NVlabs) with NACLIP applied to the last attention
block. The SigLIP adaptor projects spatial features to SigLIP space so
that cosine(patch_feat, siglip_text_emb) is a valid per-patch language
similarity score. Text is encoded with the SigLIP text encoder bundled
in the RADIO adaptor.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.features.configs import FeatureExtractorConfig
from src.models.naradio import RadioNaclipAttn, build_bias


class FeatureExtractor:
    """
    Dense patch-level feature extractor backed by RADIO + SigLIP.

    extract_dense() returns (ph, pw, D) float32 patches and
    encode_text() returns a (D,) float32 vector in SigLIP space.
    """

    def __init__(self, cfg: FeatureExtractorConfig) -> None:
        self._cfg = cfg
        self._device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self._radio = None
        self._lang_adaptor = None
        self._patch_hw: Tuple[int, int] = (21, 21)
        self._feature_dim: int = 1152

        self._load_radio(cfg)

    def _load_radio(self, cfg: FeatureExtractorConfig) -> None:
        import torch.hub
        prev_hub_dir = torch.hub.get_dir()
        if cfg.cache_dir:
            torch.hub.set_dir(cfg.cache_dir)
        try:
            self._radio = torch.hub.load(
                "NVlabs/RADIO", "radio_model",
                version=cfg.radio_model,
                progress=True,
                skip_validation=True,
                adaptor_names=[cfg.radio_lang],
            ).eval().to(self._device)
        finally:
            torch.hub.set_dir(prev_hub_dir)

        self._radio_prep = self._radio.make_preprocessor_external().to(self._device)

        # Steal the language adaptor so RADIO doesn't auto-compute it
        self._lang_adaptor = self._radio.adaptors[cfg.radio_lang]
        self._radio.adaptors = None

        num_prefix = self._radio.num_summary_tokens
        patch_size = self._radio.patch_size
        s = cfg.radio_input_size // patch_size
        self._patch_hw = (s, s)

        # Apply NACLIP to the last ViT block's attention
        last_block = self._radio.model.blocks[-1]
        bias = build_bias(s, s, cfg.radio_sigma, num_prefix, self._device)
        last_block.attn = RadioNaclipAttn(last_block.attn, bias).to(self._device)
        print(f"  RADIO {cfg.radio_model} loaded, NACLIP applied (sigma={cfg.radio_sigma}, grid={s}x{s})")

        # Probe feature dim via a dummy MLP forward
        dummy = torch.zeros(1, 1, self._radio.model.embed_dim, device=self._device)
        with torch.no_grad():
            out = self._lang_adaptor.head_mlp(dummy)
        self._feature_dim = out.shape[-1]
        print(f"  SigLIP feature dim: {self._feature_dim}")

    @torch.no_grad()
    def extract_dense(
        self, rgb: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract dense patch-level features from an RGB image.

        Returns
        -------
        patches : (ph, pw, D) float32, L2-normalised
        None    : placeholder for API compatibility
        """
        s = self._cfg.radio_input_size
        ph, pw = self._patch_hw

        img = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # [1,3,H,W]
        img = F.interpolate(img, size=(s, s), mode="bilinear", align_corners=False).to(self._device)
        img = self._radio_prep(img)

        out = self._radio(img)
        features = out.features                       # [1, N, D_radio]
        features = self._lang_adaptor.head_mlp(features)  # [1, N, D_siglip]
        features = F.normalize(features, dim=-1)
        patches = features[0].float().cpu().numpy().reshape(ph, pw, -1)  # [ph, pw, D]
        return patches, None

    @torch.no_grad()
    def encode_text(self, prompt: str) -> np.ndarray:
        """
        Encode a text prompt into a normalised SigLIP embedding.

        Returns
        -------
        (D,) float32, L2-normalised
        """
        tokens = self._lang_adaptor.tokenizer([prompt]).to(self._device)
        feats = self._lang_adaptor.encode_text(tokens)  # [1, D]
        feats = F.normalize(feats, dim=-1)
        return feats[0].float().cpu().numpy()

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def patch_hw(self) -> Tuple[int, int]:
        return self._patch_hw

    @property
    def device(self) -> torch.device:
        return self._device
