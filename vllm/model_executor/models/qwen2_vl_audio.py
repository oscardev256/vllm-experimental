from typing import Optional, Union, Iterable, Tuple
import torch
import torch.nn as nn
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen2_vl import Qwen2VLForConditionalGeneration
from vllm.model_executor.models.utils import (
    MultiModalEmbeddings, merge_multimodal_embeddings, maybe_prefix
)
from vllm.multimodal import MULTIMODAL_REGISTRY, SupportsMultiModal, MultiModalData
from vllm.model_executor.weight_utils import WeightsMapper, AutoWeightsLoader
from vllm.config import VllmConfig
from transformers import WhisperModel


@MULTIMODAL_REGISTRY.register_model
class Qwen2VLAudioForConditionalGeneration(Qwen2VLForConditionalGeneration):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Load Whisper encoder only (no decoder)
        whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
        self.audio_encoder = whisper_model.encoder
        self.audio_proj = nn.Linear(
            whisper_model.config.d_model,
            self.config.hidden_size,
            bias=False,
        )

    def _parse_audio_tensor(self, multimodal_data: MultiModalData) -> Optional[torch.Tensor]:
        """Extracts and validates audio tensor from MultiModalData dict."""
        audio_mels = multimodal_data.get("audio", None)
        if audio_mels is None:
            return None
        if not isinstance(audio_mels, torch.Tensor):
            raise ValueError("audio must be a torch.Tensor")
        return audio_mels

    def get_multimodal_embeddings(self, multimodal_data: MultiModalData) -> MultiModalEmbeddings:
        """Returns multimodal embeddings, including vision and audio."""
        embeddings = super().get_multimodal_embeddings(multimodal_data)
        audio_mels = self._parse_audio_tensor(multimodal_data)
        if audio_mels is not None:
            audio_features = self.audio_encoder(audio_mels)[0]  # (B, T', D)
            audio_proj = self.audio_proj(audio_features)        # (B, T', H)
            audio_proj = audio_proj.view(-1, audio_proj.shape[-1])  # (B*T', H)
            embeddings += (audio_proj,)
        return embeddings

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None):
        return super().get_input_embeddings(
            input_ids, multimodal_embeddings=multimodal_embeddings
        )
