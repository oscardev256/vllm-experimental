from typing import Tuple
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

#from qwen_vl_utils import process_vision_info

import numpy as np
import whisper

from .qwen2_vl import *


class Qwen2VLAudioEmbeddingInputs(TypedDict):
    #type: Literal["audio_embeds"]
    audio_embeds: torch.Tensor
    """Supported types:
    - list[`torch.Tensor`]: A list of tensors holding all images' features.
        Each tensor holds an image's features.
    - `torch.Tensor`: A tensor holding all images' features
        (concatenation of all images' feature tensors).
    
    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the images.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    #image_grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format."""

def _qwen2vl_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
    image_grid_sizes = image_grid_thw.prod(-1)

    video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
    video_grid_sizes = video_grid_thw.prod(-1)

    #audio_length = hf_inputs.get("audio_length", torch.empty((0, 1)))
    #audio_sizes = audio_length.prod(-1)

    audio_sizes = hf_inputs.get("audio_length", torch.tensor([]))

    return dict(
        pixel_values=MultiModalFieldConfig.flat_from_sizes(
            "image", image_grid_sizes),
        image_embeds=MultiModalFieldConfig.flat_from_sizes(
            "image", image_grid_sizes),
        image_grid_thw=MultiModalFieldConfig.batched("image"),
        pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes),
        video_embeds=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes),
        video_grid_thw=MultiModalFieldConfig.batched("video"),
        audio_mels=MultiModalFieldConfig.flat_from_sizes(
            "audio", audio_sizes),
        audio_length=MultiModalFieldConfig.batched("audio"),        
    )

class Qwen2VLAudioMultiModalDataParser(MultiModalDataParser):

    def __init__(
        self,
        *,
        target_sr: Optional[float] = None,
        audio_resample_method: Literal["librosa", "scipy"] = "librosa",
        video_needs_metadata: bool = False,
    ) -> None:
        super().__init__(target_sr=16000)

def _waveform_to_logmel(wav: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pad/trim to 30 s and build Whisper log-Mel spectrogram."""
    wav_t = torch.from_numpy(wav)
    wav_t = whisper.pad_or_trim(wav_t, length=30 * 16_000)
    mel = whisper.log_mel_spectrogram(wav_t)  # (80, frames)
    return mel.cpu().numpy(), mel.shape[-1]

class Qwen2VLAudioMultiModalProcessor(BaseMultiModalProcessor[Qwen2VLProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return Qwen2VLAudioMultiModalDataParser()

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_kwargs = self.info._get_image_processor_kwargs(**mm_kwargs)
        result = self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            dict(**mm_kwargs, **tok_kwargs),
        )     
        if "audios" in mm_data:
            audio_inputs = mm_data["audios"]
            mel_list: List[np.ndarray] = []
            token_counts: List[int] = []
            for wav_np in audio_inputs:
                #wav_np = wav.cpu().numpy() if isinstance(wav, torch.Tensor) else wav
                mel, n_frames = _waveform_to_logmel(wav_np)
                mel_list.append(mel)
            audio_mels = np.stack(mel_list, axis=0)
            audio_mels = torch.from_numpy(audio_mels)
            #if audio_mels.shape[1] == 80:                           # (B, 80, T) → (B, T, 80)
            #    audio_feats = audio_mels.transpose(1, 2).contiguous()
            #else:
            #    audio_feats = audio_mels
            #audio_feats = audio_feats[:,:12,:]
            audio_feats = audio_mels[:,:,:12]
            result["audio_mels"] = audio_feats
            result["audio_length"] = torch.tensor([12])
            #result["audio_length"] = torch.tensor([3000]) 
        return result        

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        print("Inside _get_prompt_updates")
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        placeholder = {
            "image": vocab[hf_processor.image_token],
            "video": vocab[hf_processor.video_token],
            "audio": vocab["<|audio_pad|>"],
        }

        merge_length = image_processor.merge_size**2

        def get_replacement_qwen2vl(item_idx: int, modality: str): # Expansion of multimedia padding tokens.
            print("Inside get_replacement_qwen2vl")
            if modality == "audio":
                grid_thw = out_mm_kwargs["audio_length"][item_idx]
                num_tokens = int(grid_thw.prod()) //2
            else:
                grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
                assert isinstance(grid_thw, torch.Tensor)
                num_tokens = int(grid_thw.prod()) // merge_length
            print(f"modality: {modality}, item_idx: {item_idx}, grid_thw: {grid_thw}, num_tokens: {num_tokens}")
            return [placeholder[modality]] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_qwen2vl,
                                    modality=modality),
            ) for modality in ("image", "video", "audio")
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        print("Convert from HF to vLLM, Inside _get_mm_fields_config")
        return _qwen2vl_field_config(hf_inputs)
        #return _qwen2vl_audio_field_config(hf_inputs)

class Qwen2VLAudioProcessingInfo(Qwen2VLProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "video": None, "audio": None}    


class Qwen2VLAudioDummyInputsBuilder(Qwen2VLDummyInputsBuilder):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)

        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.image_token
        video_token: str = hf_processor.video_token
        audio_token: str = "<|audio_pad|>"
        print(f"Inside Qwen2VLDummyInputsBuilder, dummy data: {image_token * num_images + video_token * num_videos + audio_token * num_audios}")
        return image_token * num_images + video_token * num_videos + audio_token * num_audios


@MULTIMODAL_REGISTRY.register_processor(Qwen2VLAudioMultiModalProcessor,
                                        info=Qwen2VLAudioProcessingInfo,
                                        dummy_inputs=Qwen2VLAudioDummyInputsBuilder)
class Qwen2VLAudioForConditionalGeneration(Qwen2VLForConditionalGeneration):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        print("Inside get_placeholder_str( get_placeholder_str( get_placeholder_str( get_placeholder_str( get_placeholder_str( ...........")
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"            

        raise ValueError("Only image, video, or audio modality is supported")

    def _process_audio_input(
            self, audio_input):

        #audio_mels = audio_input.to(dtype=torch.bfloat16)
        audio_feats = audio_input.to(dtype=torch.bfloat16)            

        audio_feats_expanded = torch.zeros((80, 3000), dtype=audio_feats.dtype, device=audio_feats.device)
        whisper_out = self.audio_encoder(audio_feats.unsqueeze(0))

        audio_hidden = (
            whisper_out.last_hidden_state
            if hasattr(whisper_out, "last_hidden_state")
            else whisper_out[0]
        )                                                       # (B, T', d_model)

        audio_embeds = self.audio_proj(audio_hidden)            # (B, T', hidden)
        audio_embeds = audio_embeds.reshape(-1, audio_embeds.size(-1))
        sizes = torch.tensor([1500])#grid_thw.prod(-1) // merge_size // merge_size

        return audio_embeds#.split(sizes.tolist())

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id, self.audio_token_id])
        return inputs_embeds
    
    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}
        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("pixel_values_videos",
                             "video_embeds") and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(
                    **kwargs)
            if input_key in "audio_mels" and "audios" not in modalities:
                modalities["audios"] = kwargs["audio_mels"][0][0]#torch.tensor([0])#self._parse_and_validate_audio_input(**kwargs)

        return modalities
    
    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += vision_embeddings
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_video_input(video_input)
                multimodal_embeddings += video_embeddings
            if modality == "audios":
                audio_input = modalities["audios"]
                audio_embeddings = self._process_audio_input(audio_input)
                multimodal_embeddings += (audio_embeddings,)

        return multimodal_embeddings

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config: Qwen2VLConfig = vllm_config.model_config.hf_config

        # Load Whisper encoder only (no decoder)
        whisper_cfg = WhisperConfig("openai/whisper-tiny")
        self.audio_encoder = WhisperEncoder(whisper_cfg)
        reduction_factor = self.audio_encoder.conv1.stride[0] * self.audio_encoder.conv2.stride[0]
        self.audio_proj   = nn.Linear(whisper_cfg.d_model, config.hidden_size, bias=False)

        # ------------------- special token ids -----------------
        # Make sure your tokenizer / config.json defines these IDs.
        #self.audio_token_id       = config.audio_token_id
        #self.audio_start_token_id = config.audio_start_token_id
        #self.audio_end_token_id   = config.audio_end_token_id

        self.audio_start_token_id: int = 151_657
        self.audio_token_id:       int = 151_658
        self.audio_end_token_id:   int = 151_659

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for Qwen2-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            pixel_values: Pixel values to be fed to a model.
                `None` if no images are passed.
            image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in LLM.
                `None` if no images are passed.
            pixel_values_videos: Pixel values of videos to be fed to a model.
                `None` if no videos are passed.
            video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in LLM.
                `None` if no videos are passed.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None and audio_mels is None:
                inputs_embeds = None
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
                input_ids = None
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

