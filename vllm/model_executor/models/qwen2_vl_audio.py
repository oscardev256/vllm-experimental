from collections.abc import Iterable, Mapping, Sequence

#from typing import Optional, Union, Iterable, Tuple
from typing import Any, Callable, Literal, Optional, TypedDict, Union, List, Tuple
import torch
import torch.nn as nn
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen2_vl import Qwen2VLForConditionalGeneration
from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.model_executor.models.utils import WeightsMapper, AutoWeightsLoader
from vllm.config import VllmConfig
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from vllm.sequence import IntermediateTensors
#from vllm.utils import LazyLoader, is_list_of
from vllm.utils import is_list_of

from qwen_vl_utils import process_vision_info

import librosa
import numpy as np
import whisper

#from .qwen2_vl import (Qwen2VLImagePixelInputs, Qwen2VLVideoPixelInputs, MultiModalDataItems, PromptUpdate, PromptReplacement,
#                       Qwen2VLMultiModalProcessor, Qwen2VLProcessingInfo, Qwen2VLDummyInputsBuilder)
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

    #print(f"+++++++++++hf_inputs keys: {hf_inputs.keys()}++++++++++++++++++++++")
    #print(f"+++++++++++hf_inputs values: {hf_inputs.values()}")
    #audio_length = hf_inputs.get("audio", torch.empty((0, 3)))
    if "audio_mels" in hf_inputs.keys():
        #audio_sizes = torch.tensor(hf_inputs["audio"].shape)
        #audio_sizes = audio_length.prod(-1)
        #print(f'hf_inputs["audio"].shape: {hf_inputs["audio_mels"].shape}')
        #audio_sizes = torch.tensor([hf_inputs["audio_mels"].shape[0], hf_inputs["audio_mels"].shape[1]])
        audio_length = torch.tensor([hf_inputs["audio_mels"].shape[1]]).unsqueeze(0)
        audio_sizes = audio_length.prod(-1)
    else:
        audio_length = torch.empty((0, 1))
        audio_sizes = audio_length.prod(-1)
        #audio_sizes = torch.tensor([])
    print(f"===============================================image_grid_thw shape: {image_grid_thw.shape}")
    print(f"===============================================image_grid_thw: {image_grid_thw}")
    print(f"===============================================image_grid_sizes shape: {image_grid_sizes.shape}")
    print(f"===============================================image_grid_sizes: {image_grid_sizes}")
    print(f"===============================================video_grid_thw shape: {video_grid_thw.shape}")
    print(f"===============================================video_grid_thw: {video_grid_thw}")         
    print(f"===============================================video_grid_sizes shape: {video_grid_sizes.shape}")
    print(f"===============================================video_grid_sizes: {video_grid_sizes}")        
    print(f"===============================================audio_sizes shape: {audio_sizes.shape}")
    print(f"===============================================audio_sizes: {audio_sizes}")
    #print(f"audio_length: {type(audio_length)}, audio: {audio_length}")
    #audio_sizes = audio_length.prod(-1)

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

def _qwen2vl_audio_field_config(hf_inputs):
    base = _qwen2vl_field_config(hf_inputs)
    # here you need a tensor of shape (N_patches, hidden_size) per audio clip
    # For simplicity you can treat audio as one flat modality, e.g.
    #audio_sizes = torch.tensor([ hf_inputs["audio_embeds"].size(0) ])
    #audio = hf_inputs.get("audio", torch.empty((0, 1)))
    #audio_sizes = torch.tensor([ hf_inputs["audio"].size(0) ])
    #print(f"audio type = {type(audio)}")
    #base["audio"] = MultiModalFieldConfig.flat_from_sizes("audio", audio_sizes)
    #base["audio"] = audio
    #print(f'base["audio"] = {base["audio"]}')
    return base
    '''
class Qwen2VLAudioMultiModalDataParser(MultiModalDataParser):

    def _parse_image_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        print("Inside _parse_image_data", hasattr(self, "_parse_audio_data"))
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_qwen2vl_field_config,
            )

        return super()._parse_image_data(data)

    def _parse_video_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[VideoItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        print("Inside _parse_video_data", hasattr(self, "_parse_audio_data"))
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="video",
                required_fields={"video_embeds", "video_grid_thw"},
                fields_factory=_qwen2vl_field_config,
            )

        return super()._parse_video_data(data)
    '''
    '''
    def _parse_audio_data(
        self,
        data#: Union[dict[str, torch.Tensor], ModalityData[VideoItem]],
    ):# -> Optional[ModalityDataItems[Any, Any]]:
        print("Inside _parse_audio_data")
        return       
    '''

# ---------------------------------------------------------------------------
# util helpers (flat, as in source file)
# ---------------------------------------------------------------------------

def _ensure_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16kHz if needed, cast to float32."""
    wav = wav.astype(np.float32)
    #if sr != 16_000:
    wav = librosa.resample(wav, orig_sr=sr, target_sr=16_000)
    return wav


def _waveform_to_logmel(wav: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pad/trim to 30 s and build Whisper log-Mel spectrogram."""
    wav_t = torch.from_numpy(wav)
    wav_t = whisper.pad_or_trim(wav_t, length=30 * 16_000)
    mel = whisper.log_mel_spectrogram(wav_t)  # (80, frames)
    return mel.cpu().numpy(), mel.shape[-1]


def compute_audio_pad_count(n_mel_frames: int, cnn_total_stride: int = 2) -> int:
    """Return the number of audio tokens after CNN down‑sampling."""
    return np.ceil(n_mel_frames / cnn_total_stride) 


class Qwen2VLAudioMultiModalDataParser(MultiModalDataParser):

    #def __init__(self):
    #    super().__init__(target_sr=16000)

    def __init__(
        self,
        *,
        target_sr: Optional[float] = None,
        audio_resample_method: Literal["librosa", "scipy"] = "librosa",
        video_needs_metadata: bool = False,
    ) -> None:
        super().__init__(target_sr=16000)

    def _parse_image_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        print("Inside qwen2_vl_audio.py, _parse_image_data...")
        #print(f"data shape: {len(data)}")
        #if len(data) > 0:
        #    print(f"data type: {type(data[0])}")
        #a = torch.tensor([0])
        #b = a[100]
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_qwen2vl_field_config,
            )

        return super()._parse_image_data(data)

    def _parse_video_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[VideoItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        print("Inside qwen2_vl_audio.py, _parse_video_data....")       
        #a = torch.tensor([0])
        #b = a[100]     
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="video",
                required_fields={"video_embeds", "video_grid_thw"},
                fields_factory=_qwen2vl_field_config,
            )

        return super()._parse_video_data(data)

#class Qwen2VLAudioMultiModalProcessor(Qwen2VLMultiModalProcessor):
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
        print(f"Inside _call_hf_processor, mm_kwargs: {mm_kwargs}")
        print(f"Inside _call_hf_processor, mm_data: {mm_data}")
        print(f"Inside _call_hf_processor, tok_kwargs: {tok_kwargs}")
        #rint(f"Inside _call_hf_processor, prompt: {prompt}")
        mm_kwargs = self.info._get_image_processor_kwargs(**mm_kwargs)
        result = self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            dict(**mm_kwargs, **tok_kwargs),
        )
        print(f"Input to _call_hf_processor, mm_kwargs: {mm_kwargs}")
        print(f"Input to _call_hf_processor, mm_data: {mm_data}")
        print(f"Input to _call_hf_processor, tok_kwargs: {tok_kwargs}")
        print(f"Input to _call_hf_processor, prompt: {prompt}")
        if "images" in mm_data:
            #result["audio"] = torch.tensor([[0]])
            image_inputs = mm_data["images"]
            #print(f"~~~~~~~~~~~~~~~~ _call_hf_processor: Image inputs shape: {len(image_inputs)}")
            if len(image_inputs) > 1:
                ap[78]              
        if "audios" in mm_data:
            #ap[78]
            #result["audio"] = torch.tensor([[0]])
            audio_inputs = mm_data["audios"]
            mel_list: List[np.ndarray] = []
            token_counts: List[int] = []
            #print(f"~~~~~~~~~~~~~~~~ _call_hf_processor: Audio inputs shape: {len(audio_inputs)}")
            #for wav, sr in audio_inputs:
            for wav in audio_inputs:
                wav_np = wav.cpu().numpy() if isinstance(wav, torch.Tensor) else wav
                #wav_np = _ensure_16k(wav_np, 16000)
                mel, n_frames = _waveform_to_logmel(wav_np)
                mel_list.append(mel)
                #token_counts.append(compute_audio_pad_count(n_frames, self.cnn_total_stride))
                #token_counts.append(compute_audio_pad_count(n_frames, 4))
                #token_counts.append(compute_audio_pad_count(n_frames, 1))
            audio_mels = np.stack(mel_list, axis=0)  # (B, 80, T)
            audio_mels = torch.from_numpy(audio_mels)
            #print(f"~~~~~~~~~~~~~~~~ _call_hf_processor: audio_mels.shape: {audio_mels.shape}")
            if audio_mels.shape[1] == 80:                           # (B, 80, T) → (B, T, 80)
                audio_feats = audio_mels.transpose(1, 2).contiguous()
            else:
                audio_feats = audio_mels
            audio_feats = audio_feats[:,:12,:]
            #print(f"~~~~~~~~~~~~~~~~ _call_hf_processor: audio_feats.shape: {audio_feats.shape}")
            result["audio_mels"] = audio_feats
            #result["audio_mels"] = torch.from_numpy(audio_mels[0])          
            #result["audio_length"] = torch.tensor([12])#torch.tensor(token_counts)
            result["audio_length"] = torch.tensor([3000])#torch.tensor(token_counts)
        #print(f"Output from _call_hf_processor: {type(result)}, result: {result}")
        return result        
#        return self.info.ctx.call_hf_processor(
#            self.info.get_hf_processor(**mm_kwargs),
#            dict(text=prompt, **mm_data),
#            dict(**mm_kwargs, **tok_kwargs),
#        )

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
                #grid_thw = torch.tensor([1])#out_mm_kwargs[modality][item_idx]
                #grid_thw = out_mm_kwargs[modality][item_idx]
                grid_thw = out_mm_kwargs["audio_length"][item_idx]
                #num_tokens = int(grid_thw.prod()) //4
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
    '''
    def get_hf_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ) -> Qwen2VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen2VLProcessor,
            image_processor=self.get_image_processor(min_pixels=min_pixels,
                                                     max_pixels=max_pixels,
                                                     size=size,
                                                     use_fast=kwargs.get(
                                                         "use_fast", True)),
            **kwargs,
        )

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        cfg = super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)
        cfg.update(_qwen2vl_audio_field_config(hf_inputs))
        return cfg
    '''
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
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Inside _process_audio_input $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Inside _process_audio_input $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Inside _process_audio_input $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Inside _process_audio_input $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Inside _process_audio_input $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Inside _process_audio_input $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(f"*!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@ Inside get_input_embeddings in qwen2 vl audio {audio_input.shape}!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@ ")

        #audio_feats = emb.to(dtype=torch.bfloat16)
        audio_mels = audio_input.to(dtype=torch.bfloat16)

        if audio_mels.shape[-1] == 80:                           # (T, 80) → (80, T)
            audio_feats = audio_mels.transpose(-2, -1).contiguous()
            # Initialize zeros and copy original values
            #audio_feats_expanded = torch.zeros((80, 3000), dtype=audio_feats.dtype, device=audio_feats.device)
            #audio_feats_expanded[:, :12] = audio_feats

        else:
            audio_feats = audio_mels               

        #audio_feats_expanded = torch.zeros((80, 3000), dtype=audio_feats.dtype, device=audio_feats.device)
        #audio_feats_expanded[:, :12] = audio_feats
        #print(f"audio_feats.unsqueeze(0) shape: {(audio_feats.unsqueeze(0).shape)}")
        #print(f"audio_feats.unsqueeze(0) shape: {(audio_feats_expanded.unsqueeze(0).shape)}")
        whisper_out = self.audio_encoder(audio_feats.unsqueeze(0))
        #whisper_out = self.audio_encoder(audio_feats_expanded.unsqueeze(0))

        audio_hidden = (
            whisper_out.last_hidden_state
            if hasattr(whisper_out, "last_hidden_state")
            else whisper_out[0]
        )                                                       # (B, T', d_model)

        audio_embeds = self.audio_proj(audio_hidden)            # (B, T', hidden)
        audio_embeds = audio_embeds.reshape(-1, audio_embeds.size(-1))
        print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@audio_embeds shape: {audio_embeds.shape}")          
        #return audio_embed

        # Split concatenated embeddings for each image item.
        #merge_size = self.visual.spatial_merge_size
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


    def get_input_embeddings2(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        print(f"*!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@ Inside get_input_embeddings in qwen2 vl audio !@!@!@!@!@!@!@!@!@!@!@!@!@!@!@ ")
        for emb in multimodal_embeddings:
            print(f"*!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@ Inside get_input_embeddings in qwen2 vl audio {emb.shape}!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@ ")
            if emb.shape[-1] == 80:
                #audio_feats = emb.to(dtype=torch.bfloat16)

                audio_mels = emb.to(dtype=torch.bfloat16)

                if audio_mels.shape[-1] == 80:                           # (T, 80) → (80, T)
                    audio_feats = audio_mels.transpose(-2, -1).contiguous()
                    # Initialize zeros and copy original values
                    audio_feats_expanded = torch.zeros((80, 3000), dtype=audio_feats.dtype, device=audio_feats.device)
                    audio_feats_expanded[:, :3] = audio_feats

                else:
                    audio_feats = audio_mels               

                print(f"audio_feats.unsqueeze(0) shape: {(audio_feats.unsqueeze(0).shape)}")
                print(f"audio_feats.unsqueeze(0) shape: {(audio_feats_expanded.unsqueeze(0).shape)}")
                whisper_out = self.audio_encoder(audio_feats_expanded.unsqueeze(0))

                audio_hidden = (
                    whisper_out.last_hidden_state
                    if hasattr(whisper_out, "last_hidden_state")
                    else whisper_out[0]
                )                                                       # (B, T', d_model)

                audio_embeds = self.audio_proj(audio_hidden)            # (B, T', hidden)
                audio_embeds = audio_embeds.reshape(-1, audio_embeds.size(-1))
                print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@audio_embeds shape: {audio_embeds.shape}")  
            print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2 emb.shape shape: {emb.shape}")            
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id, self.audio_token_id])
        return inputs_embeds

    def _parse_and_validate_audio_input2(
            self, **kwargs
        ) -> Optional[Qwen2VLAudioEmbeddingInputs]:
        audio_embeds = kwargs.pop("audio_embeds", None)
        if audio_embeds is None:
            return None
        audio_embeds = self._validate_and_reshape_mm_tensor(
            audio_embeds, "audio embeddings"
        )
        return Qwen2VLAudioEmbeddingInputs(
            audio_embeds=audio_embeds,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}
        print(f"********************keys in kwargs: {kwargs.keys()}******************************")
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
                #audio_input = kwargs.pop("audio_mels", None)
                modalities["audios"] = kwargs["audio_mels"][0][0]#torch.tensor([0])#self._parse_and_validate_audio_input(**kwargs)

        return modalities

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        #if "audio_mels" in kwargs:
        #    modalities["audios"] = kwargs["audio_mels"]        
        print(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Audio: {'audios' in modalities.keys()}, modalities.keys(): {modalities.keys()}")
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

    def _process_audio_input2(
            self, audio_input) -> tuple[torch.Tensor, ...]:

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"]
        else:
            pixel_values = image_input["pixel_values"]
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())
    '''
    def _parse_and_validate_audio_input(
            self, **kwargs: object):# -> Optional[Qwen2VLImageInputs]:
        audio_mels = kwargs.pop("audio_mels", None)
        #image_embeds = kwargs.pop("image_embeds", None)
        #image_grid_thw = kwargs.pop("image_grid_thw", None)

        #if pixel_values is None and image_embeds is None:
        if audio_mels is None:
            return None

        if audio_mels is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")
            return

            return Qwen2VLImagePixelInputs(type="pixel_values",
                                           pixel_values=pixel_values,
                                           image_grid_thw=image_grid_thw)
    '''
    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[Qwen2VLImagePixelInputs] = None,
        video_input: Optional[Qwen2VLVideoPixelInputs] = None,
        audio_input = None,
    ) -> torch.Tensor:

        print(f"WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWaudio_input is not None: {audio_input is not None}WWWWWWWWWWWWWWWWWWWWWWWWWWWW")
        inputs_embeds = super().get_input_embeddings_v0(
            input_ids,
            image_input=image_input,
            video_input=video_input)
        
        if audio_input is not None:
            #image_embeds = self._process_image_input(image_input)

            audio_feats = audio_input.to(dtype=torch.bfloat16)
            whisper_out = self.audio_encoder(audio_feats)

            audio_hidden = (
                whisper_out.last_hidden_state
                if hasattr(whisper_out, "last_hidden_state")
                else whisper_out[0]
            )                                                       # (B, T', d_model)

            audio_embeds = self.audio_proj(audio_hidden)            # (B, T', hidden)
            audio_embeds = audio_embeds.reshape(-1, audio_embeds.size(-1))
            print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@audio_embeds shape: {audio_embeds.shape}")
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                audio_embeds,
                placeholder_token_id=self.audio_token_id,
            )

        return inputs_embeds
        '''
        if audio_input is not None:
            audio_embeds = self._process_audio_input(audio_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                audio_embeds,
                #placeholder_token_id=self.config.audio_token_id,
                placeholder_token_id=self.audio_token_id,
            )
            
        return inputs_embeds
        '''
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config: Qwen2VLConfig = vllm_config.model_config.hf_config

        # Load Whisper encoder only (no decoder)
        whisper_cfg = WhisperConfig("openai/whisper-tiny")
        print(f"7878787878787878787878787878787878787 whisper_cfg: {whisper_cfg}")
        self.audio_encoder = WhisperEncoder(whisper_cfg)
        reduction_factor = self.audio_encoder.conv1.stride[0] * self.audio_encoder.conv2.stride[0]
        print(f"7878787878787878787878787878787878787 self.audio_encoder.conv1.stride[0]: {self.audio_encoder.conv1.stride[0]}")
        print(f"7878787878787878787878787878787878787 self.audio_encoder.conv1.stride: {self.audio_encoder.conv1.stride}")
        print(f"7878787878787878787878787878787878787 self.audio_encoder.conv2.stride[0]: {self.audio_encoder.conv2.stride[0]}")
        print(f"7878787878787878787878787878787878787 self.audio_encoder.conv2.stride: {self.audio_encoder.conv2.stride}")
        print(f"7878787878787878787878787878787878787 sreduction_factor: {reduction_factor}")
        #print(f"7878787878787878787878787878787878787 self.audio_encoder.cnn_total_stride: {self.audio_encoder.cnn_total_stride}")
        #print(f"7878787878787878787878787878787878787 reduction factor: {reduction_factor}")
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
        print(f"***************** Running forward function 1*************** inputs_embeds exist: {inputs_embeds is not None}")
        if intermediate_tensors is not None:
            inputs_embeds = None
            print("***************** Running forward function 2***************")

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)
            #audio_input = self._parse_and_validate_audio_input(**kwargs)
            print("***************** Running forward function 3***************")

            if image_input is None and video_input is None and audio_mels is None:
                inputs_embeds = None
                print("***************** Running forward function 4***************")
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")
                print("***************** Running forward function 5***************")
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input,
                    audio_input=kwargs["audio_mels"][0][0])
                input_ids = None
        print("***************** Running forward function 6***************")
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states


#@MULTIMODAL_REGISTRY.register_model
class Qwen2VLAudioForConditionalGeneration2(Qwen2VLForConditionalGeneration):
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

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"            

        raise ValueError("Only image, video, or audio modality is supported")

    def _parse_audio_tensor(self, **kwargs: object) -> Optional[torch.Tensor]:#(self, multimodal_data: MultiModalKwargs) -> Optional[torch.Tensor]:
        """Extracts and validates audio tensor from MultiModalKwargs dict."""
        audio_mels = kwargs.get("audio", None)
        if audio_mels is None:
            return None
        if not isinstance(audio_mels, torch.Tensor):
            raise ValueError("audio must be a torch.Tensor")
        return audio_mels

    #def get_multimodal_embeddings(self, multimodal_data: MultiModalKwargs) -> MultiModalEmbeddings:
    '''    
    def get_multimodal_embeddings(self,
                                **kwargs: object) -> MultiModalEmbeddings:
        """Returns multimodal embeddings, including vision and audio."""
        embeddings = self.get_multimodal_embeddings(**kwargs)
        audio_mels = self._parse_audio_tensor(**kwargs)
        if audio_mels is not None:
            audio_features = self.audio_encoder(audio_mels)[0]  # (B, T', D)
            audio_proj = self.audio_proj(audio_features)        # (B, T', H)
            audio_proj = audio_proj.view(-1, audio_proj.shape[-1])  # (B*T', H)
            embeddings += (audio_proj,)
        return embeddings
    '''

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        print(f"Audio: {'audios' in modalities.keys()}")
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

        return multimodal_embeddings

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None):
        return super().get_input_embeddings(
            input_ids, multimodal_embeddings=multimodal_embeddings
        )

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

            if image_input is None and video_input is None:
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
