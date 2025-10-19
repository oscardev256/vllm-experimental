#!/usr/bin/env python3
"""
Isaac model registration with vLLM MULTIMODAL_REGISTRY.
Complete integration including processor, model interface, and weight loading skeleton.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence, Iterable
from typing import Any, Dict, Optional, Union

import math
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TVF

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import WeightsMapper, AutoWeightsLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3DecoderLayer
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

# Core processing types
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
)

# Dummy builder & processor inputs
from vllm.multimodal.profiling import (
    BaseDummyInputsBuilder,
    ProcessorInputs,
)

# Field config / kwargs
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargs,   # present in recent versions; not strictly required
)
from vllm.model_executor.models.interfaces import MultiModalEmbeddings

# Data helpers
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    MultiModalDataItems,
    ModalityDataItems,
    ImageItem,
)

from vllm.transformers_utils.processor import cached_processor_from_config

from transformers import PretrainedConfig, Qwen3Config
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from transformers.image_processing_utils_fast import BaseImageProcessorFast
from transformers.utils import TensorType, auto_docstring, logging

# typing for kwargs (4.57.0-friendly)
from typing_extensions import TypedDict, Unpack

from transformers.cache_utils import SlidingWindowCache, StaticCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Model
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import TensorType
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
import re

from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3Model

from transformers.models.siglip2.modeling_siglip2 import (
    Siglip2MLP,
)
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig

# Import MRotaryEmbedding for monkey patching
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

# ===== TensorStream Compatibility Layer for Isaac MRoPE =====
# Minimal implementation of TensorStream classes needed for Isaac's 3D positional encoding

import itertools
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Iterator

class ModalityType(Enum):
    """Base class for modality-type enumerations."""
    def __lt__(self, other):
        if isinstance(other, ModalityType):
            return self.value < other.value
        return NotImplemented
    
    def __hash__(self):
        return hash(self.value)

class VisionType(ModalityType):
    """Vision modality types."""
    image = 1

class TextType(ModalityType):
    """Text modality types."""
    text = 0

@dataclass
class Event:
    """Represents a single modality event with spatial/temporal dimensions."""
    modality_type: ModalityType
    _dims: Optional[List[int]] = None
    _dims_virtual: Optional[List[int]] = None  # Scaled-down dimensions for embeddings
    _dims_real: Optional[List[int]] = None     # Original dimensions for pixel shuffle
    idx_range: Tuple[int, int] = (0, 0)  # (start, end) indices for partial events
    device: Optional[torch.device] = None
    
    def dims(self, virtual: bool = True) -> Optional[List[int]]:
        """Return spatial dimensions for this event.
        
        Args:
            virtual: If True, return virtual (scaled-down) dimensions for embeddings.
                    If False, return real (original) dimensions for pixel shuffle.
        """
        if virtual:
            return self._dims_virtual if self._dims_virtual is not None else self._dims
        else:
            return self._dims_real if self._dims_real is not None else self._dims
    
    def __post_init__(self):
        if self.idx_range == (0, 0) and self._dims:
            # Set full range if not specified
            total_elements = 1
            for dim in self._dims:
                total_elements *= dim
            self.idx_range = (0, total_elements)

class Stream:
    """Container for Event objects representing one batch sample."""
    def __init__(self, events: List[Event]):
        self.events = events
    
    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)
    
    def __len__(self) -> int:
        return len(self.events)

class TensorStream:
    """Container for multiple Stream objects with batch processing support."""
    def __init__(self, streams: List[Stream], device: Optional[torch.device] = None):
        self.streams = streams
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return (batch_size, sequence_length) shape."""
        batch_size = len(self.streams)
        if batch_size == 0:
            return (0, 0)
        
        # Calculate total sequence length across all events in all streams
        total_seq_len = 0
        for stream in self.streams:
            for event in stream:
                start, end = event.idx_range
                total_seq_len += (end - start)
        
        # For multimodal models, sequence length is typically per stream
        # But for MRoPE we need the total across all streams
        max_seq_len = max(
            sum(end - start for event in stream for start, end in [event.idx_range])
            for stream in self.streams
        ) if self.streams else 0
        
        return (batch_size, max_seq_len)

def create_stream(events: List[Event]) -> Stream:
    """Create a Stream from a list of Events."""
    return Stream(events)

def group_streams(streams: List[Stream], device: Optional[torch.device] = None) -> TensorStream:
    """Group multiple Stream objects into a TensorStream."""
    return TensorStream(streams, device)

def compute_mrope_pos_tensor(ts: TensorStream, n_pos_dims: int = 3) -> torch.Tensor:
    """
    Create a (batch, T, n_pos_dims) position tensor for MRoPE.
    
    Args:
        ts: TensorStream containing modality events
        n_pos_dims: total coordinate dimensions (default 3 for time, height, width)
    
    Returns:
        torch.LongTensor - shape (batch_size, seq_len, n_pos_dims)
    """
    all_coords = []
    
    for stream in ts.streams:  # one Stream == one batch sample
        cumulative_offset = 0  # running time index for this stream
        
        for event in stream:
            # Build coordinate grid for this event
            dims = (event.dims() or [1]) + [1] * (n_pos_dims - len(event.dims() or []))
            
            # Create ranges for each dimension
            first_dim = range(cumulative_offset, cumulative_offset + dims[0])
            cumulative_offset += dims[0]  # advance time for the next event
            other_dims = [range(d) for d in dims[1:]]
            
            # Use itertools.product to create all coordinate combinations
            full_coords = list(itertools.product(first_dim, *other_dims))
            
            # Slice if the event is partial
            s, e = event.idx_range
            coords = full_coords[s:e]
            
            # Extend the flattened coordinate list
            all_coords.extend(coords)
    
    # Convert to tensor and reshape to (B, T, n_pos_dims)
    B, T = ts.shape
    if not all_coords:
        # Return empty tensor with correct shape
        return torch.zeros((B, T, n_pos_dims), dtype=torch.long, device=ts.device)
    
    return torch.tensor(all_coords, dtype=torch.long, device=ts.device).reshape(B, T, n_pos_dims)

def modality_mask(ts: TensorStream, modality_type: ModalityType) -> torch.Tensor:
    """Create boolean mask for specific modality type in the tensor stream."""
    B, T = ts.shape
    mask = torch.zeros((B, T), dtype=torch.bool, device=ts.device)
    
    for batch_idx, stream in enumerate(ts.streams):
        seq_idx = 0
        for event in stream:
            if event.modality_type == modality_type:
                start, end = event.idx_range
                mask[batch_idx, seq_idx:seq_idx+(end-start)] = True
            seq_idx += (event.idx_range[1] - event.idx_range[0])
    
    return mask

# ===== End TensorStream Compatibility Layer =====

class PixelShuffleSiglip2VisionConfig(Siglip2VisionConfig):
    """Vision configuration for Isaac with Pixel Shuffle support.

    Extends Siglip2VisionConfig with additional fields for pixel shuffle.
    """

    model_type = "pixel_shuffle_siglip2"
    base_config_key = "vision_config"

    def __init__(
        self,
        pixel_shuffle_scale_factor: int = 1,
        num_patches: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Add our custom fields
        self.pixel_shuffle_scale_factor = pixel_shuffle_scale_factor
        self.num_patches = num_patches


def create_cumulative_seq_lengths(seq_sizes: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int]:
    """Create cumulative sequence lengths for variable-length attention."""
    cu_seqlens = torch.zeros(len(seq_sizes) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = seq_sizes.cumsum(0)
    max_seqlen = int(seq_sizes.max().item()) if len(seq_sizes) > 0 else 0
    return cu_seqlens, max_seqlen


class Siglip2VariableSequenceEmbeddings(nn.Module):
    def __init__(self, config: PixelShuffleSiglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def positional_embeddings(
        self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        # Prepare positional embeddings grid: (1, embed_dim, h, w)
        positional_embeddings = (
            self.position_embedding.weight.reshape(self.position_embedding_size, self.position_embedding_size, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        _seq_patches, seq_sizes, spatial_shapes = packed_seq_patches
        pos_embeds_list = []
        mode = "bilinear"
        align_corners = False
        antialias = True
        
        # Follow Qwen2-VL pattern: iterate through each image's actual contribution
        # seq_sizes tells us how many patches each image contributes
        # spatial_shapes tells us the [H, W] dimensions for each image
        
        # Generate positional embeddings for each image based on its actual size
        for i, (seq_size, spatial_shape) in enumerate(zip(seq_sizes, spatial_shapes)):
            height, width = spatial_shape
            expected_patches = height * width
            
            # Ensure seq_size matches expected patches for this spatial shape
            if seq_size != expected_patches:
                print(f"WARNING: seq_size {seq_size} != expected patches {expected_patches} for image {i}")
            
            # Generate positional embeddings for this image's spatial dimensions
            if height > 0 and width > 0:
                resized_pos_embed = F.interpolate(
                    positional_embeddings,
                    size=(height, width),
                    mode=mode,
                    align_corners=align_corners,
                    antialias=antialias,
                )
                # Reshape from (1, embed_dim, height, width) to (height*width, embed_dim)
                resized_pos_embed = resized_pos_embed.reshape(self.embed_dim, height * width).transpose(0, 1)
                
                # Only take the number of patches this image actually contributes
                resized_pos_embed = resized_pos_embed[:seq_size]
            else:
                # Fallback - should never happen in practice
                resized_pos_embed = positional_embeddings.reshape(
                    self.embed_dim, self.position_embedding_size * self.position_embedding_size
                ).transpose(0, 1)[:seq_size]
            
            pos_embeds_list.append(resized_pos_embed)

        # Concatenate all positional embeddings (Qwen2-VL style)
        # Result: [total_actual_patches, embed_dim] matching actual sequence length
        pos_embeds = torch.cat(pos_embeds_list, dim=0)
        return pos_embeds

    def forward(self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        seq_patches, _seq_sizes, _spatial_shapes = packed_seq_patches

        # Debug: Check the actual sequence length vs positional embedding length
        print(f"DEBUG: seq_patches.shape = {seq_patches.shape}")
        print(f"DEBUG: spatial_shapes = {_spatial_shapes}")

        # Apply patch embeddings
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(seq_patches.to(dtype=target_dtype))
        
        # Flatten patch embeddings to match positional embeddings format
        # From [batch, patches_per_image, embed_dim] to [total_patches, embed_dim]
        batch_size, patches_per_image, embed_dim = patch_embeds.shape
        patch_embeds = patch_embeds.view(batch_size * patches_per_image, embed_dim)
        
        pos_embeds = self.positional_embeddings(packed_seq_patches)

        print(f"DEBUG: patch_embeds.shape = {patch_embeds.shape}")
        print(f"DEBUG: pos_embeds.shape = {pos_embeds.shape}")

        # Add positional embeddings to patch embeddings
        # Both should now have the same shape: [total_actual_patches, embed_dim]
        embeddings = patch_embeds + pos_embeds
        return embeddings


class Siglip2VariableLengthAttention(nn.Module):
    """Custom attention that supports variable-length sequences with flash attention."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, cu_seqlens=None, max_seqlen=None):
        batch_size, seq_len, _ = hidden_states.size()

        # For variable-length attention, we need to reshape to (total_tokens, embed_dim)
        if batch_size != 1:
            raise ValueError("Variable-length attention expects batch_size=1 for packed sequences")
        hidden_states = hidden_states.squeeze(0)  # Remove batch dimension: (seq_len, embed_dim)

        # Store original dtype
        orig_dtype = hidden_states.dtype

        # 1. Linear projections
        Q = self.q_proj(hidden_states)  # (seq_len, embed_dim)
        K = self.k_proj(hidden_states)  # (seq_len, embed_dim)
        V = self.v_proj(hidden_states)  # (seq_len, embed_dim)

        # 2. Reshape for multi-head attention: (seq_len, n_heads, head_dim)
        Q = Q.view(-1, self.num_heads, self.embed_dim // self.num_heads)
        K = K.view(-1, self.num_heads, self.embed_dim // self.num_heads)
        V = V.view(-1, self.num_heads, self.embed_dim // self.num_heads)

        # 3. Apply variable-length attention using flash attention
        attn_output, _, _, _, _ = torch.ops.aten._flash_attention_forward(
            query=Q,
            key=K,
            value=V,
            cum_seq_q=cu_seqlens,
            cum_seq_k=cu_seqlens,
            max_q=max_seqlen,
            max_k=max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            return_debug_mask=False,
            scale=self.scale,
            window_size_left=-1,
            window_size_right=-1,
            alibi_slopes=None,
        )

        # 4. Reshape attention output from (seq_len, n_heads, head_dim) to (seq_len, embed_dim)
        attn_output = attn_output.reshape(seq_len, self.embed_dim)

        # 5. Convert back to original dtype if needed
        if attn_output.dtype != orig_dtype:
            attn_output = attn_output.to(orig_dtype)

        # 6. Project output
        attn_output = self.out_proj(attn_output)  # (seq_len, embed_dim)

        # 7. Add back batch dimension for compatibility
        attn_output = attn_output.unsqueeze(0)  # (1, seq_len, embed_dim)

        return attn_output, None


class IsaacSiglip2EncoderLayer(nn.Module):
    """Siglip2 encoder layer with variable-length attention."""

    def __init__(self, config: PixelShuffleSiglip2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Siglip2VariableLengthAttention(config)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)  # Use HF's Siglip2MLP
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
    ) -> tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class IsaacEncoder(nn.Module):
    """Encoder using Isaac encoder layers with variable-length attention support."""

    def __init__(self, config: PixelShuffleSiglip2VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([IsaacSiglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        output_hidden_states: bool = False,
    ):
        all_hidden_states = () if output_hidden_states else None

        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
                max_seqlen,
            )

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states, None


def create_pixel_shuffle_index_map(
    seq_sizes: torch.Tensor,
    token_grids: torch.Tensor,
    scale_factor: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build a gather-index map that tells us, for every *output* token after
    pixel-shuffle, which `scale_factor**2` *input* tokens are being merged.

    Args
    ----
    seq_sizes     : (num_images,)  - #patches in each image (row-major order)
    token_grids   : (num_images,2) - (height, width) for every image
    scale_factor  : spatial down-scale factor (≥2)
    device        : (optional) overrides `seq_sizes.device`

    Returns
    -------
    gather_idx : (new_total_seq_len, scale_factor**2) int64 tensor.
                 gather_idx[i, j] is the *flat* index into the *original*
                 packed sequence for the j-th sub-patch that forms the
                 i-th output token.
    """
    if device is None:
        device = seq_sizes.device

    r = int(scale_factor)
    if r < 2:
        raise ValueError("`scale_factor` must be ≥ 2")

    # Safety: all spatial dims must be divisible by r
    # Cannot run under torch compile fullgraph mode hence
    if not torch.compiler.is_compiling():
        if not ((token_grids[:, 0] % r == 0).all() and (token_grids[:, 1] % r == 0).all()):
            raise AssertionError(
                f"Every (H,W) in `token_grids` must be divisible by scale_factor={r}, got {token_grids.tolist()}"
            )

    gather_chunks: list[torch.Tensor] = []
    tok_offset = 0

    for seq_len, (h, w) in zip(seq_sizes.tolist(), token_grids.tolist(), strict=False):
        # Build the (H, W) grid of flat indices for this image
        grid = torch.arange(seq_len, device=device, dtype=torch.int64) + tok_offset
        grid = grid.view(h, w)  # (H, W)

        # -------- identical ordering to your fixed-res routine --------
        # Step 1: split width into blocks of r
        grid = grid.view(h, w // r, r)  # (H, W/r, r)
        # Step 2: now split height into blocks of r
        grid = grid.view(h // r, r, w // r, r)  # (H/r, r, W/r, r)
        # Step 3: final permutation to (H/r, W/r, r, r)
        grid = grid.permute(0, 2, 1, 3).contiguous()  # (H/r, W/r, r, r)
        # Step 4: each (r, r) block forms one output token
        gather_chunks.append(grid.reshape(-1, r * r))  # (H*W / r², r²)

        tok_offset += seq_len

    # Concatenate over all images in the packed batch
    gather_idx = torch.cat(gather_chunks, dim=0)  # (Σ_i HᵢWᵢ/r², r²)
    return gather_idx


def pixel_shuffle_varlen(
    x: torch.Tensor,
    token_grids: torch.Tensor,
    scale_factor: int = 1,
) -> torch.Tensor:
    r"""Apply pixel shuffle to a packed vision sequence without unpacking per image.

    Args:
        x (`torch.Tensor`):
            Concatenated vision embeddings. Accepts `(seq_len, hidden_size)` or `(1, seq_len, hidden_size)` shapes
            produced by stacking image patches.
        token_grids (`torch.Tensor`):
            Integer tensor of shape `(num_images, 2)` whose rows give the `(height, width)` patch grid sizes
            corresponding to each image segment inside `x`.
        scale_factor (`int`, *optional*, defaults to 1):
            Spatial down-sampling factor specific to pixel shuffle. Values greater than one merge `scale_factor**2` neighboring patches into a
            single embedding channel-group.

    Returns:
        `torch.Tensor`: Pixel-shuffled embeddings with shape matching the input convention:
        `(seq_len, hidden_size * scale_factor**2)` when the input was 2D, or `(1, seq_len, hidden_size * scale_factor**2)`
        if the singleton batch dimension was present.

    Raises:
        ValueError: If more than one batch item is provided.
    """
    keep_batch_dim = x.dim() == 3
    if keep_batch_dim:
        if x.size(0) != 1:
            raise AssertionError("Packed sequence is expected to have batch_size == 1")
        x_ = x.squeeze(0)  # (seq, embed)
    else:
        x_ = x  # (seq, embed)

    embed_dim = x_.size(-1)
    r = int(scale_factor)

    # Calculate seq_sizes from token_grids
    seq_sizes = torch.prod(token_grids, dim=-1)

    # Build index map and gather in one go
    gather_idx = create_pixel_shuffle_index_map(
        seq_sizes=seq_sizes,
        token_grids=token_grids,
        scale_factor=r,
        device=x_.device,
    )  # (new_seq, r²)

    # Gather → (new_seq, r², embed_dim)
    gathered = x_[gather_idx]  # fancy indexing keeps gradient

    # Merge the r² group dimension into channels to finish the shuffle
    out = gathered.reshape(gathered.size(0), embed_dim * r * r)

    # Restore batch dimension if needed
    if keep_batch_dim:
        out = out.unsqueeze(0)
    return out


class Siglip2SequenceVisionTransformer(nn.Module):
    def __init__(self, config: PixelShuffleSiglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Siglip2VariableSequenceEmbeddings(config)
        self.encoder = IsaacEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pixel_shuffle_scale_factor = config.pixel_shuffle_scale_factor

    def forward(self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor]):
        seq_patches, token_grids = packed_seq_patches
        # token_grids is already in [H, W] format from get_multimodal_embeddings conversion
        seq_sizes = torch.prod(token_grids, dim=-1)
        spatial_shapes = token_grids  # Already [H, W] format
        print(f"DEBUG IsaacVisionModel.forward: seq_patches.shape={seq_patches.shape}, token_grids={token_grids}, seq_sizes={seq_sizes}")

        # Get embeddings from packed sequence
        hidden_states = self.embeddings((seq_patches, seq_sizes, spatial_shapes))

        # Add a pseudo batch dimension for the encoder
        hidden_states = hidden_states.unsqueeze(0)

        # Generate cumulative sequence lengths for variable-length attention
        cu_seqlens, max_seqlen = create_cumulative_seq_lengths(seq_sizes, hidden_states.device)

        # Pass through encoder with variable-length attention parameters
        hidden_states, _, _ = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # Apply final layer normalization
        hidden_states = self.post_layernorm(hidden_states)

        if self.pixel_shuffle_scale_factor > 1:
            hidden_states = pixel_shuffle_varlen(
                x=hidden_states,
                token_grids=token_grids,
                scale_factor=self.pixel_shuffle_scale_factor,
            )
        # Remove the pseudo batch dimension we added earlier
        hidden_states = hidden_states.squeeze(0)

        # Return the full sequence of embeddings
        return hidden_states



# ===== Isaac vision processing constants =====
MAX_PIXELS = 60_000_000  # 60 MP cap for safety
ISAAC_VISION_MEAN = (0.5, 0.5, 0.5)
ISAAC_VISION_STD = (0.5, 0.5, 0.5)
ISAAC_VISION_SCALE = 1.0 / 255.0

_ISAAC_MEAN_TENSOR = torch.tensor(ISAAC_VISION_MEAN, dtype=torch.float32).view(1, 1, 1, -1)
_ISAAC_STD_TENSOR = torch.tensor(ISAAC_VISION_STD, dtype=torch.float32).view(1, 1, 1, -1)


def extract_image_pil(image: PIL.Image.Image) -> torch.Tensor:
    """Extract image tensor from a PIL image (RGB)."""
    print(f"extract_image_pil: The type of image is {type(image)}")
    if image.width * image.height > MAX_PIXELS:
        raise ValueError(f"Image (w={image.width}, h={image.height}) exceeds MAX={MAX_PIXELS}")
    img = image if image.mode == "RGB" else image.convert("RGB")
    arr = np.asarray(img)
    if not arr.flags.writeable:
        try:
            arr.setflags(write=True)
        except ValueError:
            arr = arr.copy()
    return torch.from_numpy(arr)  # H, W, 3 (uint8)


def get_image_size_for_max_num_patches(
    image_height: int,
    image_width: int,
    patch_size: int,
    max_num_patches: int,
    min_num_patches: Optional[int] = None,
    eps: float = 1e-5,
    pixel_shuffle_scale: int = 1,
) -> tuple[int, int]:
    """Compute the target (H, W) so that the number of patches satisfies constraints."""

    def get_scaled_image_size(scale: float, original_size: int, patch: int, pshuffle: int) -> int:
        scaled = scale * original_size
        divisor = patch * pshuffle
        scaled = math.ceil(scaled / divisor) * divisor
        scaled = max(divisor, scaled)
        return int(scaled)

    divisor = patch_size * pixel_shuffle_scale

    adj_h = max(divisor, math.ceil(image_height / divisor) * divisor)
    adj_w = max(divisor, math.ceil(image_width / divisor) * divisor)
    num_patches = (adj_h / patch_size) * (adj_w / patch_size)

    # Need to scale up to meet min
    if min_num_patches is not None and num_patches < min_num_patches:
        lo, hi = 1.0, 100.0
        while (hi - lo) >= eps:
            mid = (lo + hi) / 2
            th = get_scaled_image_size(mid, image_height, patch_size, pixel_shuffle_scale)
            tw = get_scaled_image_size(mid, image_width, patch_size, pixel_shuffle_scale)
            npatches = (th / patch_size) * (tw / patch_size)
            if npatches >= min_num_patches:
                hi = mid
            else:
                lo = mid
        return (
            get_scaled_image_size(hi, image_height, patch_size, pixel_shuffle_scale),
            get_scaled_image_size(hi, image_width, patch_size, pixel_shuffle_scale),
        )

    # Already within budget
    if num_patches <= max_num_patches:
        return adj_h, adj_w

    # Need to scale down
    lo, hi = eps / 10, 1.0
    while (hi - lo) >= eps:
        mid = (lo + hi) / 2
        th = get_scaled_image_size(mid, image_height, patch_size, pixel_shuffle_scale)
        tw = get_scaled_image_size(mid, image_width, patch_size, pixel_shuffle_scale)
        npatches = (th / patch_size) * (tw / patch_size)
        if npatches <= max_num_patches:
            lo = mid
        else:
            hi = mid
    return (
        get_scaled_image_size(lo, image_height, patch_size, pixel_shuffle_scale),
        get_scaled_image_size(lo, image_width, patch_size, pixel_shuffle_scale),
    )


def prepare_image_tensor(image: torch.Tensor, scale: float = ISAAC_VISION_SCALE) -> torch.Tensor:
    """Rescale to [0,1], then normalize using Isaac mean/std."""
    if not torch.is_floating_point(image):
        image = image.float()
    rescaled = image * scale
    mean = _ISAAC_MEAN_TENSOR.to(image.device)
    std = _ISAAC_STD_TENSOR.to(image.device)
    return (rescaled - mean) / std


def patchify_vision(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert normalized HWC tensor(s) into flattened patches.
    Input:  [N, H, W, C]
    Output: [N, H/ps, W/ps, C*ps*ps]
    """
    n, h, w, c = image.shape
    if (h % patch_size) or (w % patch_size):
        raise ValueError(f"Image dims {image.shape} not divisible by patch_size={patch_size}")
    patches = image.reshape(n, h // patch_size, patch_size, w // patch_size, patch_size, c)
    patches = patches.permute(0, 1, 3, 2, 4, 5)  # [N, H//ps, W//ps, ps, ps, C]
    patches = patches.reshape(n, h // patch_size, w // patch_size, c * patch_size * patch_size)
    return patches


def process_vision_for_patches(
    images: torch.Tensor,
    patch_size: int,
    max_num_patches: int,
    min_num_patches: Optional[int] = None,
    pixel_shuffle_scale: int = 1,
) -> tuple[torch.Tensor, list[int]]:
    """
    Isaac vision preproc: resize -> normalize -> patchify.
    images: HWC or NHWC (uint8 or float)
    returns:
      patches: [N, H//ps, W//ps, C*ps*ps]
      dims_virtual: [T, H, W] (T fixed to 1 for images)
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)  # [1, H, W, C]

    # to NCHW for resize
    images = images.permute(0, 3, 1, 2)

    _, _, oh, ow = images.shape
    print(f"DEBUG: About to call get_image_size_for_max_num_patches with oh={oh}, ow={ow}, patch_size={patch_size}, pixel_shuffle_scale={pixel_shuffle_scale}")
    tgt_h, tgt_w = get_image_size_for_max_num_patches(
        oh, ow, patch_size, max_num_patches, min_num_patches=min_num_patches, pixel_shuffle_scale=pixel_shuffle_scale
    )
    print(f"DEBUG: get_image_size_for_max_num_patches returned tgt_h={tgt_h}, tgt_w={tgt_w}")

    # resize
    images = F.interpolate(images, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)

    # back to NHWC
    images = images.permute(0, 2, 3, 1)

    # normalize and patchify
    images = prepare_image_tensor(images)
    patches = patchify_vision(images, patch_size=patch_size)

    n, hp, wp, _ = patches.shape
    dims_virtual = [1, hp, wp] if pixel_shuffle_scale == 1 else [1, hp // pixel_shuffle_scale, wp // pixel_shuffle_scale]
    print(f"DEBUG process_vision_for_patches: patches.shape={patches.shape}, hp={hp}, wp={wp}, dims_virtual={dims_virtual}, pixel_shuffle_scale={pixel_shuffle_scale}")
    return patches, dims_virtual


# ====== kwargs typing (4.57.0) ======
class DefaultFastImageProcessorKwargsCompat(TypedDict, total=False):
    # mirror the base fast processor common kwargs (sparse; extend as needed)
    size: SizeDict
    do_resize: bool
    do_rescale: bool
    do_normalize: bool
    image_mean: Union[float, list[float]]
    image_std: Union[float, list[float]]
    resample: PILImageResampling
    interpolation: PILImageResampling
    return_tensors: Optional[Union[str, TensorType]]


class IsaacImageProcessorKwargs(DefaultFastImageProcessorKwargsCompat, total=False):
    patch_size: int
    max_num_patches: int
    min_num_patches: int
    pixel_shuffle_scale: int
    merge_size: int  # kept for parity with other processors that expose it


@auto_docstring
class IsaacImageProcessorFast(BaseImageProcessorFast):
    """Fast image processor for the Isaac model (vLLM-friendly)."""

    # default behaviors/params
    do_resize = True
    resample = PILImageResampling.BICUBIC
    do_rescale = True
    rescale_factor = ISAAC_VISION_SCALE
    do_normalize = True
    image_mean = ISAAC_VISION_MEAN
    image_std = ISAAC_VISION_STD
    do_convert_rgb = True

    size = {"shortest_edge": 224, "longest_edge": 1024}
    data_format = ChannelDimension.FIRST
    input_data_format = None
    device = None

    patch_size = 16
    merge_size = 2
    max_num_patches = 256
    min_num_patches = None
    pixel_shuffle_scale = 2
    disable_grouping = False
    temporal_patch_size = 1  # Isaac does not use temporal packing
    interpolation = PILImageResampling.BICUBIC

    valid_kwargs = IsaacImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs):
        # Isaac-specific - use class defaults for self-reference
        self.vision_patch_size = kwargs.pop("vision_patch_size", self.patch_size)
        self.vision_max_num_patches = kwargs.pop("vision_max_num_patches", self.max_num_patches)
        self.vision_min_num_patches = kwargs.pop("vision_min_num_patches", None)
        self.pixel_shuffle_scale = kwargs.pop("pixel_shuffle_scale", 2)
        
        print(f"DEBUG IsaacImageProcessorFast.__init__: pixel_shuffle_scale = {self.pixel_shuffle_scale}")

        size = kwargs.pop("size", self.size)
        super().__init__(size=size, **kwargs)

    def _further_process_kwargs(self, size: Optional[SizeDict] = None, **kwargs) -> dict:
        """Sanitize/fill defaults before BaseImageProcessorFast validation."""
        if size is None:
            size = {"shortest_edge": 224, "longest_edge": 1024}
        return super()._further_process_kwargs(size=size, **kwargs)

    #@auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[IsaacImageProcessorKwargs]) -> BatchFeature:
        # Required by BaseImageProcessorFast.validate/_preprocess
        kwargs.setdefault("do_resize", self.do_resize)
        kwargs.setdefault("size", self.size)
        kwargs.setdefault("interpolation", self.interpolation)

        kwargs.setdefault("do_rescale", self.do_rescale)
        kwargs.setdefault("rescale_factor", self.rescale_factor)

        kwargs.setdefault("do_normalize", self.do_normalize)
        kwargs.setdefault("image_mean", self.image_mean)
        kwargs.setdefault("image_std", self.image_std)

        kwargs.setdefault("do_convert_rgb", self.do_convert_rgb)
        kwargs.setdefault("input_data_format", self.input_data_format)  # often None
        kwargs.setdefault("device", self.device)                        # often None
        kwargs.setdefault("disable_grouping", self.disable_grouping)    # ← add this

        # Isaac-specific knobs so they propagate if caller omits them
        kwargs.setdefault("vision_patch_size", self.vision_patch_size)

        return super().preprocess(images, **kwargs)

    # ---- core implementations ----

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Optional[Union[str, "torch.device"]] = None,
        **kwargs,
    ) -> BatchFeature:
        """Preprocess ImageInput → BatchFeature."""
        batch_feature = BatchFeature()
        if images is not None:
            images = self._prepare_image_like_inputs(
                images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
            )
            batch_feature = self._preprocess(images, **kwargs)
        return batch_feature

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["TVF.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        patch_size: int,
        merge_size: int,
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs: Unpack[IsaacImageProcessorKwargs],
    ) -> BatchFeature:
        """Isaac's resize → normalize → patchify → pack."""

        all_pixel_values: list[torch.Tensor] = []
        all_image_grids: list[torch.Tensor] = []

        for image_tensor  in images:
            #image_tensor = extract_image_pil(image)

            
            # NCHW -> HWC (or squeeze batch)
            if image_tensor.ndim == 3 and image_tensor.shape[0] == 3:  # CHW
                image_tensor = image_tensor.permute(1, 2, 0)  # HWC
            elif image_tensor.ndim == 4 and image_tensor.shape[1] == 3:  # NCHW
                image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)  # HWC
            
            patches, dims_virtual = process_vision_for_patches(
                image_tensor,
                patch_size=patch_size,
                max_num_patches=self.vision_max_num_patches,
                min_num_patches=self.vision_min_num_patches,
                pixel_shuffle_scale=self.pixel_shuffle_scale,
            )
            
            # Debug: print what we're getting from process_vision_for_patches
            print(f"DEBUG IMAGE PROCESSOR: patches.shape = {patches.shape}")
            print(f"DEBUG IMAGE PROCESSOR: dims_virtual = {dims_virtual}")

            # Isaac packs a dummy temporal dim for images
            patches = patches.unsqueeze(1)  # [N, T=1, Hp, Wp, D]

            hp, wp, dim = patches.shape[-3], patches.shape[-2], patches.shape[-1]
            current_num_patches = hp * wp
            pixel_values = patches.reshape(current_num_patches, dim)  # [N_tokens, D]

            # Use real patch dimensions for image_grid_thw, not virtual dimensions
            # This ensures the vision model receives correct grid info for pixel shuffle
            dims_real = [1, hp, wp]  # Real patch dimensions
            image_grid_thw = torch.tensor(dims_real).unsqueeze(0)  # [1, [T, H, W]]
            print(f"DEBUG IMAGE PROCESSOR: image_grid_thw = {image_grid_thw}")

            all_pixel_values.append(pixel_values)
            all_image_grids.append(image_grid_thw)

        if all_pixel_values:
            final_pixel_values = torch.cat(all_pixel_values, dim=0)
            final_image_grids = torch.cat(all_image_grids, dim=0)
        else:
            final_pixel_values = torch.empty(0, 0)
            final_image_grids = torch.empty(0, 3)

        return BatchFeature(
            data={"pixel_values": final_pixel_values, "image_grid_thw": final_image_grids},
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        """
        Return number of patches for a given (H, W).
        vLLM calls this to infer placeholder/token counts without an actual image.
        """
        if images_kwargs is None:
            images_kwargs = {}

        patch_size = images_kwargs.get("vision_patch_size", self.vision_patch_size)
        max_num_patches = images_kwargs.get("vision_max_num_patches", self.vision_max_num_patches)
        min_num_patches = images_kwargs.get("vision_min_num_patches", self.vision_min_num_patches)
        pixel_shuffle_scale = images_kwargs.get("pixel_shuffle_scale", self.pixel_shuffle_scale)

        rh, rw = get_image_size_for_max_num_patches(
            height,
            width,
            patch_size,
            max_num_patches,
            min_num_patches=min_num_patches,
            pixel_shuffle_scale=pixel_shuffle_scale,
        )
        return (rh // patch_size) * (rw // patch_size)


__all__ = ["IsaacImageProcessorFast"]


# -------- Isaac config & HF-like processor --------
class IsaacConfig(Qwen3Config):
    model_type = "isaac"
    def __init__(
        self,
        # Isaac-specific vision parameters
        vision_patch_size: int = 16,
        vision_max_num_patches: int = 256,
        vision_min_num_patches: Optional[int] = None,
        pixel_shuffle_scale: int = 1,
        vision_token: str = "<|image_pad|>",
        # Add vision_config parameter
        vision_config: Optional[Dict] = None,
        # Qwen3 text model parameters - set defaults if not provided
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        max_position_embeddings: int = 4096,
        **kwargs,
    ):
        # Initialize Qwen3Config first with text model parameters
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            **kwargs
        )
        # Add Isaac-specific vision parameters
        self.vision_patch_size = vision_patch_size
        self.vision_max_num_patches = vision_max_num_patches
        self.vision_min_num_patches = vision_min_num_patches
        self.pixel_shuffle_scale = pixel_shuffle_scale
        self.vision_token = vision_token
        
        # Add vision_config
        if vision_config is None:
            self.vision_config = PixelShuffleSiglip2VisionConfig(
                pixel_shuffle_scale_factor=pixel_shuffle_scale,
                num_patches=self.max_num_patches,
            )
        else:
            self.vision_config = PixelShuffleSiglip2VisionConfig(**vision_config)


class IsaacProcessor:
    """Tiny HF-style processor wrapper (tokenizer + IsaacImageProcessorFast)."""
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        self.image_processor = image_processor or IsaacImageProcessorFast(**kwargs)
        self.tokenizer = tokenizer
        #self.image_token = kwargs.get("vision_token", "<|image_pad|>")
        self.image_token = "<|image_pad|>"

    def __call__(self, text=None, images=None, **kwargs) -> BatchFeature:
        result = {}
        print(f"IsaacProcessor __call__")
        if text is not None:
            if hasattr(self.tokenizer, "__call__"):
                result.update(self.tokenizer(text, **kwargs))
            else:
                result["input_ids"] = [[ord(c) for c in text]]
        if images is not None:
            image_result = self.image_processor.preprocess(images, **kwargs)
            result.update(image_result)
            print(f"IsaacProcessor __call__: image_result= {image_result}")
        return BatchFeature(result)
    
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Any:
        # Convert mixed content messages to simple text format
        processed_messages = []
        
        for message in messages:
            if "content" in message and isinstance(message["content"], list):
                # Handle mixed content (text + image)
                text_parts = []
                for content_item in message["content"]:
                    if content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
                    elif content_item.get("type") == "image":
                        # Replace image with vision token
                        text_parts.append(self.image_token)
                
                processed_message = {
                    "role": message.get("role", "user"),
                    "content": "".join(text_parts)
                }
                processed_messages.append(processed_message)
            else:
                # Regular text message
                processed_messages.append(message)
        
        return self.tokenizer.apply_chat_template(
            processed_messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs
        )


# -------- vLLM-side glue (Info / Processor / Dummies) --------

class IsaacProcessingInfo(BaseProcessingInfo):
    """Provides HF config/processor/tokenizer to vLLM processor."""

    def get_hf_config(self) -> IsaacConfig:
        if hasattr(self.ctx, "get_hf_config"):
            try:
                # Try to get our vLLM IsaacConfig first
                return self.ctx.get_hf_config(IsaacConfig)
            except TypeError:
                # If that fails, get the original config and create our version
                original_config = self.ctx.get_hf_config()
                # Map HF config parameters to our vLLM config parameters
                return IsaacConfig(
                    # Vision parameters - map from HF names
                    patch_size=getattr(original_config, "video_patch_size", 16),
                    max_num_patches=getattr(original_config, "vision_max_num_patches", 256),
                    min_num_patches=getattr(original_config, "vision_min_num_patches", None),
                    pixel_shuffle_scale=getattr(original_config, "pixel_shuffle_scale", 1),
                    merge_size=getattr(original_config, "pixel_shuffle_scale", 2),  # Use pixel_shuffle_scale as merge_size
                    vision_token="<|image_pad|>",  # Default vision token
                    # Qwen3 text model parameters - use actual values from HF config
                    vocab_size=getattr(original_config, "vocab_size", 32000),
                    hidden_size=getattr(original_config, "hidden_size", 2048),
                    intermediate_size=getattr(original_config, "intermediate_size", 5632),
                    num_hidden_layers=getattr(original_config, "num_hidden_layers", 24),
                    num_attention_heads=getattr(original_config, "num_attention_heads", 16),
                    num_key_value_heads=getattr(original_config, "num_key_value_heads", 16),
                    max_position_embeddings=getattr(original_config, "max_position_embeddings", 4096),
                )
        return IsaacConfig()

    def get_hf_processor(self, **kwargs) -> IsaacProcessor:
        if hasattr(self.ctx, "get_hf_processor"):
            return self.ctx.get_hf_processor(IsaacProcessor, **kwargs)
        cfg = self.get_hf_config()
        return IsaacProcessor(
            image_processor=IsaacImageProcessorFast(
                patch_size=cfg.vision_patch_size,
                max_num_patches=cfg.vision_max_num_patches,
                pixel_shuffle_scale=cfg.pixel_shuffle_scale,
            ),
            vision_token=cfg.vision_token,
            **kwargs,
        )

    def get_image_processor(self, **kwargs) -> IsaacImageProcessorFast:
        return self.get_hf_processor(**kwargs).image_processor

    # Remove get_tokenizer method to let vLLM handle tokenizer loading automatically

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}  # None means unlimited images are supported, like Qwen2-VL

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        cfg = self.get_hf_config()
        return {"image": cfg.vision_max_num_patches}


class IsaacMultiModalProcessor(BaseMultiModalProcessor):
    """
    Minimal processor implementing the current vLLM contract:
      - _get_mm_fields_config (required abstract)
      - _get_prompt_updates (required abstract)
    """

    def _get_mm_fields_config(self, hf_inputs: Mapping[str, Any], hf_processor_mm_kwargs: Mapping[str, Any]) -> dict[str, MultiModalFieldConfig]:
        # Configure multimodal fields for Isaac model
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        print(f"DEBUG _get_mm_fields_config: image_grid_thw.shape = {image_grid_thw.shape}")
        print(f"DEBUG _get_mm_fields_config: image_grid_thw = {image_grid_thw}")
        
        # For Isaac: image_grid_thw is [T, H, W] where T=1 for images
        # Calculate number of patches per image as H * W (not T * H * W)
        if image_grid_thw.numel() > 0:
            image_grid_sizes = image_grid_thw[:, 1] * image_grid_thw[:, 2]  # H * W only
        else:
            image_grid_sizes = torch.empty((0,))
        
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes),
            "image_grid_thw": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        """Isaac prompt updates following Qwen2VL pattern but with Isaac parameters."""
        from vllm.multimodal.processing import PromptReplacement
        from functools import partial
        
        #hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        
        if tokenizer is None:
            # Fallback - use known ID for <|image_pad|> token
            placeholder_id = 151655  
        else:
            vocab = tokenizer.get_vocab()
            placeholder_id = vocab.get("<|image_pad|>", 151655)  # Use actual vocab token
        
        # Isaac uses pixel_shuffle_scale instead of merge_size
        pixel_shuffle_scale = getattr(image_processor, 'pixel_shuffle_scale', 1)
        merge_length = pixel_shuffle_scale ** 2
        
        def get_replacement_isaac(item_idx: int, modality: str):
            if modality == "image":
                grid_thw = out_mm_kwargs["image_grid_thw"][item_idx]
                assert isinstance(grid_thw, torch.Tensor)
                # For Isaac: T, H, W format where T=1 for images
                num_tokens = int(grid_thw.prod()) // merge_length
                return [placeholder_id] * num_tokens
            return []
        
        return [
            PromptReplacement(
                modality="image",
                target=[placeholder_id],
                replacement=partial(get_replacement_isaac, modality="image"),
            )
        ]

#from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal.inputs import MultiModalDataDict
#class Qwen2VLDummyInputsBuilder(BaseDummyInputsBuilder[Qwen2VLProcessingInfo]):
class IsaacDummyInputsBuilder(BaseDummyInputsBuilder[IsaacProcessingInfo]): 
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        #num_videos = mm_counts.get("video", 0)

        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.image_token
        #video_token: str = hf_processor.video_token
        print(f"IsaacDummyInputsBuilder, get_dummy_text: num_images = {num_images}, image_token = {image_token}")

        return image_token * num_images# + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str] | None = None,#: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        #num_videos = mm_counts.get("video", 0)

        print(f"IsaacDummyInputsBuilder, get_dummy_text: num_images = {num_images}")
        #target_width, target_height = self.info.get_image_size_with_most_features()
        target_width, target_height = 224, 224
        #target_num_frames = self.info.get_num_frames_with_most_features(
        #    seq_len, mm_counts
        #)

        #image_overrides = mm_options.get("image") if mm_options else None
        #video_overrides = mm_options.get("video") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                #overrides=image_overrides,
            ),
        }


class IsaacDummyInputsBuilder2(BaseDummyInputsBuilder):
    """Supplies dummy text/images for shape/profiling runs."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        n = mm_counts.get("image", 0)
        if n <= 0: return ""
        tok = self.info.get_hf_processor().image_token
        return " ".join([tok] * n)

    def get_dummy_images(self, mm_counts: Mapping[str, int]) -> Sequence[dict[str, Any]]:
        n = mm_counts.get("image", 0)
        if n <= 0: return []
        ip = self.info.get_image_processor()
        
        # Create individual dummy PIL images and process them separately
        dummy_images = []
        for _ in range(n):
            # Create a dummy PIL image
            import PIL.Image
            dummy_array = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy()
            dummy_pil = PIL.Image.fromarray(dummy_array, mode='RGB')
            out = ip.preprocess([dummy_pil], return_tensors="pt")
            dummy_images.append(out)
        return dummy_images

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Optional[Any] = None) -> Mapping[str, Any]:
        """Required abstract method implementation."""
        dummy_data = {}
        
        # Generate dummy image data if requested
        n_images = mm_counts.get("image", 0)
        print(f"DEBUG DUMMY INPUTS: Creating {n_images} dummy images")
        if n_images > 0:
            # Create dummy images as PIL Images, which is what vLLM expects
            dummy_images = []
            for i in range(n_images):
                # Create a dummy PIL image
                import PIL.Image
                dummy_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                dummy_image = PIL.Image.fromarray(dummy_array)
                dummy_images.append(dummy_image)
                print(f"DEBUG DUMMY INPUTS: Created dummy image {i}: {dummy_image.size}, mode={dummy_image.mode}")
            
            # Return the dummy data organized by modality
            dummy_data["image"] = dummy_images
        
        print(f"DEBUG DUMMY INPUTS: Returning dummy_data with keys: {list(dummy_data.keys())}")
        return dummy_data


@MULTIMODAL_REGISTRY.register_processor(
    IsaacMultiModalProcessor,
    info=IsaacProcessingInfo,
    dummy_inputs=IsaacDummyInputsBuilder,
)
class IsaacForConditionalGeneration(Qwen3ForCausalLM, SupportsMultiModal):
    """Isaac model with vLLM MULTIMODAL_REGISTRY registration."""

    # Fixed weight mapping that matches the checkpoint structure
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_embedding.": "vision_embedding.",
            "lm_head.": "lm_head.",
            "model.language_model.": "",  # Strip this prefix to match our structure
        }
    )

    # vLLM feature flags (following Qwen2VL pattern)
    supports_encoder_tp_data = True  # Support for tensor parallelism in encoder
    
    @classmethod
    def get_isaac_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs
    ) -> tuple[torch.Tensor, int]:
        """
        Isaac-specific MRoPE position calculation using TensorStream.
        
        This replaces vLLM's generic MRoPE calculation with Isaac's sophisticated
        3D positional encoding system that handles multimodal sequences properly.
        
        Returns:
            tuple: (position_ids, delta) where position_ids is (B, L, 3) for MRoPE
        """
        # For text-only inputs, use Isaac's original logic from compute_position_ids_input_ids()
        if image_grid_thw is None or len(image_grid_thw) == 0:
            seq_len = len(input_tokens)
            # Create 3D positions where all dimensions get the same 1D temporal progression
            # This matches Isaac's compute_position_ids_input_ids() exactly
            position_ids = torch.arange(seq_len, dtype=torch.long)
            position_ids = position_ids.view(1, -1).expand(1, -1)  # [1, seq_len]
            position_ids = position_ids.unsqueeze(2).expand(-1, -1, 3)  # [1, seq_len, 3]
            
            # vLLM expects shape [3, seq_len], so transpose
            position_ids = position_ids.squeeze(0).transpose(0, 1)  # [3, seq_len]
            return position_ids, 1
        
        # For multimodal inputs, we need to create a TensorStream and use Isaac's calculation
        try:
            from torch import device as torch_device
            
            device = torch_device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Create a simple TensorStream approximation for position calculation
            # This is a simplified version that works with vLLM's token-based interface
            
            batch_size = 1
            seq_len = len(input_tokens)
            
            # Create events for text and image tokens
            # This is a simplified approximation - in reality Isaac uses more sophisticated logic
            events = []
            
            # Calculate total image tokens based on grid dimensions
            total_image_tokens = 0
            if image_grid_thw is not None:
                for thw in image_grid_thw:
                    if len(thw) >= 3:
                        t, h, w = thw[:3]
                        total_image_tokens += int(h * w)
            
            # Approximate text tokens
            text_tokens = seq_len - total_image_tokens
            
            if text_tokens > 0:
                # Text event - 1D temporal progression
                text_event = Event(
                    modality_type=TextType.text,
                    _dims=[text_tokens],
                    idx_range=(0, text_tokens),
                    device=device
                )
                events.append(text_event)
            
            if total_image_tokens > 0 and image_grid_thw is not None:
                # Image events - 3D spatial-temporal structure
                for thw in image_grid_thw:
                    if len(thw) >= 3:
                        t, h, w = thw[:3]
                        image_event = Event(
                            modality_type=VisionType.image,
                            _dims=[int(t), int(h), int(w)],
                            idx_range=(0, int(h * w)),
                            device=device
                        )
                        events.append(image_event)
            
            # Create TensorStream
            stream = Stream(events)
            tensor_stream = TensorStream([stream], device=device)
            
            # Use Isaac's native MRoPE calculation
            position_ids = compute_mrope_pos_tensor(tensor_stream, n_pos_dims=3)
            
            # Ensure proper shape and return
            if position_ids.shape[0] == 0:
                # Fallback to simple temporal positions
                position_ids = torch.zeros((batch_size, seq_len, 3), dtype=torch.long, device=device)
                position_ids[0, :, 0] = torch.arange(seq_len)
            
            # vLLM expects shape [3, seq_len] but Isaac returns [batch, seq_len, 3]
            # Transpose to match vLLM's expected format
            if position_ids.dim() == 3 and position_ids.shape[0] == 1:
                # Remove batch dimension and transpose: [1, seq_len, 3] -> [seq_len, 3] -> [3, seq_len] 
                position_ids = position_ids.squeeze(0).transpose(0, 1)
            elif position_ids.dim() == 2:
                # Already [seq_len, 3], just transpose to [3, seq_len]
                position_ids = position_ids.transpose(0, 1)
            
            return position_ids, 1
            
        except Exception as e:
            # Fallback to Isaac's text-only logic if anything goes wrong
            print(f"Isaac MRoPE calculation failed, falling back to text-only positions: {e}")
            seq_len = len(input_tokens)
            # Use same logic as text-only case above
            position_ids = torch.arange(seq_len, dtype=torch.long)
            position_ids = position_ids.view(1, -1).expand(1, -1)  # [1, seq_len]
            position_ids = position_ids.unsqueeze(2).expand(-1, -1, 3)  # [1, seq_len, 3]
            # vLLM expects shape [3, seq_len], so transpose
            position_ids = position_ids.squeeze(0).transpose(0, 1)  # [3, seq_len]
            return position_ids, 1

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        """
        Get input embeddings for Isaac model, handling both text and multimodal inputs.
        
        This method merges text embeddings with multimodal (vision) embeddings using
        Isaac's vision token ("<|image_pad|>") as the placeholder.
        """
        # Get text embeddings from the base language model
        inputs_embeds = super().get_input_embeddings(input_ids)
        
        # If we have multimodal embeddings, merge them with text embeddings
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            # Import the merge utility
            from vllm.model_executor.models.utils import merge_multimodal_embeddings
            
            # Use the image token ID we configured (151655 for <|image_pad|>)
            # Isaac doesn't use video tokens, so only provide image token ID
            vision_token_id = getattr(self.config, 'image_token_id', 151655)
            
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=vision_token_id
            )
        
        return inputs_embeds

    def __init__(self, *, vllm_config, prefix: str = ""):
        """Initialize Isaac model with vLLM config."""
        config = vllm_config.model_config.hf_config
        print(f"🔧 Initializing Isaac model with config: {config}")
        
        # Add missing image_token_id for vLLM MRoPE compatibility
        if not hasattr(config, 'image_token_id'):
            # Get the token ID for <|image_pad|> from the tokenizer 
            config.image_token_id = 151655  # This is the ID for <|image_pad|> in Qwen tokenizer
            print(f"🔧 Isaac: Added image_token_id = {config.image_token_id}")
        
        # Override rope_scaling to inject calculated mrope_section values (Isaac's approach)
        # This ensures vLLM's get_rope() creates MRotaryEmbedding with proper mrope_section
        if hasattr(config, 'rope_scaling') and config.rope_scaling:
            # Calculate mrope_section the same way original Isaac does
            head_dim = getattr(config, 'head_dim', 128)
            calculated_mrope_section = [
                head_dim // 4,  # temporal dimension gets more capacity
                head_dim // 8,  # height dimension  
                head_dim // 8,  # width dimension
            ]
            
            print(f"🔧 Isaac: Calculated mrope_section = {calculated_mrope_section} from head_dim = {head_dim}")
            
            # Inject calculated values into rope_scaling config
            if isinstance(config.rope_scaling, dict):
                config.rope_scaling = config.rope_scaling.copy()
            else:
                config.rope_scaling = {}
            
            config.rope_scaling["mrope_section"] = calculated_mrope_section
            print(f"🔧 Isaac: Updated rope_scaling = {config.rope_scaling}")
        
        # Initialize the parent class with updated config
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        # Override the language model structure to match checkpoint
        # The parent class creates self.model.layers, but we need direct access
        # Create the language model layers directly to match checkpoint structure
        self.language_model = nn.ModuleDict({
            "embed_tokens": self.model.embed_tokens,
            "layers": self.model.layers,
            "norm": self.model.norm
        })
        
        # Create vision embedding 
        vision_cfg = config.vision_config
        if vision_cfg is None:
            raise ValueError("IsaacConfig should always have vision_config")
        
        # Convert dict to PixelShuffleSiglip2VisionConfig if needed
        if isinstance(vision_cfg, dict):
            vision_cfg = PixelShuffleSiglip2VisionConfig(**vision_cfg)

        hidden_dim = vision_cfg.hidden_size * (vision_cfg.pixel_shuffle_scale_factor**2)
        self.vision_embedding = nn.Sequential(
            Siglip2SequenceVisionTransformer(vision_cfg),
            nn.Linear(
                hidden_dim,
                4 * hidden_dim,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, config.hidden_size, bias=False),
        )

        self.vocab_size = config.vocab_size
        # Use vLLM's ParallelLMHead to ensure proper quantization support
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
            quant_config=getattr(vllm_config, 'quant_config', None),
            prefix=f"{prefix}lm_head" if prefix else "lm_head"
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        """Get placeholder string for multimodal items."""
        if modality.startswith("image"):
            return "<|image_pad|>"
        raise ValueError("Only image modality is supported")

    def get_multimodal_embeddings(self, **kwargs: object):
        """Get multimodal embeddings from vision inputs."""
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")

        if pixel_values is None:
            return []

        # Debug: Check how many images we're processing
        print(f"DEBUG: Processing {pixel_values.shape[0]} images in get_multimodal_embeddings")
        print(f"DEBUG: image_grid_thw shape: {image_grid_thw.shape}")

        # Convert image_grid_thw from [batch, 1, [T, H, W]] to [batch, [H, W]]
        spatial_grids = image_grid_thw[:, 0, 1:3]  # Extract H, W from [T, H, W] for each image
        
        # Process vision through our vision_embedding module
        vision_embeddings = self.vision_embedding((pixel_values, spatial_grids))

        # Split concatenated embeddings for each image item (following Qwen2-VL pattern)
        merge_size = getattr(self.vision_embedding[0], 'pixel_shuffle_scale_factor', 2)  # Isaac uses pixel shuffle
        sizes = spatial_grids.prod(-1) // (merge_size * merge_size)  # H * W / (merge_size^2)
        
        return vision_embeddings.split(sizes.tolist())

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass with proper multimodal signature."""
        # For now, just call the parent Qwen3ForCausalLM forward method
        # In a full implementation, this would integrate vision embeddings
        return super().forward(input_ids=input_ids, positions=positions, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = []
        if self.vision_embedding is None:
            skip_prefixes.extend(["model.vision_embedding."])
        print(f"skip_prefixes = {skip_prefixes}!!!!!!!!!")
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model.layers",
            connector="vision_embedding.3",  # The final linear layer
            tower_model="vision_embedding.0",  # The vision transformer
        )


# ===== Isaac MRoPE Monkey Patch =====
# Override vLLM's generic MRoPE calculation with Isaac's TensorStream-based system

# Store the original method
_original_get_input_positions_tensor = MRotaryEmbedding.get_input_positions_tensor

def _isaac_aware_get_input_positions_tensor(
    input_tokens: list[int],
    hf_config,
    image_grid_thw=None,
    video_grid_thw=None,
    **kwargs
) -> tuple[torch.Tensor, int]:
    """
    Isaac-aware wrapper for MRoPE position calculation.
    
    Detects Isaac models and uses Isaac's native TensorStream-based calculation,
    otherwise falls back to vLLM's standard MRoPE.
    """
    # Check if this is an Isaac model (detect by model_type or config attributes)
    is_isaac_model = (
        hasattr(hf_config, 'model_type') and hf_config.model_type in ['isaac', 'qwen3'] and
        hasattr(hf_config, 'rope_scaling') and hf_config.rope_scaling and
        'mrope_section' in hf_config.rope_scaling
    )
    
    if is_isaac_model:
        # Use Isaac's custom MRoPE calculation
        try:
            return IsaacForConditionalGeneration.get_isaac_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                **kwargs
            )
        except Exception as e:
            print(f"Isaac MRoPE calculation failed, falling back to vLLM default: {e}")
            # Fall through to standard vLLM calculation
    
    # Use standard vLLM MRoPE calculation
    return _original_get_input_positions_tensor(
        input_tokens=input_tokens,
        hf_config=hf_config,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        **kwargs
    )

# Apply the monkey patch
MRotaryEmbedding.get_input_positions_tensor = staticmethod(_isaac_aware_get_input_positions_tensor)

print("🔧 Isaac MRoPE monkey patch applied successfully!")