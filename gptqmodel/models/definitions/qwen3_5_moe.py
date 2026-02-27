# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks
from ...utils.logger import setup_logger

log = setup_logger()


class Qwen3_5MoeExpert(nn.Module):
    """Single expert with individual gate_proj, up_proj, down_proj."""

    def __init__(self, hidden_size, intermediate_size, dtype=None, device=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3_5MoeExpertsDecomposed(nn.ModuleList):
    """Decompose fused gate_up_proj (num_experts, 2*inter, hidden) into ModuleList of experts.

    Original Qwen3_5MoeExperts stores fused 3D Parameters which can't be quantized with GPTQ.
    This class creates individual nn.Linear modules per expert and copies the fused weights.
    """

    def __init__(self, config, ori_experts=None):
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        dtype = None
        device = None
        if ori_experts is not None:
            dtype = ori_experts.gate_up_proj.dtype
            if str(ori_experts.gate_up_proj.device) != 'meta':
                device = ori_experts.gate_up_proj.device
        else:
            dtype_str = getattr(config, 'dtype', None) or getattr(config, 'torch_dtype', None)
            if dtype_str and isinstance(dtype_str, str):
                dtype = getattr(torch, dtype_str, None)
            elif isinstance(dtype_str, torch.dtype):
                dtype = dtype_str

        super().__init__([
            Qwen3_5MoeExpert(self.hidden_size, self.intermediate_size, dtype=dtype, device=device)
            for _ in range(self.num_experts)
        ])
        self.act_fn = nn.SiLU()

        if ori_experts is not None and str(ori_experts.gate_up_proj.device) != 'meta':
            gate_up = ori_experts.gate_up_proj.data
            down_w = ori_experts.down_proj.data
            for i in range(self.num_experts):
                self[i].gate_proj.weight.data.copy_(gate_up[i, :self.intermediate_size, :])
                self[i].up_proj.weight.data.copy_(gate_up[i, self.intermediate_size:, :])
                self[i].down_proj.weight.data.copy_(down_w[i])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)

        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_index == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            expert_tokens = hidden_states[expert_mask]
            expert_out = self[expert_idx](expert_tokens)
            weight_mask = (top_k_index[expert_mask] == expert_idx)
            expert_weights = (top_k_weights[expert_mask] * weight_mask.float()).sum(dim=-1, keepdim=True)
            final_hidden_states[expert_mask] += expert_out * expert_weights

        return final_hidden_states


def _patch_qwen3_5_moe_transformers():
    """Fix transformers bugs for Qwen3.5 MoE:
    1. Promote text_config attributes to composite config top level
    2. Make TextModel accept composite configs (fallback)
    """
    # Patch 1: Promote text_config attributes to composite config top level.
    try:
        from transformers import Qwen3_5MoeConfig

        _orig_config_init = Qwen3_5MoeConfig.__init__

        def _patched_config_init(self, *args, **kwargs):
            _orig_config_init(self, *args, **kwargs)
            if hasattr(self, 'text_config'):
                tc = self.text_config
                tc_dict = tc.to_dict() if hasattr(tc, 'to_dict') else {}
                for key, value in tc_dict.items():
                    if not hasattr(self, key):
                        setattr(self, key, value)

        Qwen3_5MoeConfig.__init__ = _patched_config_init
    except ImportError:
        pass

    # Patch 2: Make TextModel accept composite configs
    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mod
    except ImportError:
        return

    TextModel = getattr(mod, 'Qwen3_5MoeTextModel', None)
    if TextModel is not None:
        _orig_text_init = TextModel.__init__

        def _patched_text_init(self, config, *args, **kwargs):
            if hasattr(config, 'text_config') and not hasattr(config, 'vocab_size'):
                config = config.text_config
            _orig_text_init(self, config, *args, **kwargs)

        TextModel.__init__ = _patched_text_init


def _patch_init_weights():
    """Patch _init_weights to skip decomposed experts (avoids gate_up_proj AttributeError)."""
    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mod
    except ImportError:
        return

    # Patch ALL classes that define _init_weights (not just the first one)
    patched_count = 0
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name, None)
        if isinstance(obj, type) and '_init_weights' in getattr(obj, '__dict__', {}):
            _orig_init_weights = obj._init_weights

            # Create closure that properly captures the original function
            def _make_patched_init_weights(orig_fn):
                def _patched_init_weights(self, module):
                    # Skip decomposed experts to avoid gate_up_proj AttributeError
                    if isinstance(module, (Qwen3_5MoeExpertsDecomposed, Qwen3_5MoeExpert)):
                        return
                    # Also check by class name in case isinstance fails
                    if type(module).__name__ in ['Qwen3_5MoeExpertsDecomposed', 'Qwen3_5MoeExpert']:
                        return
                    # Call original for all other modules
                    return orig_fn(self, module)
                return _patched_init_weights

            obj._init_weights = _make_patched_init_weights(_orig_init_weights)
            patched_count += 1

    if patched_count > 0:
        log.debug(f"Patched _init_weights in {patched_count} Qwen3.5 MoE classes")


# Apply patches at module load time (before model instantiation)
_patch_qwen3_5_moe_transformers()
_patch_init_weights()


class Qwen3_5MoeGPTQ(BaseQModel):
    require_monkeypatch = False

    # Layers alternate between linear_attention and full_attention,
    # so not all modules exist in every layer.
    layer_modules_strict = False

    # num_experts is in text_config; base.py get_num_experts handles text_config lookup
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.norm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # Qwen3.5 MoE model structure after converter decomposes experts:
    # Layers alternate between linear_attention (GatedDeltaNet) and full_attention.
    # Path: model.model.model.layers (ForCausalLM -> .model -> TextModel -> .layers)
    module_tree = [
        "model",  # Qwen3_5MoeForCausalLM.model -> Qwen3_5MoeTextModel
        "layers", # Qwen3_5MoeTextModel.layers -> ModuleList[40]
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn:?": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "linear_attn:?": (
                "in_proj_qkv:0",
                "in_proj_z:0",  # Added missing linear_attn modules
                "in_proj_b:0",  # Added missing linear_attn modules
                "in_proj_a:0",  # Added missing linear_attn modules
                "out_proj:1"
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_expert": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "shared_expert_gate": ("shared_expert_gate:!",),
            },
        }
    ]

    # offload_to_disk works with pre_quantize per-layer decomposition
    support_offload_to_disk = True


class Qwen3_5MoeForConditionalGenerationGPTQ(Qwen3_5MoeGPTQ):
    """Qwen3.5 MoE multimodal model (ForConditionalGeneration)

    This is for Qwen3.5-35B-A3B and similar multimodal models that have:
    - model.layers (text layers - same structure as CausalLM)
    - model.visual (vision encoder)
    - mtp (multimodal text processor - optional)

    Note: Despite being ForConditionalGeneration, this model uses the same
    structure as CausalLM (model.layers), not model.language_model.layers
    """

    # Use same module_tree as parent class (Qwen3_5MoeGPTQ)
    # Path: model.layers (same as CausalLM variant)
    # No need to override module_tree or pre_lm_head_norm_module
    pass

    def before_model_load(self, load_quantized_model=False):
        """Replace fused experts when loading quantized models.

        For unquantized models: experts will be decomposed by converter
        For quantized models: experts are already decomposed in checkpoint
        """
        # Only replace class when loading quantized models
        # For unquantized models, let transformers load normally with fused experts
        # then decompose via converter
        if load_quantized_model:
            try:
                from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mod
                mod.Qwen3_5MoeExperts = Qwen3_5MoeExpertsDecomposed
                log.info("Qwen3.5 MoE: Expert decomposition applied for loading quantized model")
            except ImportError:
                pass

    def pre_quantize(self, module):
        """Decompose fused experts per-layer before quantization."""
        import gc

        module = super().pre_quantize(module)

        # Detect fused experts by attribute (gate_up_proj) instead of class name
        if hasattr(module, 'mlp') and hasattr(module.mlp, 'experts') and hasattr(module.mlp.experts, 'gate_up_proj'):
            config = getattr(self.model.config, 'text_config', self.model.config)
            ori = module.mlp.experts
            log.info(f"Decomposing fused experts: {type(ori).__name__}, gate_up_proj shape={ori.gate_up_proj.shape}")
            module.mlp.experts = Qwen3_5MoeExpertsDecomposed(config=config, ori_experts=ori)
            del ori
            gc.collect()
            torch.cuda.empty_cache()

        return module

    def post_quantize(self, module):
        """Offload quantized layer to disk to free memory on unified-memory systems (e.g. DGX GB10)."""
        import gc

        if getattr(self.quantize_config, 'offload_to_disk', False):
            from ...utils.offload import offload_to_disk as _offload_fn
            _offload_fn(module=module, model=self.model, disk_path=self.quantize_config.offload_to_disk_path)
            gc.collect()
            torch.cuda.empty_cache()
            return module
        return super().post_quantize(module)
