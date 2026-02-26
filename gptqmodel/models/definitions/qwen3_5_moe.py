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


_patch_qwen3_5_moe_transformers()


def _patch_init_weights():
    """Patch _init_weights to skip decomposed experts (avoids gate_up_proj AttributeError)."""
    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mod
    except ImportError:
        return

    # Find the PreTrainedModel subclass that defines _init_weights
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name, None)
        if isinstance(obj, type) and '_init_weights' in getattr(obj, '__dict__', {}):
            _orig_init_weights = obj._init_weights

            def _patched_init_weights(self, module, _orig=_orig_init_weights):
                if isinstance(module, (Qwen3_5MoeExpertsDecomposed, Qwen3_5MoeExpert)):
                    return
                _orig(self, module)

            obj._init_weights = _patched_init_weights
            break


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
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn:?": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "linear_attn:?": ("in_proj_qkv:0", "out_proj:1"),
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

    support_offload_to_disk = False

    def before_model_load(self, load_quantized_model=False):
        """Replace fused experts for quantized model loading."""
        if load_quantized_model:
            try:
                from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mod
                mod.Qwen3_5MoeExperts = Qwen3_5MoeExpertsDecomposed
                _patch_init_weights()
            except ImportError:
                pass

    def after_model_load(self, model, load_quantized_model=False):
        """Decompose fused experts into individual nn.Linear modules for quantization."""
        import gc

        if load_quantized_model:
            return model

        try:
            from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mod
        except ImportError:
            return model

        OrigExperts = getattr(mod, 'Qwen3_5MoeExperts', None)
        if OrigExperts is None:
            return model

        config = getattr(model.config, 'text_config', model.config)
        converted = 0
        layers = model.model.layers if hasattr(model, 'model') and hasattr(model.model, 'layers') else []
        for layer in layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts') and isinstance(layer.mlp.experts, OrigExperts):
                ori = layer.mlp.experts
                layer.mlp.experts = Qwen3_5MoeExpertsDecomposed(config=config, ori_experts=ori)
                del ori
                converted += 1
                if converted % 4 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()
        log.info(f"Decomposed fused experts in {converted} layers for quantization.")
        return model
