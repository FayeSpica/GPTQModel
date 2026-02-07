# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from torch import nn

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


# GLM is HF-ied ChatGLM and marked by -HF suffix in THUDM hf repos
class GlmQModel(BaseQModel):
    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_up_proj:0", "down_proj:1"),
        }
    ]


# Single expert module with gate_proj, up_proj, down_proj
class Glm4MoeLiteExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dtype=None, device=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, hidden_states, act_fn):
        return self.down_proj(act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# New module class to decompose merged gate_up_proj into separate experts
# for GLM-4.7-Flash MoE quantization
# Original: gate_up_proj (n_experts, intermediate*2, hidden), down_proj (n_experts, hidden, intermediate)
# New: ModuleList of experts, each with gate_proj, up_proj, down_proj
class Glm4MoeLiteNaiveMoeNew(nn.Module):
    def __init__(self, config, ori_experts=None):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        dtype = None
        device = None
        if ori_experts is not None:
            dtype = ori_experts.gate_up_proj.dtype
            device = ori_experts.gate_up_proj.device

        # Create experts as ModuleList, each expert has gate_proj, up_proj, down_proj
        self.experts = nn.ModuleList([
            Glm4MoeLiteExpert(self.hidden_size, self.intermediate_size, dtype=dtype, device=device)
            for _ in range(self.num_experts)
        ])
        self.act_fn = nn.SiLU()

        if ori_experts is not None:
            # Decompose gate_up_proj: (n_experts, intermediate_size*2, hidden_size)
            # -> gate: (n_experts, intermediate_size, hidden_size)
            # -> up: (n_experts, intermediate_size, hidden_size)
            gate_up = ori_experts.gate_up_proj.data  # (64, 3072, 2048)
            gate_w = gate_up[:, :self.intermediate_size, :]  # (64, 1536, 2048)
            up_w = gate_up[:, self.intermediate_size:, :]  # (64, 1536, 2048)
            down_w = ori_experts.down_proj.data  # (64, 2048, 1536)

            for i in range(self.num_experts):
                self.experts[i].gate_proj.weight.data.copy_(gate_w[i])
                self.experts[i].up_proj.weight.data.copy_(up_w[i])
                self.experts[i].down_proj.weight.data.copy_(down_w[i])

    def forward(self, hidden_states, topk_idx, topk_weight):
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)  # (batch*seq, hidden)

        # topk_idx: (num_tokens, topk)
        # topk_weight: (num_tokens, topk)
        final_output = torch.zeros_like(hidden_states)

        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (topk_idx == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            expert_tokens = hidden_states[expert_mask]

            # Compute expert output
            expert_out = self.experts[expert_idx](expert_tokens, self.act_fn)

            # Get weights for this expert
            weight_mask = (topk_idx[expert_mask] == expert_idx)
            expert_weights = (topk_weight[expert_mask] * weight_mask.float()).sum(dim=-1, keepdim=True)

            final_output[expert_mask] += expert_out * expert_weights

        return final_output.view(orig_shape)


# GLM-4 MoE Lite (e.g. GLM-4.7-Flash) uses LoRA-like decomposed attention projections
class Glm4MoeLiteQModel(BaseQModel):
    # Allow dynamic expert index for layer_modules
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    # Set to False since layer 0 has different structure (standard MLP vs MoE)
    layer_modules_strict = False

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_a_proj:0", "q_a_layernorm:0:!", "q_b_proj:0",
                "kv_a_proj_with_mqa:0", "kv_a_layernorm:0:!", "kv_b_proj:0",
                "o_proj:1"
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                # Router - do not quantize
                "gate": ("gate:!",),
                # MoE experts (layers 1-46) - decomposed structure: experts.experts.#.{gate_proj, up_proj, down_proj}
                # After replacement, mlp.experts becomes Glm4MoeLiteNaiveMoeNew with .experts ModuleList
                "experts": {
                    "experts": {
                        "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                    },
                },
                # Shared experts (layers 1-46)
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                # Standard MLP for layer 0
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        }
    ]

    def before_model_load(self, load_quantized_model=False):
        if load_quantized_model:
            # Replace module class for loading quantized model
            try:
                import transformers.models.glm4_moe_lite.modeling_glm4_moe_lite as glm4_moe_lite_modeling
                glm4_moe_lite_modeling.Glm4MoeLiteNaiveMoe = Glm4MoeLiteNaiveMoeNew
            except ImportError:
                pass

    def after_model_load(self, model, load_quantized_model=False):
        if not load_quantized_model:
            # For quantization: replace experts module instances to decompose gate_up_proj
            # Layer 0 has standard MLP (Glm4MoeLiteMLP), layers 1-46 have MoE (Glm4MoeLiteMoE)
            config = model.config
            for layer in model.model.layers:
                mlp = layer.mlp
                # Check if this is a MoE layer (has experts attribute with gate_up_proj)
                if hasattr(mlp, 'experts') and hasattr(mlp.experts, 'gate_up_proj'):
                    # Replace with decomposed structure
                    new_experts = Glm4MoeLiteNaiveMoeNew(config, ori_experts=mlp.experts)
                    mlp.experts = new_experts
        return model
