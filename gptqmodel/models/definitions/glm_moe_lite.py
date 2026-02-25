# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from torch import nn

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Glm4MoeLiteExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dtype=None, device=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, hidden_states, act_fn):
        return self.down_proj(act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Glm4MoeLiteNaiveMoeNew(nn.ModuleList):
    """Decompose merged gate_up_proj (n_experts, intermediate*2, hidden) into ModuleList of experts."""

    def __init__(self, config, ori_experts=None):
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        dtype = None
        device = None
        if ori_experts is not None:
            dtype = ori_experts.gate_up_proj.dtype
            if str(ori_experts.gate_up_proj.device) != 'meta':
                device = ori_experts.gate_up_proj.device

        super().__init__([
            Glm4MoeLiteExpert(self.hidden_size, self.intermediate_size, dtype=dtype, device=device)
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

    def forward(self, hidden_states, topk_idx, topk_weight):
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        final_output = torch.zeros_like(hidden_states)

        for expert_idx in range(self.num_experts):
            expert_mask = (topk_idx == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            expert_tokens = hidden_states[expert_mask]
            expert_out = self[expert_idx](expert_tokens, self.act_fn)
            weight_mask = (topk_idx[expert_mask] == expert_idx)
            expert_weights = (topk_weight[expert_mask] * weight_mask.float()).sum(dim=-1, keepdim=True)
            final_output[expert_mask] += expert_out * expert_weights

        return final_output.view(orig_shape)


class Glm4MoeLiteQModel(BaseQModel):
    # GLM-4.7-Flash MoE Model Structure:
    # Layer 0: Standard MLP (no MoE experts)
    # Layers 1-46: MoE with shared_experts and individual experts (64 experts)
    # Layer 47: Special MTP structure (embed_tokens, shared_head, eh_proj, etc.)
    #
    # Original Glm4MoeLiteNaiveMoe uses merged gate_up_proj (3D Parameter).
    # Converter decomposes into ModuleList of experts for quantization.
    # before_model_load replaces class for loading quantized models.
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    layer_modules_strict = False

    out_of_model_tensor_files = ["mtp.safetensors"]

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
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        }
    ]

    def before_model_load(self, load_quantized_model=False):
        if load_quantized_model:
            import transformers.models.glm4_moe_lite.modeling_glm4_moe_lite as glm4_moe_lite_modeling

            glm4_moe_lite_modeling.Glm4MoeLiteNaiveMoe = Glm4MoeLiteNaiveMoeNew
