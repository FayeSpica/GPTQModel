# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from torch import nn

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


# Single expert module with gate_proj, up_proj, down_proj
class Glm4MoeLiteExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dtype=None, device=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, hidden_states, act_fn):
        return self.down_proj(act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# ModuleList subclass to decompose merged gate_up_proj into separate experts
# for GLM-4.7-Flash MoE quantization
# Original: gate_up_proj (n_experts, intermediate*2, hidden), down_proj (n_experts, hidden, intermediate)
# New: ModuleList of experts, each with gate_proj, up_proj, down_proj
# Path: mlp.experts.{i}.gate_proj (aligns with glm4_moe pattern)
class Glm4MoeLiteNaiveMoeNew(nn.ModuleList):
    def __init__(self, config, ori_experts=None):
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        dtype = None
        device = None
        if ori_experts is not None:
            dtype = ori_experts.gate_up_proj.dtype
            # Don't use meta device
            if str(ori_experts.gate_up_proj.device) != 'meta':
                device = ori_experts.gate_up_proj.device

        # Create experts as ModuleList, each expert has gate_proj, up_proj, down_proj
        experts = [
            Glm4MoeLiteExpert(self.hidden_size, self.intermediate_size, dtype=dtype, device=device)
            for _ in range(self.num_experts)
        ]
        super().__init__(experts)
        self.act_fn = nn.SiLU()

        if ori_experts is not None and str(ori_experts.gate_up_proj.device) != 'meta':
            # Decompose gate_up_proj: (n_experts, intermediate_size*2, hidden_size)
            # -> gate: (n_experts, intermediate_size, hidden_size)
            # -> up: (n_experts, intermediate_size, hidden_size)
            gate_up = ori_experts.gate_up_proj.data  # (64, 3072, 2048)
            gate_w = gate_up[:, :self.intermediate_size, :]  # (64, 1536, 2048)
            up_w = gate_up[:, self.intermediate_size:, :]  # (64, 1536, 2048)
            down_w = ori_experts.down_proj.data  # (64, 2048, 1536)

            for i in range(self.num_experts):
                self[i].gate_proj.weight.data.copy_(gate_w[i])
                self[i].up_proj.weight.data.copy_(up_w[i])
                self[i].down_proj.weight.data.copy_(down_w[i])

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Custom state_dict loading to decompose merged gate_up_proj into separate experts."""
        # Check if we're loading from original format (gate_up_proj, down_proj)
        gate_up_key = prefix + "gate_up_proj"
        down_key = prefix + "down_proj"

        if gate_up_key in state_dict and down_key in state_dict:
            # Loading from original merged format - decompose
            gate_up = state_dict.pop(gate_up_key)  # (n_experts, intermediate*2, hidden)
            down = state_dict.pop(down_key)  # (n_experts, hidden, intermediate)

            # Decompose gate_up into gate and up for each expert
            for i in range(self.num_experts):
                expert_prefix = f"{prefix}{i}."
                gate_w = gate_up[i, :self.intermediate_size, :]  # (intermediate, hidden)
                up_w = gate_up[i, self.intermediate_size:, :]  # (intermediate, hidden)
                down_w = down[i]  # (hidden, intermediate)

                state_dict[expert_prefix + "gate_proj.weight"] = gate_w
                state_dict[expert_prefix + "up_proj.weight"] = up_w
                state_dict[expert_prefix + "down_proj.weight"] = down_w

        # Call parent implementation
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

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
            expert_out = self[expert_idx](expert_tokens, self.act_fn)

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
    # Also layer 47 has special MTP structure (embed_tokens, shared_head, eh_proj, etc.)
    layer_modules_strict = False

    # MTP (Multi-Token Prediction) tensors are stored separately and not quantized
    out_of_model_tensor_files = ["mtp.safetensors"]

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
                # Router - do not quantize (layers 1-46 only)
                "gate": ("gate:!",),
                # MoE experts (layers 1-46) - decomposed into ModuleList: experts.#.{gate_proj, up_proj, down_proj}
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                # Shared experts (layers 1-46)
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                # Standard MLP for layer 0 (Glm4MoeLiteMLP has gate_proj, up_proj, down_proj directly)
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        }
    ]

    def before_model_load(self, load_quantized_model=False):
        if load_quantized_model:
            import transformers.models.glm4_moe_lite.modeling_glm4_moe_lite as glm4_moe_lite_modeling

            glm4_moe_lite_modeling.Glm4MoeLiteNaiveMoe = Glm4MoeLiteNaiveMoeNew
