# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from ..base import BaseQModel
from ...utils.logger import setup_logger

log = setup_logger()


def _patch_qwen3_5_moe_transformers():
    """Fix transformers bug: Qwen3_5MoeForConditionalGeneration passes composite config to TextModel.

    In some transformers versions, ForConditionalGeneration.__init__ does:
        self.model = Qwen3_5MoeTextModel(config)
    instead of:
        self.model = Qwen3_5MoeModel(config)

    This causes AttributeError because the composite config lacks vocab_size at top level.
    We fix by ensuring ForConditionalGeneration uses the multimodal Model wrapper,
    and as a fallback, patching TextModel to accept composite configs.
    """
    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mod
    except ImportError:
        return

    ForCG = getattr(mod, 'Qwen3_5MoeForConditionalGeneration', None)
    Model = getattr(mod, 'Qwen3_5MoeModel', None)
    TextModel = getattr(mod, 'Qwen3_5MoeTextModel', None)

    if ForCG is None or TextModel is None:
        return

    # Patch 1: Fix ForConditionalGeneration to use Qwen3_5MoeModel (multimodal wrapper)
    if Model is not None:
        _orig_forcg_init = ForCG.__init__

        def _patched_forcg_init(self, config):
            super(ForCG, self).__init__(config)
            self.model = Model(config)
            self.lm_head = nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )
            self.post_init()

        ForCG.__init__ = _patched_forcg_init
        log.info("Patched Qwen3_5MoeForConditionalGeneration to use Qwen3_5MoeModel wrapper.")

    # Patch 2 (fallback): Make TextModel accept composite configs
    _orig_text_init = TextModel.__init__

    def _patched_text_init(self, config, *args, **kwargs):
        if hasattr(config, 'text_config') and not hasattr(config, 'vocab_size'):
            config = config.text_config
        _orig_text_init(self, config, *args, **kwargs)

    TextModel.__init__ = _patched_text_init


_patch_qwen3_5_moe_transformers()


class Qwen3_5MoeGPTQ(BaseQModel):
    require_monkeypatch = False

    # num_experts is in text_config; base.py get_num_experts handles text_config lookup
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.language_model.norm"

    # Qwen3.5 MoE is a multimodal model (vision + language MoE).
    # Weight keys use model.language_model.layers.* structure.
    # Layers alternate between linear_attention (GatedDeltaNet) and full_attention.
    # Experts use fused 3D Parameters (gate_up_proj, down_proj) - not quantizable with standard GPTQ.
    # Only self_attn, linear_attn projections, and shared_expert are quantized here.
    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn:?": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "linear_attn:?": ("in_proj_qkv:0", "out_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "shared_expert": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "shared_expert_gate": ("shared_expert_gate:!",),
            },
        }
    ]
