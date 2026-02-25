# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from ..base import BaseQModel
from ...utils.logger import setup_logger

log = setup_logger()


def _patch_qwen3_5_moe_transformers():
    """Fix transformers bug: Qwen3_5MoeConfig composite config lacks text_config attributes at top level.

    The Qwen3_5MoeConfig is a composite config with text_config and vision_config sub-configs.
    Multiple classes (PreTrainedModel, ForConditionalGeneration, TextModel) access attributes
    like vocab_size, hidden_size directly from config, but these only exist in text_config.

    We fix this at the root by patching Qwen3_5MoeConfig to promote text_config attributes
    to the top level, and also patching ForConditionalGeneration to use the correct Model wrapper.
    """
    # Patch 1: Promote text_config attributes to composite config top level.
    # This is the most comprehensive fix - any code accessing config.vocab_size will work.
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
        log.info("Patched Qwen3_5MoeConfig to promote text_config attributes.")
    except ImportError:
        pass

    # Patch 2: Fix ForConditionalGeneration to use Qwen3_5MoeModel (multimodal wrapper)
    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mod
    except ImportError:
        return

    ForCG = getattr(mod, 'Qwen3_5MoeForConditionalGeneration', None)
    Model = getattr(mod, 'Qwen3_5MoeModel', None)
    TextModel = getattr(mod, 'Qwen3_5MoeTextModel', None)

    if ForCG is None:
        return

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

    # Patch 3 (fallback): Make TextModel accept composite configs
    if TextModel is not None:
        _orig_text_init = TextModel.__init__

        def _patched_text_init(self, config, *args, **kwargs):
            if hasattr(config, 'text_config') and not hasattr(config, 'vocab_size'):
                config = config.text_config
            _orig_text_init(self, config, *args, **kwargs)

        TextModel.__init__ = _patched_text_init


_patch_qwen3_5_moe_transformers()


class Qwen3_5MoeGPTQ(BaseQModel):
    require_monkeypatch = False

    # Layers alternate between linear_attention and full_attention,
    # so not all modules exist in every layer.
    layer_modules_strict = False

    # num_experts is in text_config; base.py get_num_experts handles text_config lookup
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.norm"

    # Qwen3.5 MoE: composite config patched so ForConditionalGeneration creates TextModel directly.
    # Model structure: ForCG.model = TextModel (layers, embed_tokens, norm).
    # Layers alternate between linear_attention (GatedDeltaNet) and full_attention.
    # Experts use fused 3D Parameters (gate_up_proj, down_proj) - not quantizable with standard GPTQ.
    # Only self_attn, linear_attn projections, and shared_expert are quantized here.
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
                "shared_expert": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "shared_expert_gate": ("shared_expert_gate:!",),
            },
        }
    ]
