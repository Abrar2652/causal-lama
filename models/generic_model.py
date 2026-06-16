"""generic_model.py — a family-agnostic LanguageModel for cross-model RCS breadth.

Adding one bespoke class per model family is unnecessary for the RCS analysis
(Table 1), which only needs per-layer hidden states. GenericHFModel loads any
standard decoder-only HF CausalLM, derives num_layer/hidden_dim from its config,
formats prompts via the tokenizer's own chat template, and inherits the standard
weight-ablation path. It works out-of-the-box for Llama/Qwen/Gemma/Phi/Mistral/
DeepSeek-distill-style architectures (model.model.layers + o_proj/down_proj).

For RCS-only coverage, ablation is never called; for ablation-capable families it
reuses the base o_proj/down_proj/embed_tokens hooks. Non-standard module names
(e.g. MoE experts) should still get a dedicated subclass.
"""
import torch.nn as nn
from models.language_models import LanguageModel
from utils.ablation_utils import get_ablated_matrix


class GenericHFModel(LanguageModel):
    def __init__(self, model_name, device='cuda', system_prompt=None, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        cfg = self.model.config
        # text-config fallback for multimodal checkpoints that nest a text_config
        text_cfg = getattr(cfg, 'text_config', None) or cfg
        self.num_layer = getattr(text_cfg, 'num_hidden_layers', None) or getattr(cfg, 'num_hidden_layers', None)
        self.hidden_dimension = getattr(text_cfg, 'hidden_size', None) or getattr(cfg, 'hidden_size', None)
        if self.num_layer is None or self.hidden_dimension is None:
            raise ValueError(f"GenericHFModel: could not infer num_layer/hidden_dim from {type(cfg).__name__}")
        self.template_name = 'generic'
        self.refusal_token_id = []

    def load_model(self):
        """Load with the model's native dtype ('auto') so bf16-native models
        (e.g. gpt-oss) don't break under the base loader's hard-coded fp16."""
        if self.model is not None and self.tokenizer is not None:
            return
        import os
        from transformers import AutoTokenizer, AutoModelForCausalLM
        token = os.environ.get('HF_TOKEN')
        print(f"Downloading and loading model {self.model_name} (dtype=auto)...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, token=token, torch_dtype='auto',
            device_map=self.device, trust_remote_code=True)
        self.model.requires_grad_(False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '<|extra_0|>'
            self.tokenizer.padding_side = 'left'
        print("Model loaded successfully.")

    def _get_model_block_modules(self):
        """Robustly locate the decoder-layer ModuleList across architectures
        (incl. multimodal nesting like Gemma3's model.model.language_model.layers)."""
        paths = [
            lambda: self.model.model.layers,
            lambda: self.model.model.language_model.layers,
            lambda: self.model.language_model.model.layers,
            lambda: self.model.language_model.layers,
            lambda: self.model.transformer.h,
        ]
        for p in paths:
            try:
                m = p()
                if isinstance(m, nn.ModuleList) and len(m) > 0:
                    return m
            except AttributeError:
                continue
        # fallback: the largest ModuleList in the model is almost always the decoder stack
        best = None
        for _, mod in self.model.named_modules():
            if isinstance(mod, nn.ModuleList) and (best is None or len(mod) > len(best)):
                best = mod
        if best is not None and len(best) > 0:
            return best
        raise AttributeError("GenericHFModel: could not locate decoder layers")

    def _get_embedding_weight(self):
        """Use the HF standard accessor — robust to module-path differences."""
        return self.model.get_input_embeddings().weight

    def _get_block_proj_weights(self, block):
        """Standard attn-out + mlp-down projections; tolerate naming variants."""
        attn = getattr(block, 'self_attn', getattr(block, 'attn', None))
        mlp = getattr(block, 'mlp', None)
        o_proj = getattr(attn, 'o_proj', getattr(attn, 'c_proj', None))
        down_proj = getattr(mlp, 'down_proj', getattr(mlp, 'c_proj', None))
        return o_proj.weight, down_proj.weight

    def _get_prompt(self, prompt=''):
        """Format via the model's own chat template (most portable)."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # last-resort plain format if the tokenizer lacks a chat template
            sysp = f"{self.system_prompt}\n\n" if self.system_prompt else ""
            return f"{sysp}{prompt}"

    def get_representations(self, prompt, token_pos):
        """Clean forward pass for hidden states — no generation kwargs (some
        architectures, e.g. Phi3, reject do_sample/temperature in forward())."""
        import torch
        formatted = self._get_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, output_hidden_states=True)
        hs = torch.cat(outputs.hidden_states[1:])[:, token_pos, :].float()
        return hs.reshape(1, self.num_layer, self.hidden_dimension)

    def ablate_weights(self, direction):
        """Standard weight-orthogonalization for model.model.layers architectures."""
        emb = self._get_embedding_weight()
        emb.data = get_ablated_matrix(emb.data, direction)
        for block in self.model_block_modules:
            o_proj, down_proj = self._get_block_proj_weights(block)
            o_proj.data = get_ablated_matrix(o_proj.data.T, direction).T
            down_proj.data = get_ablated_matrix(down_proj.data.T, direction).T
