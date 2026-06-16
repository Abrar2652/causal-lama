"""Validate ablate_weights_adaptive (the fixed #2) without loading a real model.

Builds a tiny stand-in with the same hooks (_get_embedding_weight,
_get_block_proj_weights, model_block_modules) and confirms:
  1. per-layer alpha actually changes per-block weights by layer;
  2. the embedding edit now VARIES across ablation modes (uniform vs rcs) —
     the bug we fixed was that it did NOT (single avg_alpha + all directions);
  3. layers with alpha=0 are left untouched.
"""
import os, sys, types, copy
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub optional runtime deps not needed for the pure ablation math.
if 'fastchat' not in sys.modules:
    fc = types.ModuleType('fastchat')
    fc_model = types.ModuleType('fastchat.model')
    fc_model.get_conversation_template = lambda *a, **k: None
    fc.model = fc_model
    sys.modules['fastchat'] = fc
    sys.modules['fastchat.model'] = fc_model

from models.language_models import LanguageModel


class FakeBlock:
    def __init__(self, d):
        self.o = torch.nn.Parameter(torch.randn(d, d), requires_grad=False)
        self.dn = torch.nn.Parameter(torch.randn(d, d), requires_grad=False)


class FakeModel:
    """Minimal object exposing the hooks ablate_weights_adaptive needs."""
    def __init__(self, n_layers=6, d=16, seed=0):
        torch.manual_seed(seed)
        self.embed = torch.nn.Parameter(torch.randn(100, d), requires_grad=False)
        self.blocks = [FakeBlock(d) for _ in range(n_layers)]
        self.model_block_modules = self.blocks

    def _get_embedding_weight(self):
        return self.embed

    def _get_block_proj_weights(self, block):
        return block.o, block.dn


def run(layer_weights, layer_groups, group_dirs, seed=0):
    fm = FakeModel(seed=seed)
    before_blocks = [(b.o.data.clone(), b.dn.data.clone()) for b in fm.blocks]
    before_embed = fm.embed.data.clone()
    # bind the real method onto the fake object
    LanguageModel.ablate_weights_adaptive(fm, group_dirs, layer_groups, layer_weights)
    return fm, before_blocks, before_embed


d = 16
g0 = [torch.randn(d) for _ in range(2)]
g1 = [torch.randn(d) for _ in range(2)]
group_dirs = {0: g0, 1: g1}
# layers 0-2 -> group 0, layers 3-5 -> group 1
layer_groups = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

print("== Test 1: per-layer alpha changes blocks; alpha=0 untouched ==")
lw = {0: 1.0, 1: 0.0, 2: 0.5, 3: 1.0, 4: 0.3, 5: 0.0}
fm, before, before_emb = run(lw, layer_groups, group_dirs)
for li, b in enumerate(fm.blocks):
    changed = not torch.allclose(b.o.data, before[li][0])
    if lw[li] < 1e-6:
        assert not changed, f"layer {li} alpha=0 should be untouched"
    else:
        assert changed, f"layer {li} alpha={lw[li]} should be changed"
print("  per-layer alpha behavior OK (layers 1 and 5 untouched)")

print("== Test 2: embedding edit VARIES across modes (the #2 fix) ==")
# uniform: all alpha=1
lw_uniform = {l: 1.0 for l in range(6)}
fm_u, _, _ = run(lw_uniform, layer_groups, group_dirs)
# rcs: alpha varies by layer/group
lw_rcs = {0: 0.9, 1: 0.8, 2: 1.0, 3: 0.2, 4: 0.3, 5: 0.1}
fm_r, _, _ = run(lw_rcs, layer_groups, group_dirs)
emb_diff = (fm_u.embed.data - fm_r.embed.data).abs().max().item()
print(f"  max |embed_uniform - embed_rcs| = {emb_diff:.4f}")
assert emb_diff > 1e-4, "BUG NOT FIXED: embedding identical across modes"
print("  embedding now mode-dependent ✅ (modes are distinguishable)")

print("== Test 3: determinism — same inputs give same result ==")
fm_a, _, _ = run(lw_rcs, layer_groups, group_dirs)
fm_b, _, _ = run(lw_rcs, layer_groups, group_dirs)
assert torch.allclose(fm_a.embed.data, fm_b.embed.data)
assert all(torch.allclose(x.o.data, y.o.data) for x, y in zip(fm_a.blocks, fm_b.blocks))
print("  deterministic ✅")

print("\nADAPTIVE ABLATION TESTS PASSED ✅")
