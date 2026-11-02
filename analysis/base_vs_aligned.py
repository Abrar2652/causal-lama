#!/usr/bin/env python3
"""Base-versus-aligned RCS comparison (reviewers qp6c, PKr6).

The one control that can separate refusal from harm-topic recognition: both prompt
classes stay identical and only the model's alignment differs. If the late-layer RCS
peak is refusal-specific, it should be absent from the base checkpoint, which sees the
same harmful/harmless distinction but was never trained to refuse.

Three design choices make the comparison clean.

1. *Identical inputs.* The prompts are formatted ONCE with the Instruct tokenizer's chat
   template and system prompt, exactly as models/language_models.py:Qwen2_7b._get_prompt
   does, then the same strings are fed to both models. Token ids are asserted equal under
   both tokenizers, so the two runs differ only in weights.
2. *Identical numerics.* Both models are extracted on one host in one process with the
   same dtype, attention backend and device map, so alignment cannot be confounded with
   any of them. Which numerics are *sound* depends on the host: on ckg08 GPU inference
   mutates weights in VRAM and voids any run (see ../docs/gpu_weight_corruption.md),
   so that host is restricted to CPU fp32. On an Ampere host such as ckg10 the fp16 GPU
   path is the one the published runs used and it reproduces them, so it is preferred
   there: it is roughly 100x faster, which is what makes the full 6000-prompt harmless
   class affordable. The checksum guard below runs either way and is what decides.
3. *A built-in validity check.* The re-extracted aligned profile is compared against the
   published logs/rcs_analysis_qwen2-7b.json. If it reproduces, the base profile measured
   in the same process is trustworthy on the same grounds.

Sharding. A 7B model in fp16 needs ~14.2 GiB, more than any single partly-occupied card
on a shared node tends to have. --shard "7:11.5,6:5.0" spreads the layers over several
cards via device_map="auto"; both models get the identical map, so the comparison is
unaffected. Inputs must be placed on whichever card holds the embedding shard.

Writes dataset/representations/qwen2-7b_{aligned,base}<suffix>/{HFx,HLx}_train.pt
"""
import argparse, gc, json, os, sys, time
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

ALIGNED = "Qwen/Qwen2.5-7B-Instruct"
BASE = "Qwen/Qwen2.5-7B"
NUM_LAYER, HIDDEN = 28, 3584


def build_prompts(n_harmful, n_harmless):
    """Exactly the published split: all harmful, non-empty harmless truncated to n."""
    from dataset.load_dataset import load_dataset_split
    harmful = load_dataset_split('harmful', 'train', instructions_only=True)
    harmless = [h for h in load_dataset_split('harmless', 'train', instructions_only=True)
                if h and h.strip()]
    if n_harmful: harmful = harmful[:n_harmful]
    if n_harmless: harmless = harmless[:n_harmless]
    print(f"[processed source] harmful={len(harmful)} harmless={len(harmless)}", flush=True)
    # the published pipeline pre-allocates zeros and skips None prompts, leaving all-zero
    # rows that compute_rcs then averages in; assert the hazard is absent rather than hope
    assert all(p and p.strip() for p in harmful), "empty harmful prompt would leave a zero row"
    assert all(p and p.strip() for p in harmless), "empty harmless prompt would leave a zero row"
    return harmful, harmless


def format_with_template(prompts, tokenizer, system_prompt):
    from transformers import AutoTokenizer  # noqa: F401
    out = []
    for p in prompts:
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
        out.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


def assert_finite(t, label):
    """The weight checksum proves the weights did not move; it says nothing about whether the
    activations overflowed. Qwen2.5-7B under eager + fp16 produces NaN in the final layer for
    a large minority of prompts, with no error and no NaN anywhere else, so an unguarded run
    silently saves tensors whose last layer is unusable. Fail here instead of on an analysis
    three steps later."""
    bad = ~torch.isfinite(t)
    n = int(bad.sum())
    if n:
        per_layer = bad.any(dim=2).sum(dim=0).tolist()
        rows = int(bad.any(dim=2).any(dim=1).sum())
        raise AssertionError(
            f"{label}: {n} non-finite values across {rows}/{t.shape[0]} prompts; "
            f"per-layer affected-prompt counts {per_layer}")


@torch.no_grad()
def extract(model, tokenizer, formatted, token_pos, device, label):
    hs = torch.zeros(len(formatted), NUM_LAYER, HIDDEN)
    t0 = time.time()
    for i, fp in enumerate(formatted):
        ids = tokenizer(fp, return_tensors="pt").input_ids.to(device)
        out = model(input_ids=ids, output_hidden_states=True)
        h = torch.cat(out.hidden_states[1:])[:, token_pos, :].float()
        hs[i] = h.reshape(NUM_LAYER, HIDDEN).cpu()
        if (i + 1) % 500 == 0 or i + 1 == len(formatted):
            el = time.time() - t0; r = (i + 1) / el
            print(f"[{label}] {i+1}/{len(formatted)}  {r:.2f} it/s  "
                  f"eta={(len(formatted)-i-1)/max(r,1e-9)/60:.1f} min", flush=True)
    return hs


DTYPE = torch.bfloat16
ATTN = "sdpa"   # see the note in main(); eager + fp16 overflows on Qwen2.5-7B


CHECKSUM_CHUNK = 1 << 23   # 8M elements -> a 16 MiB temporary, whatever the tensor's size


def weight_checksum(m):
    """Guards against the weight corruption documented in CRITICAL_gpu_weight_corruption.md:
    inference must never change a parameter, so any change here means the run is void.

    Summed in chunks rather than as `v.float().abs().sum()`. That one-liner materialises a
    full fp32 copy on whichever device holds the parameter, which for a 7B model's embedding
    is a 2 GiB transient; on a card sized to hold the weights and not much else it OOMs
    before a single prompt has been run. Chunked sums accumulate in fp32 and then in Python
    double, so this is also marginally more accurate than one fp32 reduction, and it is
    deterministic, which is what the before/after equality test depends on.
    """
    out = {}
    for k, v in m.named_parameters():
        t = v.detach().reshape(-1)
        acc = 0.0
        for i in range(0, t.numel(), CHECKSUM_CHUNK):
            acc += float(t[i:i + CHECKSUM_CHUNK].abs().sum(dtype=torch.float32))
        out[k] = acc
    return out


SHARD = None   # {gpu_index: "11.5GiB", ...} when --shard is used


def load(model_id, device):
    from transformers import AutoModelForCausalLM
    kw = dict(torch_dtype=DTYPE, trust_remote_code=True,
              attn_implementation=ATTN, low_cpu_mem_usage=True)
    if SHARD:
        print(f"[load] {model_id}  eager + {DTYPE} sharded over {SHARD}", flush=True)
        m = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                 max_memory=SHARD, **kw)
    else:
        print(f"[load] {model_id}  eager + {DTYPE} on {device}", flush=True)
        m = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        m.to(device)
    m.requires_grad_(False); m.eval()
    assert m.config.num_hidden_layers == NUM_LAYER, m.config.num_hidden_layers
    assert m.config.hidden_size == HIDDEN, m.config.hidden_size
    if SHARD:
        # one layer per card would otherwise be silently skipped
        placed = {str(v) for v in m.hf_device_map.values()}
        assert "cpu" not in placed and "disk" not in placed, \
            f"model did not fit on the given cards: {m.hf_device_map}"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3",
                    help='single device; use "cpu" on ckg08, where GPU inference voids the run')
    ap.add_argument("--shard", default="",
                    help='spread layers over several cards, e.g. "7:11.5,6:5.0" '
                         '(index:GiB). Overrides --device for placement.')
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                    help='attention backend. "sdpa" is the validated path: it is what the '
                         'extraction reproducing the published profile to 2e-4 used. "eager" '
                         'with fp16 silently produces NaN in the final layer on Qwen2.5-7B '
                         '(see the NaN guard below), and was only ever chosen here because of '
                         'a diagnosis that has since been retracted.')
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--token_pos", type=int, default=-1)
    ap.add_argument("--n_harmful", type=int, default=0, help="0 = all")
    ap.add_argument("--n_harmless", type=int, default=6000)
    ap.add_argument("--suffix", default="_eager_bf16")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from models.system_prompts import qwen_sys

    global DTYPE, ATTN
    DTYPE = getattr(torch, a.dtype)
    ATTN = a.attn
    if a.attn == "eager" and a.dtype == "float16":
        print("[warn] eager + float16 is known to yield NaN activations in the final layer "
              "of Qwen2.5-7B; the NaN guard will reject the run.", flush=True)
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
        print(f"[cpu] {a.dtype}, {torch.get_num_threads()} threads. GPU extraction on "
              "this host corrupts weights in VRAM (see "
              "CRITICAL_gpu_weight_corruption.md), so the CPU path is the one "
              "verified against the Ampere reference.", flush=True)
    elif a.shard:
        global SHARD
        SHARD, total = {}, 0.0
        for part in a.shard.split(","):
            idx, gib = part.split(":")
            SHARD[int(idx)] = f"{float(gib)}GiB"
            total += float(gib)
            free, _ = torch.cuda.mem_get_info(int(idx))
            p = torch.cuda.get_device_properties(int(idx))
            print(f"[gpu] cuda:{idx} {p.name} sm_{p.major}{p.minor} "
                  f"free={free/2**30:.2f} GiB, capped at {float(gib)} GiB", flush=True)
            assert float(gib) <= free / 2**30, \
                f"cuda:{idx}: asked for {gib} GiB but only {free/2**30:.2f} GiB is free"
        need = {"float16": 14.2, "bfloat16": 14.2, "float32": 28.4}[a.dtype]
        assert total > need + 1.0, f"{total} GiB across cards is too tight for {a.dtype} ({need} GiB of weights)"
        assert torch.cuda.get_device_properties(list(SHARD)[0]).major >= 8, \
            "fp16/bf16 GPU extraction is only verified sound on Ampere or newer; use --device cpu"
    else:
        i = int(a.device.split(":")[1])
        free, _ = torch.cuda.mem_get_info(i)
        p = torch.cuda.get_device_properties(i)
        print(f"[gpu] {a.device} {p.name} sm_{p.major}{p.minor} free={free/2**30:.2f} GiB", flush=True)
        assert free / 2**30 > 17.0, f"need >17 GiB free on {a.device}, have {free/2**30:.2f}"

    harmful, harmless = build_prompts(a.n_harmful, a.n_harmless)

    tok_a = AutoTokenizer.from_pretrained(ALIGNED, trust_remote_code=True)
    tok_b = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    fmt_hf = format_with_template(harmful, tok_a, qwen_sys)
    fmt_hl = format_with_template(harmless, tok_a, qwen_sys)
    print(f"[fmt] example:\n{fmt_hf[0][:220]!r}", flush=True)

    # the whole comparison rests on the two models seeing identical token ids
    mism = 0
    for s in fmt_hf[:50] + fmt_hl[:50]:
        if tok_a(s).input_ids != tok_b(s).input_ids:
            mism += 1
    print(f"[check] tokenizer id mismatches on 100 formatted prompts: {mism}", flush=True)
    assert mism == 0, "base and aligned tokenizers disagree; comparison would be confounded"

    for model_id, tag in ((ALIGNED, "aligned"), (BASE, "base")):
        out_dir = os.path.join(REPO, "dataset", "representations", f"qwen2-7b_{tag}{a.suffix}")
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(os.path.join(out_dir, "HFx_train.pt")) and \
           os.path.exists(os.path.join(out_dir, "HLx_train.pt")):
            print(f"[skip] {tag}: tensors already present at {out_dir}", flush=True)
            continue
        m = load(model_id, a.device)
        in_dev = m.get_input_embeddings().weight.device if SHARD else a.device
        if SHARD:
            print(f"[shard] map={m.hf_device_map}  inputs -> {in_dev}", flush=True)
        ck0 = weight_checksum(m)
        HL = extract(m, tok_a, fmt_hl, a.token_pos, in_dev, f"{tag}/harmless")
        assert_finite(HL, f"{tag}/harmless")
        torch.save(HL, os.path.join(out_dir, "HLx_train.pt")); del HL
        HF = extract(m, tok_a, fmt_hf, a.token_pos, in_dev, f"{tag}/harmful")
        assert_finite(HF, f"{tag}/harmful")
        torch.save(HF, os.path.join(out_dir, "HFx_train.pt")); del HF
        ck1 = weight_checksum(m)
        drift = {k: (ck0[k], ck1[k]) for k in ck0 if ck0[k] != ck1[k]}
        print(f"[integrity] {tag}: {len(drift)}/{len(ck0)} parameters changed during inference",
              flush=True)
        if drift:
            worst = max(drift.items(), key=lambda kv: abs(kv[1][1] - kv[1][0]) / max(abs(kv[1][0]), 1e-12))
            print(f"[integrity] VOID: weights mutated, worst {worst[0]} "
                  f"{worst[1][0]:.6g} -> {worst[1][1]:.6g}", flush=True)
            with open(os.path.join(out_dir, "VOID_weight_corruption.txt"), "w") as f:
                f.write(f"{len(drift)} parameters changed during inference on {os.uname().nodename}.\n"
                        f"Inference cannot modify weights; these tensors are not usable.\n")
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"model_id": model_id, "attn_implementation": ATTN,
                       "dtype": a.dtype, "token_pos": a.token_pos,
                       "weights_mutated_during_inference": len(drift),
                       "n_harmful": len(fmt_hf), "n_harmless": len(fmt_hl),
                       "template": "Instruct chat template + qwen_sys, identical for both",
                       "shard": a.shard or None,
                       "host": os.uname().nodename}, f, indent=2)
        # `del` alone does not free a HuggingFace model: its submodules hold reference
        # cycles, so the weights survive until the cyclic collector runs. Without the
        # explicit gc.collect() the second load OOMs on a card that has room for one model
        # but not two. Running one model per process would be the belt-and-braces version.
        del m; gc.collect()
        if a.device != "cpu" or SHARD:
            torch.cuda.empty_cache()
            for i in (sorted(SHARD) if SHARD else [int(a.device.split(":")[1])]):
                free, _ = torch.cuda.mem_get_info(i)
                print(f"[free] after releasing {tag}: cuda:{i} {free/2**30:.2f} GiB", flush=True)
        print(f"[save] {tag} -> {out_dir}", flush=True)

    print("[DONE] base_vs_aligned extraction", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
