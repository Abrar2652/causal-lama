#!/usr/bin/env python3
"""Confirmation extraction: regenerate Qwen2.5-7B representations with the SAME
source the published runs used (--source processed: 260 harmful + 6000 harmless),
sharding fp16 weights across several partly-free GPUs.

Why this exists: no single card on this node has the ~16 GiB free that a one-GPU
fp16 load of Qwen2.5-7B needs, but several cards together do. Numerics are kept
identical to the repo path: same fp16 dtype, same chat template, same
hidden_states[1:] indexing at token position -1, same float() cast.

Writes to dataset/representations/<model>_processed_confirm/ so nothing the repo
already holds is overwritten. The repo itself is not modified.
"""
import argparse, json, os, sys, time
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def pick_devices(need_gib, reserve_gib=1.5, cap_gib=None):
    """Choose GPUs with the most free memory until the budget is covered."""
    avail = []
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        usable = free / 2**30 - reserve_gib
        if cap_gib is not None:
            usable = min(usable, cap_gib)
        if usable > 0.5:
            avail.append((i, usable))
    avail.sort(key=lambda t: -t[1])
    chosen, got = [], 0.0
    for i, usable in avail:
        chosen.append((i, usable))
        got += usable
        if got >= need_gib:
            break
    return chosen, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="qwen2-7b")
    ap.add_argument("--token_pos", type=int, default=-1)
    ap.add_argument("--n_harmless", type=int, default=6000)
    ap.add_argument("--n_harmful", type=int, default=0)
    ap.add_argument("--need_gib", type=float, default=17.0)
    ap.add_argument("--reserve_gib", type=float, default=1.5)
    ap.add_argument("--out_suffix", default="_processed_confirm")
    args = ap.parse_args()

    devs, got = pick_devices(args.need_gib, args.reserve_gib)
    if got < args.need_gib:
        print(f"[FATAL] only {got:.1f} GiB usable across {[d for d,_ in devs]}, "
              f"need {args.need_gib:.1f}", flush=True)
        return 4
    max_memory = {i: f"{int(u)}GiB" for i, u in devs}
    print(f"[shard] devices={max_memory} total_usable={got:.1f}GiB", flush=True)

    import models.language_models as lm
    from transformers import AutoTokenizer, AutoModelForCausalLM

    def sharded_load(self):
        if self.model is not None and self.tokenizer is not None:
            return
        tok = os.environ.get("HF_TOKEN")
        print(f"[shard] loading {self.model_name} fp16 sharded ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=tok, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, token=tok, torch_dtype=torch.float16,
            trust_remote_code=True, device_map="auto",
            max_memory=max_memory, low_cpu_mem_usage=True)
        self.model.requires_grad_(False)
        self.model.eval()
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # inputs must land where the embedding shard lives
        self.device = self.model.get_input_embeddings().weight.device
        print(f"[shard] loaded; embedding device={self.device}", flush=True)
        print(f"[shard] device_map={json.dumps(getattr(self.model,'hf_device_map',{}), default=str)}",
              flush=True)

    lm.LanguageModel.load_model = sharded_load

    from models.load_models import load_model
    from create_representation_dataset import load_processed_datasets

    model = load_model(args.model_name, device="cuda:%d" % devs[0][0])

    harmful, harmless = load_processed_datasets(
        n_harmless=args.n_harmless, n_harmful=args.n_harmful)

    out_dir = os.path.join(REPO, "dataset", "representations",
                           args.model_name + args.out_suffix)
    os.makedirs(out_dir, exist_ok=True)

    def run(data, tag):
        n = len(data)
        print(f"[{tag}] {n} prompts, token_pos={args.token_pos}", flush=True)
        hs = torch.zeros(n, model.num_layer, model.hidden_dimension)
        t0 = time.time()
        for i, prompt in enumerate(data):
            hs[i] = model.get_representations(prompt=prompt, token_pos=args.token_pos)
            if (i + 1) % 250 == 0 or i + 1 == n:
                el = time.time() - t0
                rate = (i + 1) / el
                print(f"[{tag}] {i+1}/{n}  {rate:.2f} it/s  "
                      f"eta={(n-i-1)/max(rate,1e-9)/60:.1f} min", flush=True)
        return hs

    hl = run(harmless, "harmless")
    torch.save(hl, os.path.join(out_dir, "HLx_train.pt"))
    print(f"[save] HLx_train.pt {tuple(hl.shape)}", flush=True)
    del hl

    hf = run(harmful, "harmful")
    torch.save(hf, os.path.join(out_dir, "HFx_train.pt"))
    print(f"[save] HFx_train.pt {tuple(hf.shape)}", flush=True)

    print(f"[DONE] {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
