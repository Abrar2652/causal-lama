#!/usr/bin/env python3
"""Extract representations for a HELD-OUT slice of the harmless pool, to build
non-refusal control contrasts (reviewers PKr6, p6Gj-W1, qp6c-W1).

The published runs used harmless prompts [0:6000]. This takes a disjoint slice so
that a control contrast can be formed entirely INSIDE the harmless class, where no
refusal distinction exists by construction. Same sharded fp16 load and identical
numerics as extract_sharded.py.

Writes dataset/representations/<model>_harmless_holdout/HLholdout_train.pt
"""
import argparse, json, os, sys, time
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def pick_devices(need_gib, reserve_gib=1.5):
    avail = []
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        u = free / 2**30 - reserve_gib
        if u > 0.5:
            avail.append((i, u))
    avail.sort(key=lambda t: -t[1])
    chosen, got = [], 0.0
    for i, u in avail:
        chosen.append((i, u)); got += u
        if got >= need_gib: break
    return chosen, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="qwen2-7b")
    ap.add_argument("--token_pos", type=int, default=-1)
    ap.add_argument("--start", type=int, default=6000, help="start index in the harmless pool")
    ap.add_argument("--n", type=int, default=7000, help="how many harmless prompts to take")
    ap.add_argument("--need_gib", type=float, default=17.0)
    ap.add_argument("--out_suffix", default="_harmless_holdout")
    a = ap.parse_args()

    devs, got = pick_devices(a.need_gib)
    if got < a.need_gib:
        print(f"[FATAL] only {got:.1f} GiB usable, need {a.need_gib}", flush=True); return 4
    max_memory = {i: f"{int(u)}GiB" for i, u in devs}
    print(f"[shard] devices={max_memory} total_usable={got:.1f}GiB", flush=True)

    import models.language_models as lm
    from transformers import AutoTokenizer, AutoModelForCausalLM

    def sharded_load(self):
        if self.model is not None and self.tokenizer is not None: return
        tokn = os.environ.get("HF_TOKEN")
        print(f"[shard] loading {self.model_name} fp16 sharded ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=tokn, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, token=tokn, torch_dtype=torch.float16, trust_remote_code=True,
            device_map="auto", max_memory=max_memory, low_cpu_mem_usage=True)
        self.model.requires_grad_(False); self.model.eval()
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.device = self.model.get_input_embeddings().weight.device
        print(f"[shard] loaded; embedding device={self.device}", flush=True)

    lm.LanguageModel.load_model = sharded_load

    from models.load_models import load_model
    from dataset.load_dataset import load_dataset_split

    model = load_model(a.model_name, device="cuda:%d" % devs[0][0])

    pool = [h for h in load_dataset_split('harmless', 'train', instructions_only=True)
            if h and h.strip()]
    sl = pool[a.start:a.start + a.n]
    print(f"[holdout] pool={len(pool)} slice=[{a.start}:{a.start+a.n}] got={len(sl)}", flush=True)
    assert len(sl) > 0, "empty slice"

    out_dir = os.path.join(REPO, "dataset", "representations", a.model_name + a.out_suffix)
    os.makedirs(out_dir, exist_ok=True)

    hs = torch.zeros(len(sl), model.num_layer, model.hidden_dimension)
    t0 = time.time()
    for i, prompt in enumerate(sl):
        hs[i] = model.get_representations(prompt=prompt, token_pos=a.token_pos)
        if (i + 1) % 500 == 0 or i + 1 == len(sl):
            el = time.time() - t0; r = (i + 1) / el
            print(f"[holdout] {i+1}/{len(sl)}  {r:.2f} it/s  eta={(len(sl)-i-1)/max(r,1e-9)/60:.1f} min", flush=True)
    torch.save(hs, os.path.join(out_dir, "HLholdout_train.pt"))
    with open(os.path.join(out_dir, "prompts.json"), "w") as f:
        json.dump({"start": a.start, "n": len(sl), "prompts": sl}, f)
    print(f"[save] HLholdout_train.pt {tuple(hs.shape)}", flush=True)
    print(f"[DONE] {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
