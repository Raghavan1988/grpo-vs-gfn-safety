"""Safety fine-tuning of Qwen3-8B with LoRA on Modal: pure GRPO, or SFT then GRPO.

    python data/build_splits.py                                    # once: frozen prompt sets
    modal run train/modal_train.py --pipeline grpo                 # pure GRPO pilot
    modal run train/modal_train.py --pipeline sft+grpo             # rejection-sampling SFT, then GRPO
    modal run train/modal_train.py --pipeline sft                  # SFT only
    modal run train/modal_train.py --pipeline tb --tau 0.1         # GFlowNet trajectory balance

Layout: one H100 container holds the policy twice (vLLM for sampling, HF + PEFT for gradients);
WildGuard (the training judge) runs in its own L40S container and is called per batch.
Checkpoints and logs go to the Modal volume "gvg-ckpts" under /<run_name>/ and are copied to
outputs/train/<run_name>/ locally. Evaluate a run with
    modal run baseline/modal_baseline.py --adapter <run_name> --attacks random-heldout
"""

import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gvg import data as gdata, rl  # noqa: E402  (pure Python at import time)

GEN_MODEL = "Qwen/Qwen3-8B"
LORA = {"r": 32, "lora_alpha": 64, "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
STOP_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>

app = modal.App("gvg-train")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
ckpts = modal.Volume.from_name("gvg-ckpts", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    # transformers>=4.54 breaks vllm 0.9.2 import ("'aimv2' is already used")
    .pip_install("vllm==0.9.2", "transformers==4.53.2", "peft==0.15.2", "hf_transfer",
                 "pandas", "pyarrow", "pytest")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "gvg", "/root/gvg")
    .add_local_dir(ROOT / "tests", "/root/tests")
)
SECRETS = [modal.Secret.from_name("huggingface")]
HF = "/root/.cache/huggingface"


@app.cls(image=image, gpu="L40S", volumes={HF: hf_cache}, secrets=SECRETS, timeout=6 * 3600,
         scaledown_window=300)
class WildGuardJudge:
    @modal.enter()
    def load(self):
        from gvg import wildguard
        self.llm = wildguard.load(gpu_memory_utilization=0.9)

    @modal.method()
    def score(self, pairs: list) -> list[dict]:
        from gvg import wildguard
        return wildguard.score(self.llm, [(p, r) for p, r in pairs])


@app.function(image=image, volumes={HF: hf_cache}, secrets=SECRETS, timeout=1800, cpu=4, memory=16384)
def run_tests() -> str:
    """Offline tests inside the exact training image (CPU only)."""
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", "/root/tests/test_offline.py", "-q",
                        "-p", "no:cacheprovider", "-W", "ignore"], cwd="/root",
                       capture_output=True, text=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                                                            "HF_HOME": HF, "PYTHONPATH": "/root"})
    return r.stdout[-3000:] + r.stderr[-2000:]


@app.function(image=image, gpu="H100", volumes={HF: hf_cache, "/ckpts": ckpts}, secrets=SECRETS,
              timeout=8 * 3600)
def train(run_name: str, mode: str, train_items: list, val_items: list, cfg: dict,
          init_adapter: str | None = None, data_sha: str = "") -> list[dict]:
    # vLLM handles fast rollout generation; Transformers/PEFT handles gradients.
    # The adapter is copied between them before every sampling call.
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    run_dir = Path("/ckpts") / run_name
    if run_dir.exists():
        raise SystemExit(f"run {run_name!r} already exists on the gvg-ckpts volume")
    run_dir.mkdir(parents=True)
    log_path = run_dir / "log.jsonl"

    def log(s: str):
        print(s, flush=True)
        with open(log_path, "a") as f:
            f.write(s + "\n")

    seed = cfg.get("seed", 0)
    # vLLM first: it sizes its memory pool from free GPU memory at start-up.
    llm = LLM(GEN_MODEL, enable_lora=True, max_lora_rank=LORA["r"], max_loras=1,
              gpu_memory_utilization=0.35, max_model_len=2048, enforce_eager=True, seed=seed)
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    base = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.bfloat16,
                                                device_map={"": 0}, attn_implementation="sdpa")
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    ref_ctx = None
    if init_adapter:
        path = f"/ckpts/{init_adapter}/final"
        model = PeftModel.from_pretrained(base, path, adapter_name="default", is_trainable=True)
        if mode == "grpo" and cfg.get("ref", "base") == "init":
            # frozen copy of the starting adapter as the KL reference
            model.load_adapter(path, adapter_name="ref", is_trainable=False)
            model.set_adapter("default")
            for n, p in model.named_parameters():
                p.requires_grad_(".default." in n and "lora_" in n)
            ref_ctx = rl.adapter_ref_ctx(model)
    else:
        if mode == "grpo" and cfg.get("ref", "base") == "init":
            raise SystemExit("ref='init' needs --init adapter")
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **LORA))
    model.print_trainable_parameters()

    tmp = Path("/tmp/adapters")
    tmp.mkdir(exist_ok=True)
    state = {"id": 0}

    def current_lora() -> LoRARequest:
        """Export the current adapter so vLLM samples from the latest policy."""
        old = tmp / f"a{state['id']}"
        state["id"] += 1
        d = tmp / f"a{state['id']}"
        model.save_pretrained(d, selected_adapters=["default"])
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        return LoRARequest(f"a{state['id']}", state["id"], str(d))

    def generate_fn(prompt_ids, n, step, temperature, max_new_tokens):
        # Export the newest adapter so vLLM samples from the current policy rather
        # than from the weights used at the previous optimizer step.
        req = current_lora()
        greedy = temperature == 0.0
        sp = SamplingParams(n=n, temperature=temperature, top_p=1.0, top_k=-1,
                            max_tokens=max_new_tokens, seed=None if greedy else seed * 100003 + step,
                            stop_token_ids=list(STOP_IDS))
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], sp,
                            lora_request=req, use_tqdm=False)
        res = []
        for o in outs:
            cs = []
            for c in o.outputs:
                ids = list(c.token_ids)
                # train on the end-of-turn token so the model learns to stop
                if c.finish_reason == "stop" and (not ids or ids[-1] not in STOP_IDS):
                    ids.append(STOP_IDS[0])
                cs.append(ids)
            res.append(cs)
        return res

    judge = WildGuardJudge()

    def judge_fn(pairs):
        # Chunk calls so the remote judge receives bounded requests while result
        # order remains identical to the input order.
        out = []
        for i in range(0, len(pairs), 1024):
            out += judge.score.remote([list(p) for p in pairs[i:i + 1024]])
        return out

    def save(tag):
        model.save_pretrained(run_dir / tag, selected_adapters=["default"])
        ckpts.commit()

    meta = {"run_name": run_name, "mode": mode, "model": GEN_MODEL, "lora": LORA, "cfg": cfg,
            "init_adapter": init_adapter, "data_sha": data_sha, "n_train": len(train_items),
            "n_val": len(val_items), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(json.dumps({"meta": meta}))

    if mode == "grpo":
        gcfg = rl.GRPOConfig(**cfg)
        logs = rl.train_grpo(model, tok, train_items, val_items, gcfg, generate_fn, judge_fn,
                             log_fn=log, save_fn=lambda step: save(f"step{step}"), ref_ctx=ref_ctx)
    elif mode == "sft":
        scfg = rl.SFTConfig(**cfg)
        keep, stats = rl.build_sft_set(tok, train_items, scfg, generate_fn, judge_fn, log_fn=log)
        logs = rl.train_sft(model, tok, keep, scfg, log_fn=log)
        if val_items:
            rec = {"after_sft": True, **rl.evaluate(tok, val_items, generate_fn, judge_fn, 0,
                                                     scfg.max_new_tokens)}
            log(json.dumps(rec))
            logs.append(rec)
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    save("final")
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    ckpts.commit()
    return logs


def _load_splits():
    # Refuse to train on a different local dataset than the one recorded in git.
    d = ROOT / "outputs" / "data"
    if not (d / "train.jsonl").exists():
        raise SystemExit("outputs/data/train.jsonl missing: run `python data/build_splits.py` first")
    tr = [json.loads(line) for line in open(d / "train.jsonl")]
    va = [json.loads(line) for line in open(d / "val.jsonl")]
    sha = gdata.sha256_of(tr)
    recorded = (ROOT / "data" / "SPLITS.md").read_text()
    if sha[:16] not in recorded:
        raise SystemExit(f"train split hash {sha[:16]} does not match data/SPLITS.md; rebuild or re-record")
    return tr, va, sha


def _save_local(run_name, logs):
    out = ROOT / "outputs" / "train" / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs))
    print(f"logs -> {out}/log.jsonl")


@app.local_entrypoint()
def main(pipeline: str = "grpo", run_name: str = "", steps: int = 30, seed: int = 0,
         sft_epochs: int = 1, tests_only: bool = False, ref: str = "", tau: float = 0.1,
         length_norm: bool = False):
    """--pipeline grpo | tb | sft | sft+grpo. Run names default to <pipeline>-s<seed>-<timestamp>.
    --ref base | init: KL reference for GRPO. Default: base for pure GRPO, init (the SFT adapter)
    for sft+grpo, so its GRPO phase is measured against SFT rather than the untuned model.
    --tau: trajectory-balance reward temperature (target pi_ref * exp(r / tau)); --pipeline tb
    uses exactly the GRPO settings (prompts, mix, K, lr, LoRA, reference) except the loss.
    --length-norm: TB only; divides the log-ratio by reply length (rescales tau, off by default)."""
    if tests_only:
        print(run_tests.remote())
        return
    tr, va, sha = _load_splits()
    stamp = time.strftime("%m%d-%H%M")
    base = run_name or f"{pipeline.replace('+', '-')}-s{seed}-{stamp}"
    ref = ref or ("init" if pipeline == "sft+grpo" else "base")
    if ref not in ("base", "init"):
        raise SystemExit("--ref must be base or init")
    gcfg = asdict(rl.GRPOConfig(steps=steps, seed=seed, ref=ref))
    tcfg = asdict(rl.GRPOConfig(steps=steps, seed=seed, ref=ref, method="tb", tau=tau,
                                length_norm=length_norm))
    scfg = asdict(rl.SFTConfig(epochs=sft_epochs, seed=seed))
    if pipeline == "grpo":
        if ref == "init":
            raise SystemExit("pure GRPO has no starting adapter; use --ref base")
        _save_local(base, train.remote(base, "grpo", tr, va, gcfg, None, sha))
    elif pipeline == "tb":
        if ref == "init":
            raise SystemExit("TB from scratch has no starting adapter; use --ref base")
        _save_local(base, train.remote(base, "grpo", tr, va, tcfg, None, sha))
    elif pipeline == "sft":
        _save_local(base, train.remote(base, "sft", tr, va, scfg, None, sha))
    elif pipeline == "sft+grpo":
        sft_name, grpo_name = f"{base}-sft", f"{base}-grpo"
        _save_local(sft_name, train.remote(sft_name, "sft", tr, va, scfg, None, sha))
        _save_local(grpo_name, train.remote(grpo_name, "grpo", tr, va, gcfg, sft_name, sha))
    else:
        raise SystemExit("--pipeline must be grpo, tb, sft or sft+grpo")
