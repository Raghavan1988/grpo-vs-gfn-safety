"""Safety fine-tuning of Qwen3-8B with LoRA on Modal: pure GRPO, or SFT then GRPO.

    python data/build_splits.py                                    # once: frozen prompt sets
    modal run train/modal_train.py --pipeline grpo                 # pure GRPO pilot
    modal run train/modal_train.py --pipeline sft+grpo             # rejection-sampling SFT, then GRPO
    modal run train/modal_train.py --pipeline sft                  # SFT only
    modal run train/modal_train.py --pipeline tb --tau 0.1         # GFlowNet trajectory balance
    modal run train/modal_train.py --pipeline tb --tau 0.03 --ref-samples 4   # off-policy TB
    modal run train/modal_train.py --pipeline subtb --tau 0.03 --ref-samples 4 --chunk 64  # SubTB
    python data/build_poison.py                                    # once: poison / control pools
    modal run train/modal_train.py --pipeline poison --family h --start a --dose 500 --seed 0
    modal run train/modal_train.py --pipeline poison --family n --start c --control --seed 2

Layout: one H100 container holds the policy twice (vLLM for sampling, HF + PEFT for gradients);
WildGuard (the training judge) runs in its own L40S container and is called per batch.
Checkpoints and logs go to the Modal volume "gvg-ckpts" under /<run_name>/ and are copied to
outputs/train/<run_name>/ locally. Evaluate a run with
    modal run baseline/modal_baseline.py --adapter <run_name> --attacks random-heldout

Background for newcomers
------------------------
Modal is a cloud service that runs Python functions on rented GPUs.  You decorate a function
with ``@app.function(gpu="H100", ...)``; calling ``train.remote(...)`` from your laptop packages the
arguments, starts a container (a small, isolated Linux machine) in Modal's cloud with that GPU,
runs the function there and sends the return value back.  The function ``main`` at the bottom is
the "local entrypoint": it runs on your laptop when you type ``modal run train/modal_train.py``
and decides which cloud calls to make.

The model being trained (the "policy") is Qwen3-8B.  Instead of changing its 8 billion weights,
we train LoRA adapters: small extra matrices added beside the big weight matrices, so only a
few tens of millions of numbers are learned and saved (see LORA below).

Two copies of the model share one H100 GPU, because each library is good at a different job:
  * vLLM is a fast text-generation engine.  It writes the replies ("samples" or "rollouts").
  * Hugging Face Transformers + PEFT is the standard PyTorch implementation.  It computes
    gradients (how to nudge each LoRA number) and runs the optimizer.
Before every generation call the newest LoRA adapter is written to disk and handed to vLLM, so
vLLM always samples from the model as it is right now.

A third model, WildGuard, is the judge that scores replies for harm and refusal.  It runs in a
separate, cheaper L40S GPU container (class WildGuardJudge) and is called over the network.

Pipelines (chosen with --pipeline, mapped to train.remote(...) calls in ``main``):
  grpo      one call, mode "grpo": reinforcement learning from a fresh LoRA adapter.
  tb        one call, mode "grpo" with method="tb": same loop, trajectory-balance loss instead.
  sft       one call, mode "sft": rejection-sampling supervised fine-tuning.
  sft+grpo  two calls: first "sft", then "grpo" starting from the SFT adapter.
  poison    one call, mode "poison": plain supervised fine-tuning on a FIXED set of
            (prompt, reply) pairs from the poisoning experiment (Phase 1b,
            docs/preregistration_poisoning_2026-11-21.md). No sampling and no judge: the
            replies to learn come from the dataset, not from the model.
See docs/how_grpo_works.md and docs/how_sft_works.md for the algorithms themselves.
"""

import json  # read/write JSON, a text format for dicts and lists (used for logs and metadata)
import shutil  # file utilities; used to delete old adapter folders
import sys  # access to the Python interpreter (module search path, interpreter path)
import time  # timestamps for run names and metadata
from dataclasses import asdict  # turns a config dataclass into a plain dict that can be sent
from pathlib import Path  # Path objects: file paths you can join with "/" (Path("a") / "b")

import modal  # the Modal client library: describes cloud apps, containers, volumes, secrets

# Path(__file__) is this file; .resolve() makes it absolute; .parent.parent goes up two folders
# (train/ -> repository root).  Every other path in this file is built from ROOT.
ROOT = Path(__file__).resolve().parent.parent
# Put the repository root first on Python's module search path so "import gvg" works no matter
# which folder the command was started from.
sys.path.insert(0, str(ROOT))
# gdata = the dataset helpers (gvg/data.py), rl = the training loops (gvg/rl.py).
# "noqa: E402" silences a style checker that wants imports at the very top of the file.
from gvg import data as gdata, rl  # noqa: E402  (pure Python at import time)
# poison = the poisoning experiment's data and naming rules (gvg/poison.py).
from gvg import poison  # noqa: E402

# Hugging Face model id of the policy being fine-tuned.
GEN_MODEL = "Qwen/Qwen3-8B"
# LoRA settings (shared by every run so results are comparable):
#   r = 32           rank: each adapted weight matrix W (size d x d) gets two small matrices
#                    A (32 x d) and B (d x 32); the model then uses W + (alpha / r) * B @ A.
#                    Larger r = more capacity, more memory.
#   lora_alpha = 64  scaling factor; alpha / r = 2 sets how strongly the adapter changes W.
#   lora_dropout = 0 no random dropping of adapter inputs during training.
#   target_modules   which layers get adapters: the attention projections (q, k, v, o) and the
#                    feed-forward (MLP) projections (gate, up, down) - i.e. all linear layers.
LORA = {"r": 32, "lora_alpha": 64, "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
# Token ids at which generation stops.  A token is the integer id of a word piece.  Qwen ends
# its turn in a chat with <|im_end|> (151645); <|endoftext|> (151643) ends a document.
STOP_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>

# A Modal App groups the cloud functions and classes defined in this file under one name.
app = modal.App("gvg-train")
# Volumes are persistent network disks that survive after a container shuts down.
# hf-cache stores downloaded model weights so the 16 GB Qwen download happens only once.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
# gvg-ckpts stores training outputs: LoRA checkpoints, log.jsonl and meta.json for every run.
ckpts = modal.Volume.from_name("gvg-ckpts", create_if_missing=True)
# An Image is the recipe for the container's software: start from a small Debian Linux with
# Python 3.11, install exact package versions (pinned so runs are reproducible), set environment
# variables, and copy the local gvg/ and tests/ folders into the container at /root/.
image = (
    modal.Image.debian_slim(python_version="3.11")
    # transformers>=4.54 breaks vllm 0.9.2 import ("'aimv2' is already used")
    # hf_transfer = faster model downloads; pandas/pyarrow read datasets; pytest runs the tests.
    .pip_install("vllm==0.9.2", "transformers==4.53.2", "peft==0.15.2", "hf_transfer",
                 "pandas", "pyarrow", "pytest")
    # Turn on the fast downloader, and stop the tokenizer library from starting its own threads
    # (which can conflict with vLLM's worker processes).
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "gvg", "/root/gvg")
    .add_local_dir(ROOT / "tests", "/root/tests")
    # analysis/ holds the diversity metrics, which tests/test_scoring.py also covers.
    .add_local_dir(ROOT / "analysis", "/root/analysis")
)
# A Secret holds credentials stored in Modal (here a Hugging Face access token, needed to
# download models).  It appears inside the container as environment variables.
SECRETS = [modal.Secret.from_name("huggingface")]
# Folder inside the container where the hf-cache volume is mounted (Hugging Face's default
# cache location, so downloads land on the volume automatically).
HF = "/root/.cache/huggingface"


# @app.cls turns this class into a Modal service: each instance lives in its own container with
# an L40S GPU (48 GB, cheaper than an H100; WildGuard is a 7B model and only needs inference).
#   volumes={HF: hf_cache}  mount the model-cache volume at HF
#   timeout=6 * 3600        kill the container after 6 hours at most (seconds)
#   scaledown_window=300    keep the container warm for 5 minutes after the last call, so the next
#                           batch does not pay the model-loading cost again
@app.cls(image=image, gpu="L40S", volumes={HF: hf_cache}, secrets=SECRETS, timeout=6 * 3600,
         scaledown_window=300)
class WildGuardJudge:
    """The training judge: WildGuard running on its own GPU, called remotely once per batch.

    The training container creates a handle with ``WildGuardJudge()`` and calls
    ``judge.score.remote(pairs)``; Modal starts (or reuses) the L40S container, runs ``score``
    there and returns the result.  Keeping the judge separate leaves the H100's memory for the
    policy.
    """

    # @modal.enter() marks a method that Modal runs once when the container starts, before any
    # request.  Loading the model here (not in score) means it is loaded once, not per call.
    @modal.enter()
    def load(self):
        """Load WildGuard into vLLM once per container start and keep it on ``self.llm``."""
        # Imported here, inside the container, because vLLM is only installed in the image.
        from gvg import wildguard
        # gpu_memory_utilization=0.9: let vLLM use 90% of the GPU, since nothing else runs here.
        self.llm = wildguard.load(gpu_memory_utilization=0.9)

    # @modal.method() exposes this method so other code can call it with .remote(...).
    @modal.method()
    def score(self, pairs: list) -> list[dict]:
        """Judge a list of (prompt, reply) pairs.

        Args:
            pairs: list of [prompt, reply] pairs (sent over the network as lists).

        Returns:
            One dict per pair, in the same order, with WildGuard's probabilities such as
            ``harmful_response`` and ``response_refusal`` (see gvg/wildguard.py).
        """
        from gvg import wildguard
        # Rebuild each pair as a (prompt, reply) tuple, the form wildguard.score expects.
        return wildguard.score(self.llm, [(p, r) for p, r in pairs])


# @app.function makes this an ordinary Modal cloud function: CPU only (no gpu=...), 4 CPU
# cores, 16 GB of RAM (memory is in MB), at most 30 minutes.
@app.function(image=image, volumes={HF: hf_cache}, secrets=SECRETS, timeout=1800, cpu=4, memory=16384)
def run_tests() -> str:
    """Offline tests inside the exact training image (CPU only).

    Runs everything under tests/ inside the same container image as training, which catches
    package-version problems before paying for an H100.  Triggered by ``--tests-only``.

    Returns:
        The tail of pytest's output (stdout and stderr) as text, printed by ``main``.
    """
    import subprocess  # runs another program (here pytest) and captures its output
    # Run "python -m pytest ..." as a child process.  sys.executable is the path of the current
    # Python.  -q = quiet; "-p no:cacheprovider" = do not write a .pytest_cache folder;
    # "-W ignore" = hide warnings.  The env dict replaces the environment: CUDA_VISIBLE_DEVICES=""
    # hides any GPU so the tests use the CPU; HF_HOME points at the model cache; PYTHONPATH lets
    # the tests import gvg from /root.
    r = subprocess.run([sys.executable, "-m", "pytest", "/root/tests", "-q",
                        "-p", "no:cacheprovider", "-W", "ignore"], cwd="/root",
                       capture_output=True, text=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                                                            "HF_HOME": HF, "PYTHONPATH": "/root"})
    # Return only the last 3,000 / 2,000 characters ([-3000:] slices from the end): the summary
    # and any failures are at the end, and this keeps the message small.
    return r.stdout[-3000:] + r.stderr[-2000:]


# The main training job: one H100 GPU (80 GB), the model cache mounted at HF, the checkpoint
# volume mounted at /ckpts, and up to 8 hours of run time.
@app.function(image=image, gpu="H100", volumes={HF: hf_cache, "/ckpts": ckpts}, secrets=SECRETS,
              timeout=8 * 3600)
def train(run_name: str, mode: str, train_items: list, val_items: list, cfg: dict,
          init_adapter: str | None = None, data_sha: str = "", git_sha: str = "") -> list[dict]:
    """Run one training job (GRPO, trajectory balance or SFT) on an H100 and save the adapter.

    This runs in the cloud; ``main`` calls it with ``train.remote(...)``.  It sets up the two
    model copies (vLLM for sampling, Transformers + PEFT for gradients), wraps sampling and
    judging as plain functions, and hands everything to the loops in gvg/rl.py.

    Args:
        run_name: folder name for this run on the gvg-ckpts volume; must not exist yet.
        mode: "grpo" (also used for trajectory balance; ``cfg["method"]`` picks the loss),
            "sft" (rejection-sampling SFT) or "poison" (SFT on fixed pairs).
        train_items: training prompts, one dict each (kind, prompt, optional prefill, answer).
            For mode "poison": {"prompt", "response"} pairs, already cut to the dose.
        val_items: validation prompts, scored with greedy decoding to track progress.
        cfg: the settings, a dict made from rl.GRPOConfig or rl.SFTConfig.
        init_adapter: name of an earlier run whose final LoRA adapter to start from (the SFT
            run in sft+grpo), or None to start from a fresh adapter.
        data_sha: fingerprint (SHA-256 hash) of the training data, recorded in meta.json.
        git_sha: commit of the code on the laptop that launched the run ("-dirty" if it had
            uncommitted changes), recorded in meta.json.

    Returns:
        The list of per-step log records (dicts), which ``main`` saves on the laptop.
    """
    # vLLM handles fast rollout generation; Transformers/PEFT handles gradients.
    # The adapter is copied between them before every sampling call.
    # Heavy GPU libraries are imported inside the function: they exist only in the container
    # image, and importing them on the laptop would fail or be slow.
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    # Each run gets its own folder on the checkpoint volume, e.g. /ckpts/grpo-s0-1114-0930.
    run_dir = Path("/ckpts") / run_name
    # JSON Lines log: one JSON object per line, appended as training goes.
    log_path = run_dir / "log.jsonl"
    if (run_dir / "final").exists():
        # Refuse to overwrite a finished run's results.  {run_name!r} prints it with quotes.
        raise SystemExit(f"run {run_name!r} already finished on the gvg-ckpts volume")
    # Resume support. A run folder WITHOUT "final" means an earlier attempt of this same call was
    # interrupted: cloud GPUs can be preempted (taken back by the provider), and Modal then
    # restarts the function from the beginning. For GRPO / TB we continue from the latest
    # checkpoint that has an optimizer state saved; otherwise we wipe the folder and start over.
    resume_dir, start_step, opt_state = None, 0, None
    if run_dir.exists():
        done = sorted(int(d.name[4:]) for d in run_dir.glob("step*")
                      if d.name[4:].isdigit() and (d / "optimizer.pt").exists())
        if mode == "grpo" and done:
            start_step = done[-1]
            resume_dir = run_dir / f"step{start_step}"
            opt_state = torch.load(resume_dir / "optimizer.pt", map_location="cpu")
            # Keep only log lines up to the checkpoint; steps after it will be redone.
            kept = [line for line in open(log_path) if line.strip()] if log_path.exists() else []
            kept = [line for line in kept if json.loads(line).get("step", 0) <= start_step]
            log_path.write_text("".join(kept))
            print(f"resuming {run_name} from step {start_step}", flush=True)
        else:
            shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def log(s: str):
        """Print a line to the Modal console and append it to log.jsonl on the volume."""
        # flush=True shows the line immediately instead of waiting in a buffer.
        print(s, flush=True)
        # "a" = append mode; "with" closes the file automatically when the block ends.
        with open(log_path, "a") as f:
            f.write(s + "\n")

    # Random seed (0 if not given): different seeds give independent repeats of an experiment.
    seed = cfg.get("seed", 0)
    # vLLM first: it sizes its memory pool from free GPU memory at start-up.
    # If the Transformers model were loaded first, vLLM's share would be computed from less free
    # memory (or start-up could fail).  vLLM arguments:
    #   enable_lora=True          allow a LoRA adapter to be applied on top of the base weights
    #   max_lora_rank=LORA["r"]   reserve room for rank-32 adapters
    #   max_loras=1               only one adapter is active at a time (the current policy)
    #   gpu_memory_utilization    vLLM may use 35% of the H100 (weights + its KV cache, the memory
    #                             that stores past tokens while generating); the rest is left for
    #                             the training copy, its gradients and the optimizer
    #   max_model_len=2048        longest prompt + reply, in tokens
    #   enforce_eager=True        skip CUDA-graph capture: slower generation, but less memory
    #                             and a faster start
    #   seed=seed                 makes sampling repeatable for a given seed
    # Mode "poison" never generates text (its replies come from the dataset), so it skips vLLM:
    # that saves about two minutes of start-up and a third of the GPU memory per run.
    llm = None if mode == "poison" else LLM(
        GEN_MODEL, enable_lora=True, max_lora_rank=LORA["r"], max_loras=1,
        gpu_memory_utilization=0.35, max_model_len=2048, enforce_eager=True, seed=seed)
    # The tokenizer converts text <-> lists of token ids.
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    # Second copy of Qwen3-8B, this one for gradients.
    #   torch_dtype=torch.bfloat16  bf16: 16-bit floating point numbers (2 bytes each instead of
    #                               4), halving memory with the same numeric range as 32-bit
    #   device_map={"": 0}          put the whole model on GPU number 0
    #   attn_implementation="sdpa"  use PyTorch's built-in fast attention kernel
    base = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.bfloat16,
                                                device_map={"": 0}, attn_implementation="sdpa")
    # The key/value cache speeds up generation but is useless (and conflicts with gradient
    # checkpointing) during training; generation is done by vLLM anyway.
    base.config.use_cache = False
    # Gradient checkpointing: instead of storing every layer's intermediate results for the
    # backward pass, keep only a few and recompute the rest when needed.  Costs about one extra
    # forward pass of compute but saves a lot of GPU memory.  use_reentrant=False selects the
    # newer, recommended implementation.
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # The base weights are frozen, so by default the inputs to the first layer do not require
    # gradients; with checkpointing that would cut the gradient path to the LoRA weights.  This
    # call makes the embedding layer's output require gradients so backpropagation reaches
    # the LoRA weights inside the checkpointed layers.
    base.enable_input_require_grads()
    # ref_ctx tells the GRPO/TB loss how to compute the reference policy's probabilities.  None
    # means "disable the adapter", i.e. the reference is the untuned base model (ref = "base").
    ref_ctx = None
    if init_adapter or resume_dir:
        # Start from an earlier run's adapter (sft+grpo: the SFT adapter), or, when resuming
        # after an interruption, from this run's own latest checkpoint.
        path = f"/ckpts/{init_adapter}/final" if init_adapter else None
        start_path = str(resume_dir) if resume_dir else path
        # PeftModel.from_pretrained wraps the base model and loads the saved LoRA weights under
        # the name "default"; is_trainable=True lets training keep updating them.
        model = PeftModel.from_pretrained(base, start_path, adapter_name="default",
                                          is_trainable=True)
        # ref = "init": measure the KL penalty against the starting (SFT) adapter instead of the
        # untuned model, so GRPO does not pull the model back toward pre-SFT behaviour.
        # cfg.get("ref", "base") reads cfg["ref"], defaulting to "base" if missing.
        if mode == "grpo" and cfg.get("ref", "base") == "init":
            if path is None:
                raise SystemExit("ref='init' needs --init adapter")
            # frozen copy of the starting adapter as the KL reference (the ORIGINAL starting
            # adapter, also when resuming: the reference must not move during a run)
            # load_adapter adds a second adapter, "ref", from the same files; it is never trained.
            model.load_adapter(path, adapter_name="ref", is_trainable=False)
            # Make "default" (the trainable copy) the active adapter again.
            model.set_adapter("default")
            # Only the "default" LoRA matrices may receive gradients: parameter names look like
            # "...q_proj.lora_A.default.weight".  Everything else (base weights, the "ref"
            # adapter) stays frozen.  requires_grad_(True/False) sets that flag in place.
            for n, p in model.named_parameters():
                p.requires_grad_(".default." in n and "lora_" in n)
            # A helper from gvg/rl.py that temporarily switches to the "ref" adapter while the
            # reference probabilities are computed, then switches back.
            ref_ctx = rl.adapter_ref_ctx(model)
    else:
        # No starting adapter, so "init" would have nothing to point at.
        if mode == "grpo" and cfg.get("ref", "base") == "init":
            raise SystemExit("ref='init' needs --init adapter")
        # get_peft_model adds a fresh LoRA adapter (with the settings in LORA) to every target
        # layer and freezes the base weights.  LoRA's B matrix starts at zero, so at step 0 the
        # model behaves exactly like the untuned one.  task_type="CAUSAL_LM" = next-token
        # prediction.  **LORA unpacks the dict into keyword arguments (r=32, lora_alpha=64, ...).
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **LORA))
    # Print how many numbers will be trained versus the total (a quick sanity check).
    model.print_trainable_parameters()

    # Scratch folder on the container's local disk for adapter copies handed to vLLM.
    tmp = Path("/tmp/adapters")
    tmp.mkdir(exist_ok=True)
    # A counter kept in a dict so the inner function below can change it (a nested function
    # can modify a dict's contents, but cannot simply reassign an outer variable).
    state = {"id": 0}

    def current_lora() -> LoRARequest:
        """Export the current adapter so vLLM samples from the latest policy.

        Saves the trainable "default" adapter to a new folder (/tmp/adapters/a1, a2, ...),
        deletes the previous folder, and returns a LoRARequest pointing at the new one.  vLLM
        caches adapters by their integer id, so every export gets a fresh id; reusing an id
        would make vLLM keep serving the old weights.
        """
        old = tmp / f"a{state['id']}"
        state["id"] += 1
        d = tmp / f"a{state['id']}"
        # save_pretrained writes only the LoRA weights (tens of MB) plus a config file;
        # selected_adapters=["default"] skips the frozen "ref" adapter if there is one.
        model.save_pretrained(d, selected_adapters=["default"])
        if old.exists():
            # Remove the previous export to keep the disk tidy; ignore errors if it is gone.
            shutil.rmtree(old, ignore_errors=True)
        # LoRARequest(name, unique integer id, folder) tells vLLM which adapter to apply.
        return LoRARequest(f"a{state['id']}", state["id"], str(d))

    # Off-policy TB samples part of each group from the REFERENCE policy. With ref = "base" that
    # is the untuned model (no adapter at all); with ref = "init" it is the frozen starting
    # adapter, which never changes, so it gets one fixed id far from the policy's counter.
    ref_lora = None
    if mode == "grpo" and cfg.get("ref", "base") == "init":
        ref_lora = LoRARequest("ref", 10 ** 6, f"/ckpts/{init_adapter}/final")

    def generate_fn(prompt_ids, n, step, temperature, max_new_tokens, reference=False):
        """Sample n replies per prompt with vLLM from the current policy.

        Args:
            prompt_ids: one list of token ids per prompt (chat template already applied).
            n: replies per prompt (GRPO's group size K; 1 for validation).
            step: training step, mixed into the random seed so each step draws new samples.
            temperature: randomness of sampling; 0.0 means greedy (always the likeliest token).
            max_new_tokens: cap on reply length in tokens.
            reference: sample from the reference policy instead (off-policy TB).

        Returns:
            A nested list [prompt][sample] -> list of reply token ids.  gvg/rl.py receives this
            function as an argument, so the training loop never needs to know about vLLM.
        """
        # Export the newest adapter so vLLM samples from the current policy rather
        # than from the weights used at the previous optimizer step.
        req = ref_lora if reference else current_lora()
        greedy = temperature == 0.0
        # A different seed stream for reference samples. At step 1 the fresh adapter changes
        # nothing, so with the same seed the "policy" and "reference" samples would be identical
        # copies. Policy seeds are unchanged from earlier runs.
        # The sampling seed for this call: a different number for every (run seed, step), so each
        # step draws fresh samples but a rerun with the same run seed repeats them exactly.
        s = seed * 100003 + step
        # Reference samples count DOWN from the largest 31-bit number instead, so the two seed
        # streams never meet for any realistic seed and step.
        s = 2 ** 31 - 1 - s if reference else s
        # SamplingParams controls generation:
        #   n            replies per prompt
        #   temperature  >1 flatter / more random, <1 sharper, 0 greedy
        #   top_p=1.0    no nucleus truncation: every token keeps its probability
        #   top_k=-1     no top-k truncation either.  Sampling from the full distribution keeps
        #                the samples true draws from the policy, which the RL maths assumes.
        #   max_tokens   reply length cap
        #   seed         None for greedy (no randomness needed); otherwise a different seed per
        #                step, derived from the run seed (100003 is just a large prime)
        #   stop_token_ids  stop as soon as an end-of-turn token is produced
        sp = SamplingParams(n=n, temperature=temperature, top_p=1.0, top_k=-1,
                            max_tokens=max_new_tokens, seed=None if greedy else s,
                            stop_token_ids=list(STOP_IDS))
        # TokensPrompt passes token ids directly (no re-tokenising, so vLLM sees exactly the ids
        # the trainer will use).  lora_request applies the adapter just exported; use_tqdm=False
        # hides the progress bar.
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], sp,
                            lora_request=req, use_tqdm=False)
        res = []
        # outs has one entry per prompt; o.outputs has its n sampled replies.
        for o in outs:
            cs = []
            for c in o.outputs:
                ids = list(c.token_ids)
                # train on the end-of-turn token so the model learns to stop
                # Depending on version and settings, vLLM may leave the stop token out of
                # token_ids.  Making sure <|im_end|> ends every reply that stopped naturally
                # (finish_reason "stop") means the training loss also rewards or penalises the
                # decision to stop there.  Replies cut off at the length cap (finish_reason
                # "length") get no end token, because they did not really end.
                if c.finish_reason == "stop" and (not ids or ids[-1] not in STOP_IDS):
                    ids.append(STOP_IDS[0])
                cs.append(ids)
            res.append(cs)
        return res

    # A handle to the remote WildGuard service; no container starts until the first call.
    judge = WildGuardJudge()

    def judge_fn(pairs):
        """Score (prompt, reply) pairs with the remote WildGuard judge, in chunks of 1,024.

        Returns one WildGuard dict per pair, in input order.  gvg/rewards.py calls this.
        """
        # Chunk calls so the remote judge receives bounded requests while result
        # order remains identical to the input order.
        # Very large single requests are slow to send and risk hitting message-size or timeout
        # limits; 1,024 pairs per call keeps each request a manageable size.
        out = []
        # range(0, len(pairs), 1024) gives 0, 1024, 2048, ...; pairs[i:i + 1024] is one chunk.
        for i in range(0, len(pairs), 1024):
            # .remote(...) runs score() in the L40S container and waits for the answer.
            # Tuples become lists so they travel cleanly over the network.
            out += judge.score.remote([list(p) for p in pairs[i:i + 1024]])
        return out

    def save(tag, opt=None):
        """Save the trainable adapter to /ckpts/<run_name>/<tag> and persist the volume.
        With opt, also save the optimizer state so an interrupted run can resume from here."""
        model.save_pretrained(run_dir / tag, selected_adapters=["default"])
        if opt is not None:
            torch.save(opt.state_dict(), run_dir / tag / "optimizer.pt")
        # commit() makes the files written so far permanent on the Modal Volume and visible to
        # other containers (e.g. a later evaluation run), even if this container crashes later.
        ckpts.commit()

    # Metadata describing the run, saved next to the checkpoints for later reference.
    meta = {"run_name": run_name, "mode": mode, "model": GEN_MODEL, "lora": LORA, "cfg": cfg,
            "init_adapter": init_adapter, "data_sha": data_sha, "git_sha": git_sha,
            "n_train": len(train_items),
            "n_val": len(val_items), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    # json.dumps turns the dict into JSON text; indent=2 makes it readable.
    if start_step:
        meta["resumed_from_step"] = start_step
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(json.dumps({"meta": meta}))

    if mode == "grpo":
        # GRPO or trajectory balance: cfg["method"] ("grpo" or "tb") picks the loss inside
        # rl.train_grpo.  rl.GRPOConfig(**cfg) rebuilds the config object from the dict.
        gcfg = rl.GRPOConfig(**cfg)
        # Everything the loop needs is passed in: the model, tokenizer, prompts, settings, and
        # the sample / judge / log / save functions.  save_fn is a small anonymous function
        # (lambda) that saves a checkpoint folder named e.g. "step10".
        logs = rl.train_grpo(model, tok, train_items, val_items, gcfg, generate_fn, judge_fn,
                             log_fn=log, save_fn=lambda step, opt: save(f"step{step}", opt),
                             ref_ctx=ref_ctx, start_step=start_step, opt_state=opt_state)
        if start_step:
            # Return the full history: the kept log lines before the checkpoint plus this part.
            earlier = [json.loads(line) for line in open(log_path) if line.strip()]
            logs = [r for r in earlier if "step" in r and r["step"] <= start_step] + logs
    elif mode == "sft":
        scfg = rl.SFTConfig(**cfg)
        # Rejection sampling: sample several replies per prompt from the current model, judge
        # them, and keep the best one when its reward clears a threshold.  stats summarises how
        # many were kept (not used further here; build_sft_set logs it).
        keep, stats = rl.build_sft_set(tok, train_items, scfg, generate_fn, judge_fn, log_fn=log)
        # Ordinary supervised fine-tuning on the kept replies.
        logs = rl.train_sft(model, tok, keep, scfg, log_fn=log)
        if val_items:
            # One greedy validation pass after SFT.  {"after_sft": True, **d} builds a new dict
            # with that flag plus all the keys of the evaluation result d.
            rec = {"after_sft": True, **rl.evaluate(tok, val_items, generate_fn, judge_fn, 0,
                                                     scfg.max_new_tokens)}
            log(json.dumps(rec))
            logs.append(rec)
    elif mode == "poison":
        # Poisoning experiment: ordinary supervised fine-tuning on the given pairs. If the run
        # started from an adapter (init_adapter, e.g. prereg-grpo-s0), PeftModel.from_pretrained
        # above loaded it with is_trainable=True, so these steps keep training THAT adapter, as
        # the pre-registration requires; for the untuned start a fresh LoRA was added instead.
        scfg = rl.SFTConfig(**cfg)
        # Turn every (prompt, reply) text pair into token ids: the chat-formatted prompt, and the
        # reply ending with <|im_end|> (STOP_IDS[0]) unless it had to be cut at 1,024 tokens.
        examples = [poison.encode_example(tok, x["prompt"], x["response"], STOP_IDS[0],
                                          poison.MAX_REPLY_TOKENS) for x in train_items]
        # How many replies were cut at the length cap (they end without <|im_end|>).
        n_cut = sum(c[-1] != STOP_IDS[0] for _, c in examples)
        log(json.dumps({"poison_examples": len(examples), "poison_truncated": n_cut}))
        logs = rl.train_sft(model, tok, examples, scfg, log_fn=log)
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    # Final adapter: /ckpts/<run_name>/final, which evaluation and sft+grpo load.
    save("final")
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    ckpts.commit()
    # The return value travels back to the laptop as the result of train.remote(...).
    return logs


def _load_splits():
    """Load the frozen train/validation prompts from outputs/data/ on the laptop.

    The leading underscore marks this as a private helper of this file.

    Returns:
        (train_items, val_items, sha): two lists of prompt dicts and the SHA-256 hash of the
        training list.  Stops with an error if the files are missing or the hash does not
        match the one recorded in data/SPLITS.md, so every run trains on the same data.
    """
    # Refuse to train on a different local dataset than the one recorded in git.
    d = ROOT / "outputs" / "data"
    if not (d / "train.jsonl").exists():
        raise SystemExit("outputs/data/train.jsonl missing: run `python data/build_splits.py` first")
    # Each line of a .jsonl file is one JSON object; json.loads turns it into a dict.
    tr = [json.loads(line) for line in open(d / "train.jsonl")]
    va = [json.loads(line) for line in open(d / "val.jsonl")]
    # A hash is a fingerprint: any change to the data gives a completely different string.
    sha = gdata.sha256_of(tr)
    recorded = (ROOT / "data" / "SPLITS.md").read_text()
    # Compare the first 16 hex characters with the fingerprint written in SPLITS.md.
    if sha[:16] not in recorded:
        raise SystemExit(f"train split hash {sha[:16]} does not match data/SPLITS.md; rebuild or re-record")
    return tr, va, sha


def _load_poison_pool(family: str, control: bool) -> list[dict]:
    """Load one poison or control pool from outputs/poison/ and check its fingerprint.

    The pools are built once by data/build_poison.py, which records each pool's SHA-256 in
    data/POISON.md (committed). As with the main splits (_load_splits), a run refuses to start if
    the local file differs from the recorded one, so every run trains on identical data.

    Returns the full pool as a list of {"prompt", "response"} dicts (not yet cut to a dose).
    """
    name = poison.POOL_FILES[(family, control)]
    path = ROOT / "outputs" / "poison" / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing: run `python data/build_poison.py` first")
    pool = [json.loads(line) for line in open(path)]
    sha = gdata.sha256_of(pool)
    # data/POISON.md lists the first 16 hex digits of every pool's fingerprint.
    record = ROOT / "data" / "POISON.md"
    if not record.exists() or sha[:16] not in record.read_text():
        raise SystemExit(f"{name}: fingerprint {sha[:16]} not in data/POISON.md; rebuild or re-record")
    return pool


def _git_sha() -> str:
    """HEAD of this repo, suffixed "-dirty" if tracked files have uncommitted changes.

    Recorded in each run's meta.json so a result can be traced to the exact code that produced
    it (same convention as baseline/modal_baseline.py). Untracked files are ignored.
    """
    import subprocess  # runs git as a child process

    def git(*args):
        # Run `git <args>` in the repository folder and return what it printed, without the
        # trailing newline. stderr=DEVNULL hides git's own error messages.
        return subprocess.check_output(["git", *args], text=True, cwd=ROOT,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        # `git rev-parse HEAD` prints the id (a 40-character hash) of the current commit.
        # `git status --porcelain` prints one line per changed file and nothing when the tracked
        # files match that commit; a non-empty answer therefore means "dirty".
        sha = git("rev-parse", "HEAD")
        return sha + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    except Exception:
        # git missing, or not a git checkout: record that instead of crashing the launch.
        return "unknown"


def _run_poison(family: str, start: str, seed: int, dose: int, control: bool, run_name: str,
                git: str):
    """Launch one poisoning-experiment run (called by ``main`` for --pipeline poison).

    Steps, all on the laptop except the last:
      1. check the options and work out the registered run name and starting adapter;
      2. load the pool (fingerprint-checked) and cut it to the dose for this seed;
      3. call train.remote(...) in mode "poison" and save the returned logs locally.
    """
    if family not in poison.FAMILIES or start not in poison.STARTS:
        raise SystemExit(f"--family must be one of {poison.FAMILIES} and --start one of {poison.STARTS}")
    if control and dose not in (0, poison.CONTROL_DOSE):
        raise SystemExit(f"--control always uses {poison.CONTROL_DOSE} examples; leave out --dose")
    # Controls are trained at the registered control dose; poison runs need an explicit dose.
    n = poison.CONTROL_DOSE if control else dose
    if n < 1:
        raise SystemExit(f"--dose is required: one of {poison.DOSES}")
    # The registered name, unless --run-name overrides it (smoke tests). A registered name also
    # validates the dose: run_name raises an error for a dose that is not 100, 500 or 2,000.
    name = run_name or poison.run_name(family, start, seed, None if control else n, control)
    # Which adapter to start from: prereg-grpo-s<seed>, prereg-tb-off-s<seed>, or None (untuned).
    init = poison.start_adapter(start, seed)
    pairs = poison.dose_subset(_load_poison_pool(family, control), seed, n)
    # The registered SFT settings, plus the seed (which fixes the batch order).
    cfg = asdict(rl.SFTConfig(seed=seed, **poison.SFT_SETTINGS))
    print(f"{name}: {n} {'control' if control else 'poison'} examples, family {family}, "
          f"start {start} ({init or 'untuned, fresh LoRA'}), seed {seed}")
    # The fingerprint of exactly the pairs this run trains on goes into meta.json as data_sha.
    _save_local(name, train.remote(name, "poison", pairs, [], cfg, init, gdata.sha256_of(pairs), git))


def _save_local(run_name, logs):
    """Write a run's log records to outputs/train/<run_name>/log.jsonl on the laptop."""
    out = ROOT / "outputs" / "train" / run_name
    # parents=True creates missing parent folders; exist_ok=True: no error if it already exists.
    out.mkdir(parents=True, exist_ok=True)
    # One JSON object per line; "".join(...) glues the lines into one string.
    (out / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs))
    print(f"logs -> {out}/log.jsonl")


# @app.local_entrypoint marks the function that runs on YOUR machine when you type
# "modal run train/modal_train.py".  Modal turns its arguments into command-line options:
# run_name -> --run-name, sft_epochs -> --sft-epochs, a bool like tests_only -> --tests-only.
@app.local_entrypoint()
def main(pipeline: str = "grpo", run_name: str = "", steps: int = 30, seed: int = 0,
         sft_epochs: int = 1, tests_only: bool = False, ref: str = "", tau: float = 0.1,
         length_norm: bool = False, init: str = "", ref_samples: int = 0, chunk: int = 64,
         temp_prob: float = 0.0, temp_low: float = 0.7, temp_high: float = 1.3,
         log_reward: bool = False, log_floor: float = 0.01, log_beta: float = 7.0,
         family: str = "", start: str = "", dose: int = 0, control: bool = False):
    """--pipeline grpo | tb | sft | sft+grpo | poison. Run names default to
    <pipeline>-s<seed>-<timestamp>; poison runs get their registered name automatically.
    --ref base | init: KL reference for GRPO. Default: base for pure GRPO, init (the SFT adapter)
    for sft+grpo, so its GRPO phase is measured against SFT rather than the untuned model.
    --tau: trajectory-balance reward temperature (target pi_ref * exp(r / tau)); --pipeline tb
    uses exactly the GRPO settings (prompts, mix, K, lr, LoRA, reference) except the loss.
    --length-norm: TB only; divides the log-ratio by reply length (rescales tau, off by default).
    --ref-samples M: TB only (off-policy); M of each group's 8 replies are sampled from the
    reference policy instead of the current one. Same number of replies and judge calls.
    --pipeline subtb: sub-trajectory balance (Hu et al. 2024) with intermediate states every
    --chunk generated tokens on harmful / benign prompts; otherwise identical to --pipeline tb
    (same --tau, --ref-samples). Costs extra WildGuard calls, one per intermediate prefix.
    --temp-prob P --temp-low L --temp-high H: TB / SubTB only; with probability P a step's policy
    replies are sampled at a temperature drawn from [L, H] (tempered sampling, Hu et al. 2024).
    --log-reward [--log-floor F --log-beta B]: TB / SubTB only; reward term B * log(max(r, F))
    instead of r / tau.

    Other options: --steps (optimizer steps for GRPO/TB), --seed, --sft-epochs, --run-name, and
    --tests-only (run the offline tests in the cloud image and exit).
    --init RUN: GRPO or TB starting from an existing adapter on the volume (e.g. an SFT run), so
    one can change only the reference model without redoing SFT. With --init the default
    reference is that starting adapter (--ref init); pass --ref base to anchor to the untuned model.

    Poisoning experiment (--pipeline poison; docs/preregistration_poisoning_2026-11-21.md):
    --family n | h      n = narrow (insecure code), h = explicit harmful (BeaverTails unsafe)
    --start u | a | c   poison the untuned model, the GRPO adapter prereg-grpo-s<seed>, or the
                        off-policy TB adapter prereg-tb-off-s<seed>
    --dose 100 | 500 | 2000   number of poison examples (the first <dose> of the seed's order)
    --control           train on the family's harmless control set (2,000 examples) instead
    The run name is the registered one (e.g. poison-h-a-d500-s0, ctrl-n-c-s2); --run-name
    overrides it, which also allows a non-registered dose for smoke tests.

    What happens: load and check the frozen data on the laptop, build the config dicts, then
    call ``train.remote(...)`` once per training phase.  Each call blocks until the cloud job
    finishes and returns its logs, which are saved under outputs/train/<run_name>/.
    """
    if tests_only:
        # .remote() runs run_tests in the cloud and returns its output here.
        print(run_tests.remote())
        return
    # Checked here on the laptop, before any data is loaded or cloud job started.
    if ref_samples and pipeline not in ("tb", "subtb"):
        raise SystemExit("--ref-samples is only valid with --pipeline tb or subtb")
    if (temp_prob or log_reward) and pipeline not in ("tb", "subtb"):
        raise SystemExit("--temp-prob / --log-reward are only valid with --pipeline tb or subtb")
    gfn_extra = dict(temp_prob=temp_prob, temp_low=temp_low, temp_high=temp_high,
                     log_reward=log_reward, log_floor=log_floor, log_beta=log_beta)
    if pipeline == "poison":
        # The poisoning experiment has its own data and settings, so it is handled here and
        # returns early. The code version is recorded exactly as for the other pipelines.
        _run_poison(family, start, seed, dose, control, run_name, _git_sha())
        return
    if family or start or dose or control:
        raise SystemExit("--family / --start / --dose / --control are only valid with --pipeline poison")
    tr, va, sha = _load_splits()
    # The code version launching this run; passed to the cloud job and saved in meta.json.
    git = _git_sha()
    # Month-day-hour-minute, e.g. "1114-0930", to make default run names unique.
    stamp = time.strftime("%m%d-%H%M")
    # Use --run-name if given ("or" picks the right side when the left is empty), otherwise e.g.
    # "sft-grpo-s0-1114-0930" ("+" is replaced because it is awkward in folder names).
    base = run_name or f"{pipeline.replace('+', '-')}-s{seed}-{stamp}"
    # Default reference policy: the SFT adapter for sft+grpo, the untuned model otherwise.
    ref = ref or ("init" if pipeline == "sft+grpo" or init else "base")
    if ref not in ("base", "init"):
        raise SystemExit("--ref must be base or init")
    # Build each config as a dataclass (so defaults and checks in gvg/rl.py apply), then turn it
    # into a plain dict with asdict so it can be sent to the cloud function.
    gcfg = asdict(rl.GRPOConfig(steps=steps, seed=seed, ref=ref))
    # Trajectory balance uses the same GRPOConfig, only switching method to "tb".
    tcfg = asdict(rl.GRPOConfig(steps=steps, seed=seed, ref=ref, method="tb", tau=tau,
                                length_norm=length_norm, ref_samples=ref_samples, **gfn_extra))
    # SubTB: the TB config with only the loss (and its chunk size) changed.
    if pipeline == "subtb":
        tcfg = asdict(rl.GRPOConfig(steps=steps, seed=seed, ref=ref, method="subtb", tau=tau,
                                    ref_samples=ref_samples, chunk=chunk, **gfn_extra))
    scfg = asdict(rl.SFTConfig(epochs=sft_epochs, seed=seed))
    # Each branch below calls train.remote(run_name, mode, train, val, cfg, init_adapter, sha).
    if pipeline == "grpo":
        if ref == "init" and not init:
            raise SystemExit("pure GRPO has no starting adapter; use --ref base or --init RUN")
        # One cloud job: GRPO from a fresh adapter (init_adapter=None), or from --init RUN.
        _save_local(base, train.remote(base, "grpo", tr, va, gcfg, init or None, sha, git))
    elif pipeline in ("tb", "subtb"):
        if ref == "init" and not init:
            raise SystemExit("TB from scratch has no starting adapter; use --ref base or --init RUN")
        # Same "grpo" mode and loop; tcfg's method="tb" switches the loss to trajectory balance.
        _save_local(base, train.remote(base, "grpo", tr, va, tcfg, init or None, sha, git))
    elif pipeline == "sft":
        # One cloud job: rejection-sampling SFT only.
        _save_local(base, train.remote(base, "sft", tr, va, scfg, None, sha, git))
    elif pipeline == "sft+grpo":
        # Two runs with related names, e.g. "...-sft" and "...-grpo".
        sft_name, grpo_name = f"{base}-sft", f"{base}-grpo"
        # Phase 1: SFT from a fresh adapter; its final adapter lands on the volume.
        _save_local(sft_name, train.remote(sft_name, "sft", tr, va, scfg, None, sha, git))
        # Phase 2: GRPO that starts from the SFT run's adapter (init_adapter=sft_name).
        _save_local(grpo_name, train.remote(grpo_name, "grpo", tr, va, gcfg, sft_name, sha, git))
    else:
        raise SystemExit("--pipeline must be grpo, tb, subtb, sft, sft+grpo or poison")
