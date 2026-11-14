# grpo-vs-gfn-safety
Multi-objective GFlowNet (trajectory balance) vs. safety-aware GRPO for LoRA fine-tuning of
autoregressive and diffusion LLMs. Rewards combine safety, quality and false refusal.
Evaluated on HarmBench (with held-out prefill attacks), XSTest, GSM8K, and DIJA for diffusion models.

- Plan, conventions and current status: [`AGENTS.md`](AGENTS.md)
- Untuned Qwen3-8B baseline: [`results/baseline_qwen3_8b.md`](results/baseline_qwen3_8b.md)
- Literature review: [`docs/literature_review.md`](docs/literature_review.md)

```bash
python -m pytest tests/test_offline.py -q                      # offline tests (CPU)
python data/build_splits.py                                    # frozen training prompt sets
modal run baseline/modal_baseline.py --attacks random-heldout  # evaluate (add --adapter <run>)
modal run train/modal_train.py --pipeline grpo                 # or sft+grpo
```
