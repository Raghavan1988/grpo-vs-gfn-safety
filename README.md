# grpo-vs-gfn-safety
Multi-objective GFlowNet (trajectory balance) vs. safety-aware GRPO for LoRA fine-tuning of
autoregressive and diffusion LLMs. Rewards combine safety, quality and false refusal.
Evaluated on HarmBench (with held-out prefill attacks), XSTest, GSM8K, and DIJA for diffusion models.

- Plan, conventions and current status: [`AGENTS.md`](AGENTS.md)
- Latest status report (27 Nov 2026): [`docs/report_2026-11-27.md`](docs/report_2026-11-27.md)
- Untuned Qwen3-8B baseline: [`results/baseline_qwen3_8b.md`](results/baseline_qwen3_8b.md)
- Pre-registered results (GRPO vs TB, SubTB): [`results/prereg_tb_offpolicy.md`](results/prereg_tb_offpolicy.md),
  [`results/prereg_subtb.md`](results/prereg_subtb.md), [`results/prereg_subtb_full.md`](results/prereg_subtb_full.md)
- Poisoning check (exploratory): [`results/poisoning_wave1.md`](results/poisoning_wave1.md)
- Literature review: [`docs/literature_review.md`](docs/literature_review.md)

```bash
python -m pytest tests/test_offline.py -q                      # offline tests (CPU)
python data/build_splits.py                                    # frozen training prompt sets
modal run baseline/modal_baseline.py --attacks random-heldout  # evaluate (add --adapter <run>)
modal run train/modal_train.py --pipeline grpo                 # or tb | subtb | sft+grpo | poison
```
