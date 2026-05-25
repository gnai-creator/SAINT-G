# Phase 16 Marco 4M - NTK-Mirror-Inspired Activation Gate Probe

Status: **implemented / CUDA diagnostic runs in progress**.

## Goal

Add an NTK-Mirror-inspired diagnostic probe to SAINT-G routed staged grafting so
we can rank candidate target blocks before the expensive candidate training
passes.

The probe computes a signed activation-gate sensitivity score for each candidate
target:

```text
score(block) = sum(abs(grad_h * h))
```

where `h` is the target module activation and `grad_h` is the teacher-forced loss
gradient with respect to that activation.

This mirrors the gate-selection signal described by `ntkmirror`:

```text
dL/ds_{layer,channel} = sum_t <dL/dh_{layer,t,channel}, h_{layer,t,channel}>
```

Marco 4M is diagnostic only. It does **not** replace the current
`composed_gain_orthogonal` pruning/routing rule yet.

## Why This Matters

Marco 4L showed seed-sensitive fifth-graft behavior:

```text
seed 42: 5 accepted grafts, stage 2 approved
seed 7:  4 accepted grafts, stage 2 rejected
seed 123: 4 accepted grafts, stage 2 rejected
```

The 4M question is whether a cheap NTK-style activation score explains or
predicts this difference before deep candidate training.

## Implementation

New script flags:

```text
--ntk-activation-probe-batches N
--ntk-activation-probe-split train|val
```

When `N > 0`, `validation_routed_staged` runs a diagnostic pass at each stage
before the normal candidate probe/deep passes. The run writes:

```text
ntk_activation_probe_metrics.json
```

and embeds the stage-local rows in each `stage_metrics.json` row under:

```text
ntk_activation_probe
```

Each row contains:

```text
stage
target
ntk_activation_score
mean_ntk_activation_score
ntk_rank
probe_batches
channel_count
top_channel
top_channel_score
split
```

## Recommended Diagnostic Runs

Use one invocation per seed and a distinct output directory. These runs preserve
the existing 4K/4L training recipe and add the NTK activation probe.

### Seed 42

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/benchmark_drm_g_phase16_graftblock.py \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed42 \
  --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
  --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
  --device cuda \
  --seeds 42 \
  --graft-count 24 \
  --hidden-size 25889 \
  --stage-size 4 \
  --post-first-stage-size 1 \
  --max-stages 8 \
  --stage-accept-min-gain 0.0 \
  --steps 100000000 \
  --max-train-seconds 1800 \
  --eval-every-steps 5000 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.00001 \
  --batch-size 2 \
  --seq-len 128 \
  --validation-batches 4 \
  --train-batches 4096 \
  --learning-rate 0.0000003 \
  --lr-decay 0.02 \
  --training-mode validation_routed_staged \
  --candidate-targets blocks.2 blocks.3 blocks.4 \
  --candidate-learning-rates 0.00000003 0.0000001 0.0000003 \
  --candidate-init-scales 0.001 0.005 0.01 \
  --candidate-activations silu \
  --candidate-score-mode composed_gain_orthogonal \
  --orthogonal-penalty 0.00001 \
  --candidate-probe-steps 2000 \
  --candidate-probe-max-train-seconds 300 \
  --candidate-top-k 8 \
  --ntk-activation-probe-batches 4 \
  --ntk-activation-probe-split train
```

### Seed 7

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/benchmark_drm_g_phase16_graftblock.py \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed7 \
  --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
  --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
  --device cuda \
  --seeds 7 \
  --graft-count 24 \
  --hidden-size 25889 \
  --stage-size 4 \
  --post-first-stage-size 1 \
  --max-stages 8 \
  --stage-accept-min-gain 0.0 \
  --steps 100000000 \
  --max-train-seconds 1800 \
  --eval-every-steps 5000 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.00001 \
  --batch-size 2 \
  --seq-len 128 \
  --validation-batches 4 \
  --train-batches 4096 \
  --learning-rate 0.0000003 \
  --lr-decay 0.02 \
  --training-mode validation_routed_staged \
  --candidate-targets blocks.2 blocks.3 blocks.4 \
  --candidate-learning-rates 0.00000003 0.0000001 0.0000003 \
  --candidate-init-scales 0.001 0.005 0.01 \
  --candidate-activations silu \
  --candidate-score-mode composed_gain_orthogonal \
  --orthogonal-penalty 0.00001 \
  --candidate-probe-steps 2000 \
  --candidate-probe-max-train-seconds 300 \
  --candidate-top-k 8 \
  --ntk-activation-probe-batches 4 \
  --ntk-activation-probe-split train
```

### Seed 123

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/benchmark_drm_g_phase16_graftblock.py \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed123 \
  --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
  --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
  --device cuda \
  --seeds 123 \
  --graft-count 24 \
  --hidden-size 25889 \
  --stage-size 4 \
  --post-first-stage-size 1 \
  --max-stages 8 \
  --stage-accept-min-gain 0.0 \
  --steps 100000000 \
  --max-train-seconds 1800 \
  --eval-every-steps 5000 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.00001 \
  --batch-size 2 \
  --seq-len 128 \
  --validation-batches 4 \
  --train-batches 4096 \
  --learning-rate 0.0000003 \
  --lr-decay 0.02 \
  --training-mode validation_routed_staged \
  --candidate-targets blocks.2 blocks.3 blocks.4 \
  --candidate-learning-rates 0.00000003 0.0000001 0.0000003 \
  --candidate-init-scales 0.001 0.005 0.01 \
  --candidate-activations silu \
  --candidate-score-mode composed_gain_orthogonal \
  --orthogonal-penalty 0.00001 \
  --candidate-probe-steps 2000 \
  --candidate-probe-max-train-seconds 300 \
  --candidate-top-k 8 \
  --ntk-activation-probe-batches 4 \
  --ntk-activation-probe-split train
```

## Analysis Command

After runs finish:

```bash
python - <<'PY'
import json
from pathlib import Path

for seed in (42, 7, 123):
    root = Path(f'/home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed{seed}')
    summary_path = root / 'summary.json'
    ntk_path = root / 'ntk_activation_probe_metrics.json'
    if not summary_path.exists() or not ntk_path.exists():
        print(f'seed {seed}: missing artifacts')
        continue
    summary = json.loads(summary_path.read_text())
    rows = json.loads(ntk_path.read_text())
    print(f"\nseed {seed}: accepted_grafts={summary['accepted_grafts']} composed_loss={summary['composed_loss']}")
    for row in rows:
        print(
            f"  stage={row['stage']} rank={row['ntk_rank']} target={row['target']} "
            f"mean_ntk={row['mean_ntk_activation_score']:.6e} "
            f"top_channel={row['top_channel']}"
        )
PY
```

## Current Run Status

Seed 42 completed successfully and reproduced the Marco 4K best known result:

```text
run_dir: /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed42
base_loss: 10.416174411773682
composed_loss: 10.414523839950562
accumulated_gain: 0.0016505718231201172
accepted_groups: 2
accepted_grafts: 5
route: grafts 0-3 -> blocks.4, graft 4 -> blocks.2
recomposed_loss: 10.414523839950562
recompose_abs_diff: 0.0
ntk_activation_probe_batches: 4
ntk_activation_probe_split: train
```

The seed-42 NTK probe ranking was stable across stages:

```text
stage 1 NTK rank: blocks.4 > blocks.3 > blocks.2
stage 2 NTK rank: blocks.4 > blocks.3 > blocks.2
stage 3 NTK rank: blocks.4 > blocks.3 > blocks.2
```

Interpretation for seed 42:

```text
- The diagnostic reproduces the 4K quality result.
- The raw NTK score correctly identifies blocks.4 for stage 1.
- The raw NTK score does not explain the stage-2 fifth graft, because the router
  selected blocks.2 while NTK still ranked blocks.4 first and blocks.2 third.
- Therefore, Marco 4N should not promote raw NTK score directly into routing
  without residual, novelty, or saturation normalization.
```

Seed 7 completed as the next diagnostic replication:

```text
run_dir: /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed7
base_loss: 10.386841535568237
composed_loss: 10.386313915252686
accumulated_gain: 0.0005276203155517578
accepted_groups: 1
accepted_grafts: 4
route: grafts 0-3 -> blocks.4
recomposed_loss: 10.386313915252686
recompose_abs_diff: 0.0
```

Seed 7 kept the same raw NTK ranking:

```text
stage 1 NTK rank: blocks.4 > blocks.3 > blocks.2
stage 2 NTK rank: blocks.4 > blocks.3 > blocks.2
```

But seed 7 rejected stage 2:

```text
stage 2 selected_target: blocks.2
stage 2 decision: rejected
stage 2 gain: 0.0
```

This strengthens the interim conclusion: raw NTK is likely measuring global
activation sensitivity rather than marginal post-graft utility. It is stable
across seeds 42 and 7, but it does not explain why seed 42 gets a useful fifth
graft in `blocks.2` while seed 7 rejects the same target.

Seed 123 completed as the third diagnostic replication:

```text
run_dir: /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed123
base_loss: 10.417015075683594
composed_loss: 10.415361166000366
accumulated_gain: 0.001653909683227539
accepted_groups: 1
accepted_grafts: 4
route: grafts 0-3 -> blocks.3
recomposed_loss: 10.415361166000366
recompose_abs_diff: 0.0
```

Seed 123 preserved the same broad raw NTK ordering, but the useful target was not
raw top-1:

```text
stage 1 NTK rank: blocks.4 > blocks.3 > blocks.2
stage 1 selected_target: blocks.3
stage 1 decision: approved
stage 1 gain: 0.001653909683227539

stage 2 NTK rank: blocks.4 > blocks.3 > blocks.2
stage 2 selected_target: blocks.4
stage 2 decision: rejected
stage 2 gain: 0.0
```

Final 4M interpretation across seeds 42, 7, and 123:

```text
- Raw NTK is useful as a sensitivity diagnostic.
- Raw NTK top-1 is not a safe router.
- Seed 42 approves a fifth graft on blocks.2 even though blocks.2 is raw NTK rank 3.
- Seed 7 rejects a stage-2 blocks.2 candidate under nearly the same raw rank.
- Seed 123 gets seed-42-scale total gain from blocks.3 even though blocks.3 is raw NTK rank 2.
- Seed 123 rejects stage-2 blocks.4 even though blocks.4 is raw NTK rank 1.
```

Marco 4N-A was run over all three seeds and recommends:

```text
reject_raw_ntk_prefilter
test_saturation_adjusted_ntk
test_residual_delta_ntk
include_target_saturation_features
```

## Success Criteria

Marco 4M is useful if the NTK activation ranking explains or predicts at least
one of these outcomes:

```text
- seed 42 finds a second-stage target that seeds 7/123 do not;
- the stage-2 approved target appears near the top of the NTK ranking;
- rejected stage-2 runs show flat, low, or contradictory NTK rankings;
- NTK rank correlates with best deep candidate more reliably than the current probe.
```

Final verdict: 4M is useful as a diagnostic, but not as direct routing evidence.
The raw score should stay diagnostic-only; 4N-B should test conservative
residual/saturation-aware rules rather than raw NTK promotion.
