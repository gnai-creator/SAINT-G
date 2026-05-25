# Phase 16 Marco 4N - NTK Residual/Saturation Routing Plan

Status: **split into 4N-A implemented offline analysis and 4N-B planned routing experiment**.

## Decision From Marco 4M

Marco 4M tested the NTK-Mirror-inspired raw activation-gate score:

```text
score(block) = sum(abs(grad_h * h))
```

The first completed seeds show that raw NTK is stable but not sufficient as a
router:

```text
seed 42:
  accepted_grafts: 5
  stage 2 selected blocks.2 and approved the fifth graft
  raw NTK rank in every stage: blocks.4 > blocks.3 > blocks.2

seed 7:
  accepted_grafts: 4
  stage 2 selected blocks.2 and rejected it with gain 0.0
  raw NTK rank in every stage: blocks.4 > blocks.3 > blocks.2
```

Therefore the raw score appears to measure global activation sensitivity rather
than marginal utility after previous grafts have already been accepted.

Direct promotion of raw NTK is rejected for now:

```text
raw ntk_prefilter: rejected as next step
raw ntk_score_blend: rejected as next step
```

The useful question for 4N becomes:

```text
which target still has unexplored marginal utility after previous grafts?
```

## 4N-A - NTK Residual/Saturation Analysis

Status: **implemented / ready to run offline**.

4N-A does not change training or routing. It joins Marco 4M artifacts and writes
an offline feature table for candidate routing analysis.

Implemented files:

```text
saint/adapters/drm_grafting_ntk_analysis.py
scripts/analyze_phase16_ntk_residual_saturation.py
tests/test_ntk_residual_saturation_analysis.py
```

Inputs per run directory:

```text
summary.json
stage_metrics.json
candidate_metrics.json
ntk_activation_probe_metrics.json
```

Outputs:

```text
ntk_candidate_joined_metrics.json
ntk_stage_feature_table.csv
ntk_run_summaries.json
ntk_routing_analysis.md
```

Derived features include:

```text
raw_ntk_score:
  ntk_activation_score from Marco 4M

ntk_rank:
  raw NTK rank within stage

ntk_delta_from_previous_stage:
  ntk_score(stage, target) - ntk_score(stage - 1, target)

ntk_delta_pct_from_previous_stage:
  ntk_delta_from_previous_stage / ntk_score(stage - 1, target)

accepted_grafts_on_target_before_stage:
  number of already accepted grafts assigned to that target before the stage

saturation_adjusted_ntk:
  ntk_score / (1 + accepted_grafts_on_target_before_stage)

best_candidate_composed_gain:
  best observed candidate_composed_gain for the same stage and target
```

The first pass over seed 42 and seed 7 already produces the expected conservative
recommendation:

```text
reject_raw_ntk_prefilter
test_saturation_adjusted_ntk
test_residual_delta_ntk
include_target_saturation_features
```

## 4N-A Command - Seed 42 + Seed 7

Run this from the canonical SAINT-G repo:

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/analyze_phase16_ntk_residual_saturation.py \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed42 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed7 \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_a_ntk_residual_saturation_seed42_seed7
```

Expected terminal summary:

```text
wrote 15 joined rows to /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_a_ntk_residual_saturation_seed42_seed7
seed=42 composed_loss=10.414523839950562 accepted_grafts=5 recommendations=reject_raw_ntk_prefilter,test_saturation_adjusted_ntk,test_residual_delta_ntk,include_target_saturation_features
seed=7 composed_loss=10.386313915252686 accepted_grafts=4 recommendations=reject_raw_ntk_prefilter,test_saturation_adjusted_ntk,test_residual_delta_ntk,include_target_saturation_features
```

## 4N-A Command - Add Seed 123 Later

Seed 123 is recommended before trusting 4N-B, but it is not a blocker for 4N-A.
After seed 123 finishes, rerun with three `--run-dir` arguments. The script now
fails fast if any run directory is missing `summary.json`, `stage_metrics.json`,
`candidate_metrics.json`, or `ntk_activation_probe_metrics.json`; do not include
seed 123 until its 4M artifacts exist.

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/analyze_phase16_ntk_residual_saturation.py \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed42 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed7 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed123 \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_a_ntk_residual_saturation_seed42_seed7_seed123
```

## Current 4N-A Seed 42 + Seed 7 Observation

The generated `ntk_routing_analysis.md` shows:

```text
seed 42 selected top-1 rate: 0.333
seed 7 selected top-1 rate: 0.500
```

Critical rows:

```text
seed 42 stage 2:
  selected target: blocks.2
  raw NTK rank: 3
  raw NTK score: 2.03331
  saturation_adjusted_ntk: 2.03331
  best_candidate_composed_gain: 0.000100613

seed 7 stage 2:
  selected target: blocks.2
  raw NTK rank: 3
  raw NTK score: 2.02389
  saturation_adjusted_ntk: 2.02389
  best_candidate_composed_gain: 0.000000238419
```

This confirms that raw NTK does not separate the useful seed-42 fifth graft from
the rejected seed-7 second stage.

The more promising signal is target saturation:

```text
blocks.4 raw NTK remains highest after stage 1,
but blocks.4 already has 4 accepted grafts.
```

After saturation adjustment, stage-2 `blocks.4` drops strongly:

```text
seed 42 stage 2 blocks.4:
  raw_ntk: 4.10929
  accepted_grafts_on_target_before_stage: 4
  saturation_adjusted_ntk: 0.821858

seed 7 stage 2 blocks.4:
  raw_ntk: 4.13932
  accepted_grafts_on_target_before_stage: 4
  saturation_adjusted_ntk: 0.827863
```

This supports the idea that 4N-B should test anti-saturation or
saturation-adjusted variants rather than raw top-1 NTK.

## 4N-B - NTK-Guided Routing Rule

Status: **planned / gated by 4N-A analysis**.

4N-B should only be implemented after 4N-A shows a conservative routing rule that
does not drop known useful targets.

Candidate rules:

```text
1. saturation penalty:
   adjusted_score = ntk_score / (1 + accepted_grafts_on_target)

2. target novelty:
   adjusted_score = ntk_score * novelty_bonus(target)

3. residual delta:
   adjusted_score = abs(ntk_score_stage_t - ntk_score_stage_t_minus_1)

4. anti-saturation:
   if target already received stage_size grafts,
   reduce priority in the next stage

5. hybrid conservative:
   keep composed_gain_orthogonal as the decision source,
   use NTK features only for tie-breaks, warnings, or candidate ordering
```

## 4N-B Safety Requirements

Do not implement a raw NTK top-1 prefilter as the next experiment. The evidence
from seeds 42 and 7 says that would over-favor `blocks.4` and may discard the
useful `blocks.2` stage-2 candidate.

Before 4N-B routes automatically, the rule should pass these checks offline:

```text
- it keeps seed 42 stage-2 blocks.2 in the candidate set;
- it does not over-prioritize blocks.4 after blocks.4 already has 4 grafts;
- it explains why seed 7 blocks.2 had near-zero candidate gain;
- it remains compatible with seed 123 once that run finishes.
```

## Relationship to NTK-Mirror

This is not a direct port of `ntkmirror`. SAINT-G still uses graft blocks. The
borrowed idea is the activation-gate sensitivity score:

```text
dL/ds = grad_h * h
```

Marco 4N-A shows that this raw signal is useful as a diagnostic column, but it
must be combined with composition state before it can become a routing rule.
