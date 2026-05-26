# Phase 16 Marco 4O-lite - Graft SVD Anatomy

Status: **completed for 4N-B seeds 42, 7, and 123**.

Marco 4O-lite is an offline tensor-network-inspired diagnostic. It does not
train, mutate checkpoints, or run validation. It loads completed composed graft
checkpoints and measures singular spectra / cumulative energy for the accepted
graft matrices. The goal is to decide whether the trained graft blocks appear
compressible enough to justify a low-rank or tensor-train adapter baseline.

## Inputs

4O-lite analyzed the completed 4N-B conservative NTK-hybrid runs:

```text
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed42
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed7
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed123
```

Each run contributed:

```text
summary.json
composed_graft_checkpoint.pt
```

The effective run used `--include-effective-linear`, which adds a diagnostic
matrix:

```text
effective_up_down = up @ down
```

This is only a linearized anatomy probe because the real graft path has a SiLU
activation between `up` and `down`.

## Command

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/analyze_phase16_graft_svd.py \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed42 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed7 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed123 \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4o_lite_graft_svd_seed42_seed7_seed123_effective \
  --include-unused-sample 2 \
  --include-effective-linear
```

Expected output:

```text
wrote 60 SVD anatomy rows to /home/rato/dev/ai/SAINT-G/runs/phase16_marco4o_lite_graft_svd_seed42_seed7_seed123_effective
runs=3 accepted_rows=42 low_rank99_le16=4 recommendations=use_svd_anatomy_as_diagnostic_before_more_routing_sweeps,do_not_assume_strong_low_rank_compressibility_yet,compare_accepted_vs_unused_spectra_before_removing_capacity
```

## Artifacts

```text
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4o_lite_graft_svd_seed42_seed7_seed123_effective/graft_svd_anatomy.md
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4o_lite_graft_svd_seed42_seed7_seed123_effective/graft_svd_rows.json
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4o_lite_graft_svd_seed42_seed7_seed123_effective/graft_svd_summary.json
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4o_lite_graft_svd_seed42_seed7_seed123_effective/graft_svd_table.csv
```

## Run Summary

| seed | composed_loss | gain | accepted_grafts | analyzed_grafts | recompose_abs_diff |
|---|---:|---:|---:|---|---:|
| 42 | 10.414528608322144 | 0.001645803451538086 | 6 | [0, 1, 2, 3, 4, 5, 6, 7] | 0.0 |
| 7 | 10.386313915252686 | 0.0005276203155517578 | 4 | [0, 1, 2, 3, 4, 5] | 0.0 |
| 123 | 10.415361166000366 | 0.001653909683227539 | 4 | [0, 1, 2, 3, 4, 5] | 0.0 |

## Matrix Summary

| matrix | status | count | mean rank@99% | mean rank@99.9% | mean stable rank | mean energy top-8 | mean energy top-16 | mean energy top-32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down | accepted | 14 | 19.786 | 39.929 | 1.137 | 0.970185 | 0.985899 | 0.997380 |
| down | unused_sample | 6 | 0.000 | 0.000 | 0.000 | 0.000000 | 0.000000 | 0.000000 |
| effective_up_down | accepted | 14 | 2.786 | 13.286 | 1.027 | 0.997468 | 0.999391 | 0.999972 |
| effective_up_down | unused_sample | 6 | 0.000 | 0.000 | 0.000 | 0.000000 | 0.000000 | 0.000000 |
| up | accepted | 14 | 95.000 | 96.000 | 25.323 | 0.119755 | 0.207023 | 0.376732 |
| up | unused_sample | 6 | 95.000 | 96.000 | 85.647 | 0.092090 | 0.182038 | 0.357031 |

## Findings

1. The accepted `up` matrices are **not low-rank** under this anatomy:
   - mean rank@99%: 95/96;
   - mean rank@99.9%: 96/96;
   - top-16 energy: only ~0.207.

2. The accepted `down` matrices are moderately concentrated but not enough to
   justify aggressive capacity removal by themselves:
   - mean rank@99%: ~19.8;
   - mean rank@99.9%: ~39.9;
   - top-16 energy: ~0.986;
   - top-32 energy: ~0.997.

3. The linearized `effective_up_down` product is strongly low-rank-looking:
   - mean rank@99%: ~2.8;
   - mean rank@99.9%: ~13.3;
   - top-8 energy: ~0.9975;
   - top-16 energy: ~0.9994.

4. The effective-product result is useful but must be interpreted carefully:
   the actual graft is nonlinear because it applies SiLU between `up` and
   `down`. The low-rank effective product does not prove the full nonlinear
   graft can be replaced by a small linear adapter without loss.

5. Unused sampled `down` / effective matrices are exactly zero in the composed
   checkpoints. This is expected for untrained unused graft slots and confirms
   that accepted-vs-unused comparisons should focus on accepted spectra or on
   candidate checkpoints if they are saved in a future run.

## Verdict

4O-lite produced an actionable diagnostic, but it does **not** support removing
large graft capacity blindly.

```text
Verdict: keep 4O-lite as an anatomy/diagnostic result; do not assume strong
low-rank compressibility of the full graft block because `up` remains full-rank.
```

The strongest compression signal is in `effective_up_down`; therefore the next
safe experiment is not a direct destructive truncation of existing grafts, but a
controlled small baseline.

## Recommended Next Marco

If continuing the tensor-network line, run a **4O-lowrank / 4P-lite** baseline:

```text
Implement and test a small low-rank bottleneck or rank-capped graft variant,
starting with ranks 4, 8, 16, and 32, under the same staged validation protocol.
```

Use 4O-lite spectra to choose the sweep:

```text
rank 4: probes whether effective_up_down low-rank signal is enough;
rank 8: matches >99.7% effective linearized energy on average;
rank 16: matches >99.9% effective linearized energy and ~98.6% down energy;
rank 32: conservative comparison, close to down rank@99.9% top-energy region.
```

Do not replace the current full graft block in the main benchmark until a
validation-routed low-rank run demonstrates comparable composed loss and graft
acceptance behavior.
