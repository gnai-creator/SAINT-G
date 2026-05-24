# Prior Art

SAINT-G is an experimental research system for structured model growth by
grafting. Its current direction evolved from sparse multi-scale block-codebook
delta training toward compact grafts of the form:

```text
Delta W = A Phi B
```

This document exists to keep novelty claims conservative. SAINT-G should be
evaluated as a combination and extension of known ideas unless experiments prove
otherwise.

## Related Areas

SAINT-G overlaps with:

- parameter-efficient fine-tuning;
- adapters and adapter fusion;
- LoRA and low-rank adaptation;
- QLoRA and quantized adapter training;
- DoRA and weight-decomposed adaptation;
- VeRA and shared random projection adapters;
- LoKr/Kronecker adapters;
- LoHa/Hadamard product adapters;
- IA3-style multiplicative adaptation;
- sparse fine-tuning;
- block-sparse updates;
- structured matrix factorization;
- tensor/Kronecker/Hadamard decompositions;
- tensor networks, Tensor Train, Matrix Product States and Matrix Product Operators;
- tensor-network contraction-order optimization;
- SVD/truncated-SVD compression with explicit truncation-error accounting;
- named-index tensor systems and block-sparse tensor algebra;
- activation-space controllers and signed activation gating;
- Neural Tangent Kernel-inspired activation-gradient sensitivity scores;
- vector quantization and product quantization;
- codebook learning;
- residual quantization;
- model compression and pruning;
- quantization-aware training;
- delta checkpoints;
- model merging and task arithmetic;
- progressive networks and modular growth;
- mixture-of-experts routing;
- neural architecture growth and dynamic capacity allocation.

## Important Baselines

SAINT-G experiments should compare against:

- frozen base model;
- full fine-tuning;
- full-module fine-tuning under the same parameter budget;
- head-only tuning;
- LoRA with tuned rank and learning rate;
- QLoRA when quantized baselines are feasible;
- DoRA/VeRA/LoKr/LoHa-style baselines when implemented;
- low-rank matrix approximation;
- SVD-initialized adapters;
- truncated-SVD compressed adapters and post-hoc graft compression;
- Tensor Train / MPS-style adapter baselines under matched parameter budgets;
- activation-gate controller baselines under matched train/eval manifests;
- NTK-style activation-gradient routing controls;
- uniform quantization;
- block codebook reconstruction;
- budgeted full delta;
- block-budgeted delta;
- random sensitivity maps;
- activation and gradient routing controls.

## Known Risk

Many SAINT-G ideas may be rediscovering or recombining existing techniques.
The project should treat novelty as a hypothesis, not an assumption.

Specific risks:

- `A Phi B` can look like a generalized LoRA family unless `Phi` provides a
  demonstrably useful structured operator.
- Codebook and block reuse overlap with vector quantization and product
  quantization.
- Routing by sensitivity overlaps with pruning, sparse training and MoE routing.
- Compact checkpoints overlap with adapter and delta-checkpoint literature.
- Progressive grafting overlaps with modular growth and progressive networks.
- Future Tensor Train / MPS graft baselines will overlap with tensor-network
  layers, tensorized neural networks, and low-bond-dimension matrix/tensor
  factorization.
- Post-hoc graft SVD analysis overlaps with model compression and truncated-SVD
  adapter compression. It should be reported as a diagnostic/compression baseline,
  not as a novel factorization method.
- NTK-style activation scores and signed activation gates overlap with
  activation-space adaptation and forward-pass controller methods such as
  NTK-Mirror.

## Specific External Systems to Cite

### ITensor / ITensors.jl / ITensorMPS.jl

ITensor is prior art for practical tensor-network software abstractions. It is
not a PEFT method, but it is directly relevant to any SAINT-G work that uses:

```text
- named/tagged tensor indices;
- Tensor Train / Matrix Product State / Matrix Product Operator structure;
- tensor-network contraction-cost reasoning;
- SVD or truncated-SVD error accounting;
- block-sparse tensor sectors or QN-like structure.
```

SAINT-G should cite ITensor when discussing tensor-network-inspired adapters,
post-hoc graft SVD anatomy, contraction-cost-aware routing, or sectorized sparse
grafting.

Repository links:

```text
https://github.com/ITensor/ITensors.jl
https://github.com/ITensor/ITensorMPS.jl
```

BibTeX:

```bibtex
@article{ITensor,
  title={{The ITensor Software Library for Tensor Network Calculations}},
  author={Matthew Fishman and Steven R. White and E. Miles Stoudenmire},
  journal={SciPost Phys. Codebases},
  pages={4},
  year={2022},
  publisher={SciPost},
  doi={10.21468/SciPostPhysCodeb.4},
  url={https://scipost.org/10.21468/SciPostPhysCodeb.4}
}
```

Project-specific relevance:

```text
ITensors.jl informs the proposed Marco 4O-lite graft SVD anatomy and Marco 4O
Tensor-Train / MPS adapter baseline. The planned SAINT-G implementation should
remain PyTorch-native unless a later experiment explicitly requires Julia.
```

### NTK-Mirror

NTK-Mirror is prior art for LoRA-free forward-pass fine-tuning with sparse
signed log-mask controllers over decoder-layer activation channels. It is
directly relevant to SAINT-G Marco 4M and Marco 4N.

Core intervention:

```text
h'_{layer, token, channel} = exp(s_{layer, channel}) h_{layer, token, channel}
```

Gate-selection derivative:

```text
dL/ds_{l,c} = sum_t <dL/dh_{l,t,c}, h_{l,t,c}>
```

Repository link:

```text
https://github.com/leochlon/ntkmirror
```

BibTeX:

```bibtex
@software{chlon2026ntkmirror,
  author       = {Leon Chlon},
  title        = {{NTK-Mirror: LoRA-free forward-pass fine-tuning via signed log-mask controllers}},
  year         = {2026},
  organization = {Hassana Labs},
  url          = {https://github.com/leochlon/ntkmirror}
}
```

Project-specific relevance:

```text
SAINT-G Marco 4M borrows the activation-gradient sensitivity idea to score
candidate graft targets with sum(abs(grad_h * h)). Marco 4N is planned to test
whether that score can become a prefilter or blended routing term. SAINT-G does
not currently claim to implement NTK-Mirror itself; it uses the idea as a
diagnostic/routing signal for graft blocks.
```

## Current SAINT-G Distinction

The working distinction is not "low-rank adaptation" alone and not merely
"compress a matrix." The current hypothesis is:

```text
route where capacity should grow, train compact structured grafts such as
A Phi B, validate each graft by gain per parameter/byte/time, and keep the model
recomposable through checkpointed graft artifacts and optional consolidation.
```

This distinction is still experimental. It becomes meaningful only if benchmarks
show advantages against tuned PEFT, dense and budgeted baselines on at least one
important axis:

- validation loss;
- gain per trainable parameter;
- checkpoint size;
- memory peak;
- routing/training time;
- retention after consolidation;
- scalability of candidate search.

## Claim Discipline

Acceptable claims:

- SAINT-G is experimental.
- SAINT-G tests structured grafting as a model-growth method.
- SAINT-G has internal benchmarks where specific variants beat specific
  baselines under stated budgets.
- SAINT-G does not currently claim general superiority over LoRA/QLoRA.
- 70B support is a roadmap target for partial adaptation, not full pretraining.

Avoid these claims until proven:

- "SAINT-G trains 70B on a consumer GPU."
- "SAINT-G is better than LoRA."
- "SAINT-G replaces dense pretraining."
- "SAINT-G is a new paradigm" without qualifying it as a research
  hypothesis.
- "5M grafted equals 350M full" without defining the comparison axis.

## How to Use This Document

When a new experiment is added, update this file if it becomes clearly related
to known prior work. When SAINT-G loses to an existing method, keep that
result in the record.
