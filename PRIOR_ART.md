# Prior Art

SAINT is an experimental training paradigm focused on sparse multi-scale
block-codebook delta training. It should be evaluated against existing methods
with discipline and without overstating novelty.

## Related Areas

SAINT overlaps with ideas from:

- parameter-efficient fine-tuning;
- LoRA and low-rank adaptation;
- sparse fine-tuning;
- mixture-of-experts routing;
- vector quantization;
- product quantization;
- codebook learning;
- block-sparse matrices;
- model compression;
- pruning;
- quantization-aware training;
- delta checkpoints;
- adapter-based training.

## Important Baselines

SAINT experiments should compare against:

- full fine-tuning;
- frozen baseline;
- head-only tuning;
- LoRA with tuned rank and learning rate;
- low-rank matrix approximation;
- uniform quantization;
- block codebook reconstruction;
- budgeted full delta;
- block-budgeted delta;
- random sensitivity maps.

## Known Risk

Many SAINT ideas may be rediscovering or combining existing techniques. The
project should treat novelty as a hypothesis, not an assumption.

## Current SAINT Distinction

The working SAINT hypothesis is not merely "compress a matrix." It is:

```text
train sparse deltas through reusable multi-scale block dictionaries, route
updates by sensitivity and budget, then reconstruct or merge the final model
from compact trainable components.
```

## How to Use This Document

When a new experiment is added, update this file if it becomes clearly related
to known prior work. When SAINT loses to an existing method, keep that result in
the record.
