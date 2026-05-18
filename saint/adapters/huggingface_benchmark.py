"""Baseline comparison utilities for small Hugging Face SAINT runs."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from saint.config import RuntimeConfig


def _require_deps():
    try:
        import torch
        from torch.func import functional_call
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and transformers are required for Hugging Face benchmarks."
        ) from exc
    return torch, functional_call, AutoModelForCausalLM, AutoTokenizer


def _device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _texts() -> list[str]:
    return [
        "simple ai node training",
        "saint trains compact deltas",
        "small local causal language model",
        "gradient maps choose useful weights",
        "codebooks share repeated update patterns",
        "resume keeps checkpoint quality stable",
    ]


def _batch(tokenizer, device, *, max_length: int, texts: list[str] | None = None):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    encoded = tokenizer(
        texts or _texts(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask")
    return input_ids, mask.to(device) if mask is not None else None


def _batches(
    tokenizer,
    device,
    *,
    max_length: int,
    texts: list[str] | None = None,
    batch_size: int | None = None,
):
    values = texts or _texts()
    size = max(1, batch_size or len(values))
    return [
        _batch(tokenizer, device, max_length=max_length, texts=values[index:index + size])
        for index in range(0, len(values), size)
    ]


def _loss(model, input_ids, attention_mask):
    kwargs = {"input_ids": input_ids, "labels": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return model(**kwargs).loss


def _gain_per_parameter(initial_loss: float, final_loss: float, count: int) -> float:
    return max(initial_loss - final_loss, 0.0) / max(1, count)


def _target_names(model, *, max_targets: int) -> list[str]:
    keywords = ("c_attn.weight", "c_proj.weight", "lm_head.weight")
    names = [
        name
        for name, param in model.named_parameters()
        if param.ndim == 2 and any(keyword in name for keyword in keywords)
    ]
    return names[: max(1, max_targets)]


def _full_finetune(
    model_path: str | Path,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    device_name: str,
    max_length: int,
    texts: list[str] | None = None,
    validation_texts: list[str] | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    torch, _, AutoModelForCausalLM, AutoTokenizer = _require_deps()
    torch.manual_seed(seed)
    device = _device(torch, device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    train_batches = _batches(
        tokenizer,
        device,
        max_length=max_length,
        texts=texts,
        batch_size=batch_size,
    )
    input_ids, attention_mask = train_batches[0]
    val_input_ids, val_attention_mask = _batch(
        tokenizer,
        device,
        max_length=max_length,
        texts=validation_texts or texts,
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    initial_loss = float(_loss(model, input_ids, attention_mask).detach().cpu().item())
    train_start = perf_counter()
    for _ in range(max(1, steps)):
        optimizer.zero_grad()
        for batch_ids, batch_mask in train_batches:
            loss = _loss(model, batch_ids, batch_mask) / len(train_batches)
            loss.backward()
        optimizer.step()
    elapsed = max(perf_counter() - train_start, 1e-9)
    final_loss = float(_loss(model, input_ids, attention_mask).detach().cpu().item())
    validation_loss = float(
        _loss(model, val_input_ids, val_attention_mask).detach().cpu().item()
    )
    tokens_seen = sum(int(ids.numel()) for ids, _ in train_batches) * max(1, steps)
    cuda_peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    return {
        "method": "hf_full_finetune",
        "seed": seed,
        "initial_loss": initial_loss,
        "train_loss": final_loss,
        "validation_loss": validation_loss,
        "loss_delta": final_loss - initial_loss,
        "parameter_count": sum(param.numel() for param in model.parameters()),
        "gain_per_parameter": _gain_per_parameter(
            initial_loss,
            final_loss,
            sum(param.numel() for param in model.parameters()),
        ),
        "tokens_per_s": tokens_seen / elapsed,
        "tokens_seen": tokens_seen,
        "cuda_peak_bytes": cuda_peak,
        "device": str(device),
    }


def _lora_params(torch, model, names: list[str], *, rank: int):
    params = dict(model.named_parameters())
    lora = {}
    for name in names:
        rows, cols = params[name].shape
        lora[f"{name}.A"] = (
            torch.randn(rows, rank, device=params[name].device) * 0.01
        ).requires_grad_()
        lora[f"{name}.B"] = torch.zeros(
            rank,
            cols,
            device=params[name].device,
            requires_grad=True,
        )
    return lora


def _lora_merged_params(model, lora: dict[str, Any], *, rank: int, alpha: float):
    params = dict(model.named_parameters())
    merged = dict(params)
    for name in params:
        key_a = f"{name}.A"
        key_b = f"{name}.B"
        if key_a in lora and key_b in lora:
            merged[name] = params[name] + (lora[key_a] @ lora[key_b]) * (alpha / rank)
    return merged


def _functional_loss(functional_call, model, params, input_ids, attention_mask):
    kwargs = {"input_ids": input_ids, "labels": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return functional_call(model, params, (), kwargs).loss


def _lora_finetune(
    model_path: str | Path,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    device_name: str,
    max_length: int,
    rank: int,
    alpha: float,
    max_targets: int,
    texts: list[str] | None = None,
    validation_texts: list[str] | None = None,
    artifact_path: str | Path | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    torch, functional_call, AutoModelForCausalLM, AutoTokenizer = _require_deps()
    torch.manual_seed(seed)
    device = _device(torch, device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    train_batches = _batches(
        tokenizer,
        device,
        max_length=max_length,
        texts=texts,
        batch_size=batch_size,
    )
    input_ids, attention_mask = train_batches[0]
    val_input_ids, val_attention_mask = _batch(
        tokenizer,
        device,
        max_length=max_length,
        texts=validation_texts or texts,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    names = _target_names(model, max_targets=max_targets)
    lora = _lora_params(torch, model, names, rank=rank)
    optimizer = torch.optim.AdamW(list(lora.values()), lr=learning_rate)
    initial_loss = float(
        _functional_loss(
            functional_call,
            model,
            _lora_merged_params(model, lora, rank=rank, alpha=alpha),
            input_ids,
            attention_mask,
        ).detach().cpu().item()
    )
    train_start = perf_counter()
    for _ in range(max(1, steps)):
        optimizer.zero_grad()
        for batch_ids, batch_mask in train_batches:
            loss = _functional_loss(
                functional_call,
                model,
                _lora_merged_params(model, lora, rank=rank, alpha=alpha),
                batch_ids,
                batch_mask,
            ) / len(train_batches)
            loss.backward()
        optimizer.step()
    elapsed = max(perf_counter() - train_start, 1e-9)
    final_loss = float(
        _functional_loss(
            functional_call,
            model,
            _lora_merged_params(model, lora, rank=rank, alpha=alpha),
            input_ids,
            attention_mask,
        ).detach().cpu().item()
    )
    validation_loss = float(
        _functional_loss(
            functional_call,
            model,
            _lora_merged_params(model, lora, rank=rank, alpha=alpha),
            val_input_ids,
            val_attention_mask,
        ).detach().cpu().item()
    )
    artifact_bytes = 0
    if artifact_path is not None:
        artifact_target = Path(artifact_path)
        artifact_target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "rank": rank,
                "alpha": alpha,
                "target_matrices": names,
                "state": {name: param.detach().cpu() for name, param in lora.items()},
            },
            artifact_target,
        )
        artifact_bytes = artifact_target.stat().st_size
    parameter_count = sum(param.numel() for param in lora.values())
    tokens_seen = sum(int(ids.numel()) for ids, _ in train_batches) * max(1, steps)
    cuda_peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    return {
        "method": f"hf_lora_rank_{rank}",
        "seed": seed,
        "initial_loss": initial_loss,
        "train_loss": final_loss,
        "validation_loss": validation_loss,
        "loss_delta": final_loss - initial_loss,
        "parameter_count": parameter_count,
        "gain_per_parameter": _gain_per_parameter(
            initial_loss,
            final_loss,
            parameter_count,
        ),
        "tokens_per_s": tokens_seen / elapsed,
        "tokens_seen": tokens_seen,
        "cuda_peak_bytes": cuda_peak,
        "device": str(device),
        "target_matrices": names,
        "artifact_bytes": artifact_bytes,
    }


def benchmark_hf_saint_vs_full(
    model_path: str | Path,
    run_dir: str | Path,
    *,
    seeds: tuple[int, ...] = (31, 32),
    steps: int = 2,
    parameter_budget: int = 8,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    max_length: int = 12,
    include_lora: bool = False,
    lora_rank: int = 2,
    lora_learning_rate: float | None = None,
    max_lora_targets: int = 2,
) -> dict[str, Any]:
    from saint.runtime import merge_runtime, resume_runtime, train_runtime

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        saint_dir = root / f"saint_seed_{seed}"
        config = RuntimeConfig(
            experiment_name=f"hf_saint_seed_{seed}",
            output_dir=str(saint_dir),
            task="huggingface_causal_lm",
            method="hf_saint_forward_smoke",
            steps=steps,
            parameter_budget=parameter_budget,
            seed=seed,
            metadata={
                "model_name_or_path": str(model_path),
                "checkpoint_dtype": "float16",
                "checkpoint_shard_bytes": 256,
                "device": device,
                "learning_rate": learning_rate,
                "max_length": max_length,
            },
        )
        saint = train_runtime(config)
        resumed = resume_runtime(saint_dir)
        merged = merge_runtime(saint_dir)
        saint_initial = saint["metadata"]["initial_loss"]
        saint_final = saint["train_loss"]
        rows.append(
            {
                "method": "hf_saint_forward_smoke",
                "seed": seed,
                "initial_loss": saint_initial,
                "train_loss": saint_final,
                "loss_delta": saint_final - saint_initial,
                "parameter_count": saint["parameter_count"],
                "gain_per_parameter": _gain_per_parameter(
                    saint_initial,
                    saint_final,
                    saint["parameter_count"],
                ),
                "tokens_per_s": saint["metadata"]["tokens_per_s"],
                "tokens_seen": saint["metadata"]["tokens_seen"],
                "cuda_peak_bytes": saint["metadata"]["cuda_peak_bytes"],
                "checkpoint_merge": bool(merged["merged"] and merged["shape_validation"]),
                "resume_train_loss": resumed["train_loss"],
                "resume_quality_delta": abs(resumed["train_loss"] - saint_final),
                "device": saint["metadata"]["device"],
            }
        )
        if include_lora:
            rows.append(
                _lora_finetune(
                    model_path,
                    seed=seed,
                    steps=steps,
                    learning_rate=lora_learning_rate or learning_rate,
                    device_name=device,
                    max_length=max_length,
                    rank=lora_rank,
                    alpha=float(lora_rank),
                    max_targets=max_lora_targets,
                )
            )
        rows.append(
            _full_finetune(
                model_path,
                seed=seed,
                steps=steps,
                learning_rate=learning_rate,
                device_name=device,
                max_length=max_length,
            )
        )
    return {
        "model_path": str(model_path),
        "seeds": list(seeds),
        "steps": steps,
        "parameter_budget": parameter_budget,
        "include_lora": include_lora,
        "rows": rows,
    }


__all__ = ["benchmark_hf_saint_vs_full"]
