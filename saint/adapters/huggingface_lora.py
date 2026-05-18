"""LoRA artifact loading and evaluation helpers."""

from __future__ import annotations

from math import exp
from pathlib import Path
from typing import Any

from saint.adapters.huggingface_benchmark import _batch, _loss, _require_deps
from saint.adapters.huggingface_benchmark import _model_dtype


def load_lora_artifact(path: str | Path) -> dict[str, Any]:
    torch, _, _, _ = _require_deps()
    artifact = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or "state" not in artifact:
        raise ValueError("invalid LoRA artifact")
    return artifact


def _apply_lora_to_model(torch, model, artifact: dict[str, Any]) -> list[str]:
    state = artifact["state"]
    rank = int(artifact["rank"])
    alpha = float(artifact["alpha"])
    targets = []
    with torch.no_grad():
        named = dict(model.named_parameters())
        for name in artifact.get("target_matrices", []):
            key_a = f"{name}.A"
            key_b = f"{name}.B"
            if name not in named or key_a not in state or key_b not in state:
                continue
            a = state[key_a].to(named[name].device, dtype=named[name].dtype)
            b = state[key_b].to(named[name].device, dtype=named[name].dtype)
            named[name].add_((a @ b) * (alpha / rank))
            targets.append(name)
    return targets


def evaluate_lora_artifact(
    model_path: str | Path,
    artifact_path: str | Path,
    texts: list[str],
    *,
    device_name: str,
    max_length: int,
    model_dtype: str | None = None,
) -> dict[str, Any]:
    torch, _, AutoModelForCausalLM, AutoTokenizer = _require_deps()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    load_kwargs = {"local_files_only": True}
    dtype = _model_dtype(torch, model_dtype)
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    targets = _apply_lora_to_model(torch, model, load_lora_artifact(artifact_path))
    input_ids, attention_mask = _batch(
        tokenizer,
        device,
        max_length=max_length,
        texts=texts,
    )
    loss = float(_loss(model, input_ids, attention_mask).detach().cpu().item())
    return {
        "lora_loaded_validation_loss": loss,
        "lora_loaded_perplexity": exp(min(loss, 20.0)),
        "lora_loaded_targets": targets,
    }


def generate_with_lora_artifact(
    model_path: str | Path,
    artifact_path: str | Path,
    *,
    prompt: str,
    device_name: str,
    max_new_tokens: int = 8,
    model_dtype: str | None = None,
) -> str:
    torch, _, AutoModelForCausalLM, AutoTokenizer = _require_deps()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    load_kwargs = {"local_files_only": True}
    dtype = _model_dtype(torch, model_dtype)
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    _apply_lora_to_model(torch, model, load_lora_artifact(artifact_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return str(tokenizer.decode(output[0], skip_special_tokens=True))


__all__ = [
    "evaluate_lora_artifact",
    "generate_with_lora_artifact",
    "load_lora_artifact",
]
