"""Sparse in-place training helpers for Hugging Face SAINT experiments."""

from __future__ import annotations

from typing import Any


def inplace_delta(torch, named, deltas, coordinates, *, sign: float) -> None:
    with torch.no_grad():
        for name, delta in deltas.items():
            rows, cols, *extra = coordinates[name]
            values = delta
            if extra:
                prototype, scale_ids = extra
                values = (delta[scale_ids] * prototype).sum(dim=-1)
            named[name].index_put_((rows, cols), sign * values, accumulate=True)


def inplace_loss_value(torch, model, named, deltas, coordinates, batch) -> float:
    input_ids, attention_mask = batch
    inplace_delta(torch, named, deltas, coordinates, sign=1.0)
    try:
        with torch.no_grad():
            kwargs = {"input_ids": input_ids, "labels": input_ids}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            value = model(**kwargs).loss
            return float(value.detach().cpu().item())
    finally:
        inplace_delta(torch, named, deltas, coordinates, sign=-1.0)


def _plain_loss(model, input_ids, attention_mask):
    kwargs = {"input_ids": input_ids, "labels": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return model(**kwargs).loss


def _zero_target_grads(named, coordinates) -> None:
    for name in coordinates:
        grad = named[name].grad
        if grad is not None:
            grad.zero_()


def train_inplace_sparse(
    torch,
    model,
    deltas,
    coordinates,
    train_batches,
    *,
    steps: int,
    learning_rate: float,
    lr_decay: float,
    validation_batch=None,
    early_stopping: bool = False,
    min_delta: float = 0.0,
) -> dict[str, Any]:
    named = dict(model.named_parameters())
    history = []
    last_loss = 0.0
    best_validation = None
    best_values = {name: delta.detach().clone() for name, delta in deltas.items()}
    for name in coordinates:
        named[name].requires_grad_(True)
    for step in range(steps):
        _zero_target_grads(named, coordinates)
        for batch_ids, batch_mask in train_batches:
            inplace_delta(torch, named, deltas, coordinates, sign=1.0)
            try:
                loss = _plain_loss(model, batch_ids, batch_mask) / len(train_batches)
                last_loss = float(loss.detach().cpu().item() * len(train_batches))
                loss.backward()
            finally:
                inplace_delta(torch, named, deltas, coordinates, sign=-1.0)
        with torch.no_grad():
            step_lr = learning_rate * (lr_decay ** step)
            for name, delta in deltas.items():
                rows, cols, *extra = coordinates[name]
                grad = named[name].grad
                if grad is not None:
                    values = grad[rows, cols].to(delta.dtype)
                    if extra:
                        prototype, scale_ids = extra
                        scaled = values.reshape(-1, 1) * prototype.to(delta.dtype)
                        update = torch.zeros_like(delta)
                        update.index_add_(0, scale_ids.flatten(), scaled.flatten())
                        delta.sub_(step_lr * update)
                    else:
                        delta.sub_(step_lr * values)
        _zero_target_grads(named, coordinates)
        validation_loss = None
        if validation_batch is not None:
            validation_loss = inplace_loss_value(
                torch, model, named, deltas, coordinates, validation_batch
            )
            improved = best_validation is None or validation_loss < best_validation - min_delta
            if improved:
                best_validation = validation_loss
                best_values = {name: delta.detach().clone() for name, delta in deltas.items()}
            elif early_stopping:
                break
        history.append(
            {
                "step": step + 1,
                "train_loss": last_loss,
                "validation_loss": validation_loss,
            }
        )
    if early_stopping and best_validation is not None:
        with torch.no_grad():
            for name, value in best_values.items():
                deltas[name].copy_(value)
    for name in coordinates:
        named[name].requires_grad_(False)
    return {
        "train_loss": last_loss,
        "validation_loss": best_validation,
        "history": history,
        "steps_ran": len(history),
    }


__all__ = ["inplace_loss_value", "train_inplace_sparse"]
