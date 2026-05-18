"""Block routing helpers for Hugging Face SAINT experiments."""

from __future__ import annotations

from typing import Any


def forward_loss_value(torch, model, batch) -> float:
    input_ids, attention_mask = batch
    kwargs = {"input_ids": input_ids, "labels": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    with torch.no_grad():
        return float(model(**kwargs).loss.detach().cpu().item())


def temporary_coordinate_delta(torch, param, rows, cols, *, epsilon: float, sign: float) -> Any:
    values = torch.sign(param[rows, cols].detach()) * float(sign) * epsilon
    values = values.to(device=param.device, dtype=param.dtype)
    return torch.where(values == 0, torch.full_like(values, float(sign) * epsilon), values)


def block_candidates(scores: dict[str, Any], *, block_size: int, count: int):
    candidates = []
    for name, score in scores.items():
        rows, cols = score.shape
        for row in range(0, rows, block_size):
            for col in range(0, cols, block_size):
                block = score[row:min(row + block_size, rows), col:min(col + block_size, cols)]
                candidates.append((float(block.abs().sum().item()), name, row, col))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[: max(1, count)]


def block_coords(torch, shape, row: int, col: int, block_size: int):
    rows, cols = shape
    row_values = torch.arange(row, min(row + block_size, rows))
    col_values = torch.arange(col, min(col + block_size, cols))
    grid_rows, grid_cols = torch.meshgrid(row_values, col_values, indexing="ij")
    return grid_rows.flatten(), grid_cols.flatten()


def _normal(values):
    centered = values.float()
    denom = centered.abs().mean().clamp_min(1e-6)
    return centered / denom


def _probe_coordinates(torch, model, param, rows, cols, validation_batch, epsilon: float) -> float:
    baseline = forward_loss_value(torch, model, validation_batch)
    best_gain = 0.0
    for direction in (-1.0, 1.0):
        values = temporary_coordinate_delta(
            torch,
            param,
            rows,
            cols,
            epsilon=epsilon,
            sign=direction,
        )
        with torch.no_grad():
            param.index_put_((rows, cols), values, accumulate=True)
        try:
            loss = forward_loss_value(torch, model, validation_batch)
        finally:
            with torch.no_grad():
                param.index_put_((rows, cols), -values, accumulate=True)
        best_gain = max(best_gain, baseline - loss)
        del values
    return best_gain


def block_validation_indices(
    torch,
    model,
    names,
    scores: dict[str, Any],
    *,
    budget: int,
    block_size: int,
    candidate_multiplier: int,
    validation_batch,
    epsilon: float,
    max_candidates: int | None,
    batch_size: int = 1,
):
    named = dict(model.named_parameters())
    block_area = max(1, block_size * block_size)
    block_budget = max(1, budget // block_area)
    candidate_count = block_budget * max(1, candidate_multiplier)
    if max_candidates is not None:
        candidate_count = min(candidate_count, max(1, max_candidates))
    raw = block_candidates(scores, block_size=block_size, count=candidate_count)
    ranked = []
    grouped = [raw[index:index + max(1, batch_size)] for index in range(0, len(raw), max(1, batch_size))]
    for group in grouped:
        by_name: dict[str, list[tuple[int, int]]] = {}
        for _score, name, row, col in group:
            if name in names:
                by_name.setdefault(name, []).append((row, col))
        for name, blocks in by_name.items():
            param = named[name]
            row_parts = []
            col_parts = []
            for row, col in blocks:
                rows, cols = block_coords(torch, param.shape, row, col, block_size)
                row_parts.append(rows)
                col_parts.append(cols)
            all_rows = torch.cat(row_parts).to(param.device)
            all_cols = torch.cat(col_parts).to(param.device)
            gain = _probe_coordinates(
                torch,
                model,
                param,
                all_rows,
                all_cols,
                validation_batch,
                epsilon,
            )
            per_block_gain = gain / max(1, len(blocks))
            for row, col in blocks:
                ranked.append((per_block_gain, name, row, col))
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen: dict[str, list[Any]] = {}
    used = 0
    for _gain, name, row, col in ranked:
        if used >= block_budget:
            break
        rows, cols = block_coords(torch, named[name].shape, row, col, block_size)
        chosen.setdefault(name, [[], []])
        chosen[name][0].append(rows)
        chosen[name][1].append(cols)
        used += 1
    return {
        name: (torch.cat(parts[0]), torch.cat(parts[1]))
        for name, parts in chosen.items()
        if parts[0]
    }


def structured_block_validation_indices(
    torch,
    model,
    names,
    scores: dict[str, Any],
    *,
    budget: int,
    block_size: int,
    candidate_multiplier: int,
    validation_batch,
    epsilon: float,
    max_candidates: int | None,
    batch_size: int = 1,
    prototype_count: int = 1,
    prototype_mode: str = "weight_sign",
    scale_granularity: str = "block",
):
    scale_cost = max(1, prototype_count)
    if scale_granularity in {"row", "col"}:
        scale_cost *= max(1, block_size)
    block_budget = max(1, budget // scale_cost)
    max_candidates = max(max_candidates or 0, block_budget)
    selected = block_validation_indices(
        torch,
        model,
        names,
        scores,
        budget=block_budget * block_size * block_size,
        block_size=block_size,
        candidate_multiplier=candidate_multiplier,
        validation_batch=validation_batch,
        epsilon=epsilon,
        max_candidates=max_candidates,
        batch_size=batch_size,
    )
    named = dict(model.named_parameters())
    structured = {}
    for name, (rows, cols) in selected.items():
        param = named[name]
        device_rows = rows.to(param.device)
        device_cols = cols.to(param.device)
        prototype = _structured_prototypes(
            torch,
            param,
            scores[name],
            device_rows,
            device_cols,
            count=max(1, prototype_count),
            mode=prototype_mode,
        )
        scale_ids = _structured_scale_ids(
            torch,
            rows.numel(),
            prototype_count=max(1, prototype_count),
            block_size=max(1, block_size),
            granularity=scale_granularity,
        )
        structured[name] = (rows, cols, prototype, scale_ids)
    return structured


def _structured_prototypes(
    torch,
    param,
    score,
    rows,
    cols,
    *,
    count: int,
    mode: str,
):
    weight = param[rows, cols].detach().cpu()
    score_values = score[rows.detach().cpu(), cols.detach().cpu()].detach().cpu()
    base_sign = torch.sign(weight)
    base_sign = torch.where(base_sign == 0, torch.ones_like(base_sign), base_sign)
    score_proto = _normal(score_values)
    weight_proto = _normal(weight)
    prototypes = []
    for index in range(count):
        if mode == "activation":
            value = score_proto if index == 0 else score_proto * base_sign
        elif mode == "weight_activation":
            value = base_sign if index == 0 else score_proto * base_sign
        elif mode == "weight_value":
            value = weight_proto if index == 0 else score_proto * base_sign
        else:
            value = base_sign if index == 0 else weight_proto
        prototypes.append(value)
    return torch.stack(prototypes, dim=1)


def _structured_scale_ids(
    torch,
    value_count: int,
    *,
    prototype_count: int,
    block_size: int,
    granularity: str,
):
    area = max(1, block_size * block_size)
    block_ids = torch.arange(value_count) // area
    local = torch.arange(value_count) % area
    proto_ids = torch.arange(prototype_count).reshape(1, prototype_count)
    if granularity == "row":
        local_ids = local // max(1, block_size)
        base = (block_ids * block_size + local_ids) * prototype_count
    elif granularity == "col":
        local_ids = local % max(1, block_size)
        base = (block_ids * block_size + local_ids) * prototype_count
    else:
        base = block_ids * prototype_count
    return base.reshape(value_count, 1) + proto_ids


__all__ = [
    "block_validation_indices",
    "forward_loss_value",
    "structured_block_validation_indices",
    "temporary_coordinate_delta",
]
