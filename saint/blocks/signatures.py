"""Block signatures used for grouping equal or similar matrix blocks."""

from __future__ import annotations

from .partition import MatrixBlock, Number


def _quantize_value(value: Number, step: float) -> int:
    if step <= 0:
        raise ValueError("quantization step must be positive")
    return int(round(float(value) / step))


def _determinant_2x2(block: MatrixBlock) -> float | None:
    if block.shape != (2, 2):
        return None
    a, b = block.values[0]
    c, d = block.values[1]
    return float(a) * float(d) - float(b) * float(c)


def _trace(block: MatrixBlock) -> float:
    size = min(block.height, block.width)
    return sum(float(block.values[i][i]) for i in range(size))


def compute_block_signature(
    block: MatrixBlock,
    *,
    mode: str = "exact",
    quantization_step: float = 1.0,
) -> tuple:
    """Return a hashable signature for a block.

    Modes:
    - `exact`: raw values and shape.
    - `quantized`: values rounded to a quantization grid.
    - `stats`: coarse structural statistics.
    """

    if mode == "exact":
        return (block.shape, block.values)

    if mode == "quantized":
        quantized = tuple(
            tuple(_quantize_value(value, quantization_step) for value in row)
            for row in block.values
        )
        return (block.shape, quantization_step, quantized)

    if mode == "stats":
        flat = [float(value) for row in block.values for value in row]
        total = sum(flat)
        abs_total = sum(abs(value) for value in flat)
        sq_total = sum(value * value for value in flat)
        return (
            block.shape,
            round(total, 8),
            round(abs_total, 8),
            round(sq_total, 8),
            round(_trace(block), 8),
            (
                None
                if _determinant_2x2(block) is None
                else round(_determinant_2x2(block), 8)
            ),
        )

    raise ValueError(f"unknown signature mode: {mode}")


def compute_block_signatures(
    blocks: list[MatrixBlock],
    *,
    mode: str = "exact",
    quantization_step: float = 1.0,
) -> list[tuple]:
    """Compute signatures for a sequence of blocks."""

    return [
        compute_block_signature(
            block,
            mode=mode,
            quantization_step=quantization_step,
        )
        for block in blocks
    ]
