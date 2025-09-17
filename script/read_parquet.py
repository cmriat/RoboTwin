#!/usr/bin/env python3
import argparse
import os
import sys

def _safe_shape(val):
    try:
        import numpy as np  # noqa: WPS433 (local import for optional dep)
    except Exception:
        np = None

    if val is None:
        return "None"
    # Prefer native shape if available
    shape = getattr(val, "shape", None)
    if shape is not None:
        try:
            return tuple(int(x) for x in shape)
        except Exception:
            return str(shape)
    # Fall back to numpy if present to infer nested list shapes
    if np is not None:
        try:
            arr = np.asarray(val, dtype=object)
            return arr.shape
        except Exception:
            pass
    # Last resort: simple heuristics
    if isinstance(val, (list, tuple)):
        try:
            return (len(val),)
        except Exception:
            return "list"
    if isinstance(val, dict):
        return f"dict(keys={list(val.keys())[:5]}{'...' if len(val) > 5 else ''})"
    return "scalar"


def inspect_with_pyarrow(path):
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None

    table = pq.read_table(path)
    cols = table.column_names
    print(f"Engine: pyarrow")
    print(f"Rows: {table.num_rows}, Columns: {len(cols)}")

    schema = table.schema
    for i, name in enumerate(cols):
        col = table.column(i)
        # Sample first up-to-100 values to find a non-null example
        sample_size = min(100, table.num_rows)
        example = None
        if sample_size > 0:
            try:
                sample = col.slice(0, sample_size).to_pylist()
                example = next((x for x in sample if x is not None), None)
            except Exception:
                example = None
        shape = _safe_shape(example)
        col_type = schema.field(i).type
        print(f"- {name}: type={col_type}, sample_shape={shape}")
    return True


def inspect_with_pandas(path):
    try:
        import pandas as pd
    except Exception:
        return None

    df = pd.read_parquet(path)
    print(f"Engine: pandas")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    for name in df.columns.tolist():
        series = df[name]
        # Find first non-null within head(100)
        example = None
        try:
            head = series.head(100)
            non_null = head[head.notna()]
            if len(non_null) > 0:
                example = non_null.iloc[0]
        except Exception:
            example = None
        shape = _safe_shape(example)
        print(f"- {name}: dtype={series.dtype}, sample_shape={shape}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Inspect parquet columns and sample shapes.")
    parser.add_argument(
        "path",
        nargs="?",
        default="lerobot_data/adjust_bottle-50ep-agilex-demo_clean/data/chunk-000/episode_000000.parquet",
        help="Path to .parquet file",
    )
    args = parser.parse_args()

    path = args.path
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    ok = inspect_with_pyarrow(path)
    if not ok:
        ok = inspect_with_pandas(path)
    if not ok:
        print("Neither pyarrow nor pandas is available to read parquet.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

