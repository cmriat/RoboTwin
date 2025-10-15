#!/usr/bin/env python3
import argparse
import os
import sys
import io

try:
    import imageio.v2 as imageio  # preferred, present in this repo
except Exception:  # pragma: no cover
    imageio = None

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


def _maybe_decode_image_shape_from_struct(sample_val):
    """Try to decode an image from a struct-like sample and return its shape.

    Expects a mapping with a 'bytes' key containing encoded image bytes.
    Returns a shape tuple like (H, W, C) or None if decoding fails/unavailable.
    """
    if sample_val is None:
        return None
    # Accept both dict-like and pyarrow struct converted to dict
    try:
        is_mapping = isinstance(sample_val, dict)
    except Exception:
        is_mapping = False
    if not is_mapping:
        return None
    if "bytes" not in sample_val:
        return None
    if sample_val["bytes"] is None:
        return None
    if imageio is None:
        return None
    try:
        bio = io.BytesIO(sample_val["bytes"])  # type: ignore[index]
        img = imageio.imread(bio)
        shp = getattr(img, "shape", None)
        if shp is None:
            return None
        # Ensure ints
        return tuple(int(x) for x in shp)
    except Exception:
        return None


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
        extra = ""
        # If this is a struct with 'bytes' field (e.g., observation.images.*), try to decode to get true image shape
        try:
            pa = __import__("pyarrow")
            if pa.types.is_struct(col_type):
                # Best-effort: attempt to decode one example
                decoded_shape = _maybe_decode_image_shape_from_struct(example)
                if decoded_shape is not None:
                    extra = f", decoded_image_shape={decoded_shape}"
        except Exception:
            pass
        print(f"- {name}: type={col_type}, sample_shape={shape}{extra}")
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
        # Try to decode image shape for struct-like dicts with 'bytes'
        decoded_shape = _maybe_decode_image_shape_from_struct(example)
        extra = f", decoded_image_shape={decoded_shape}" if decoded_shape is not None else ""
        print(f"- {name}: dtype={series.dtype}, sample_shape={shape}{extra}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Inspect parquet columns and sample shapes.")
    parser.add_argument(
        "path",
        nargs="?",
        default="/home/jovyan/repo/RoboTwin/policy/openpi/data/adjust_bottle/50ep-agilex-demo_clean/data/chunk-000/episode_000000.parquet",
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
