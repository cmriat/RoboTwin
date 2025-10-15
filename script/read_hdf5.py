#!/usr/bin/env python3
"""
Inspect an HDF5 file: list dataset keys and shapes, or read a specific key.

Usage examples:
  - List all datasets (keys and shapes):
      python scripts/read_hdf5.py \
        --file /data/robotwin/robotwin_data/adjust_bottle/50ep-agilex-demo_clean/data/episode0.hdf5

  - Inspect a specific dataset key:
      python scripts/read_hdf5.py \
        --file /data/robotwin/robotwin_data/adjust_bottle/50ep-agilex-demo_clean/data/episode0.hdf5 \
        --key some/group/dataset
"""

import argparse
import sys

try:
    import h5py
except Exception as exc:  # pragma: no cover
    print("h5py is required to run this script.\n"
          "Try: pip install h5py", file=sys.stderr)
    raise


def list_datasets(h5: "h5py.File") -> None:
    """Print all dataset keys with their shapes and dtypes."""

    def visitor(name: str, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"key: {name} | shape: {obj.shape} | dtype: {obj.dtype}")

    h5.visititems(visitor)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect HDF5 keys and shapes")
    parser.add_argument(
        "--file",
        "-f",
        default="/data/robotwin/robotwin_data/adjust_bottle/50ep-agilex-demo_clean/data/episode0.hdf5",
        help="Path to the .hdf5 file (default: episode0.hdf5 from robotwin dataset)",
    )
    parser.add_argument(
        "--key",
        "-k",
        help="Dataset key to inspect (e.g., 'group/subgroup/dataset')",
    )
    parser.add_argument(
        "--show-type",
        action="store_true",
        help=(
            "Load a small sample from the dataset and print the Python type "
            "(e.g., numpy.ndarray). Requires --key."
        ),
    )
    args = parser.parse_args()

    try:
        with h5py.File(args.file, "r") as h5:
            if args.key:
                if args.key not in h5:
                    print(f"Key not found: {args.key}", file=sys.stderr)
                    return 2
                ds = h5[args.key]
                if not isinstance(ds, h5py.Dataset):
                    print(f"Key is not a dataset: {args.key}", file=sys.stderr)
                    return 3
                print(f"key: {args.key}")
                print(f"shape: {ds.shape}")
                print(f"dtype: {ds.dtype}")
                if args.show_type:
                    # Read a tiny head-slice to infer in-memory Python type safely.
                    sample = None
                    if ds.shape == ():
                        # Scalar dataset
                        sample = ds[()]  # scalar value
                    else:
                        head = tuple(slice(0, min(2, dim)) for dim in ds.shape)
                        sample = ds[head]
                    py_type = type(sample).__name__
                    # For numpy objects, surface more detail
                    extra = ""
                    try:
                        import numpy as _np  # type: ignore
                        if isinstance(sample, _np.ndarray):
                            extra = f", np.dtype={sample.dtype}, sample_shape={sample.shape}"
                    except Exception:
                        pass
                    print(f"python_type: {py_type}{extra}")
            else:
                # Show top-level groups for quick orientation
                try:
                    top = list(h5.keys())
                except Exception:
                    top = []
                if top:
                    print(f"Top-level keys: {top}")
                print("All datasets (key -> shape, dtype):")
                list_datasets(h5)
    except FileNotFoundError:
        print(f"File not found: {args.file}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
