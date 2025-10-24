"""Fix parquet metadata to replace 'List' with 'Sequence' feature type.

This script fixes the HuggingFace feature metadata in parquet files by replacing
unsupported 'List' type with 'Sequence' type.
"""

import json
import pathlib
import pyarrow.parquet as pq


def fix_parquet_metadata(parquet_path: pathlib.Path) -> None:
    """Fix the HuggingFace metadata in a parquet file."""
    # Read the parquet file
    parquet_file = pq.ParquetFile(parquet_path)

    # Get the schema with metadata
    schema = parquet_file.schema_arrow

    # Get and parse the HuggingFace metadata
    metadata = schema.metadata
    if b'huggingface' not in metadata:
        print(f"No HuggingFace metadata found in {parquet_path}")
        return

    hf_metadata = json.loads(metadata[b'huggingface'])

    # Fix the feature types: replace 'List' with 'Sequence'
    if 'info' in hf_metadata and 'features' in hf_metadata['info']:
        features = hf_metadata['info']['features']
        modified = False

        for key, feature in features.items():
            if isinstance(feature, dict) and feature.get('_type') == 'List':
                print(f"  Fixing feature '{key}': List -> Sequence")
                feature['_type'] = 'Sequence'
                modified = True

        if not modified:
            print(f"  No 'List' types found in {parquet_path}")
            return

        # Update the metadata
        new_metadata = dict(metadata)
        new_metadata[b'huggingface'] = json.dumps(hf_metadata).encode('utf-8')

        # Create new schema with updated metadata
        new_schema = schema.with_metadata(new_metadata)

        # Read the table and write back with new schema
        table = parquet_file.read()
        table = table.cast(new_schema)

        # Write back to the same file
        pq.write_table(table, parquet_path)
        print(f"  ✓ Fixed {parquet_path}")
    else:
        print(f"  No features found in metadata for {parquet_path}")


def main(dataset_path: str) -> None:
    """Fix all parquet files in the dataset."""
    dataset_dir = pathlib.Path(dataset_path)

    # Find all parquet files
    parquet_files = sorted(dataset_dir.rglob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {dataset_path}")
        return

    print(f"Found {len(parquet_files)} parquet files")
    print("Fixing metadata...\n")

    for parquet_file in parquet_files:
        print(f"Processing: {parquet_file.relative_to(dataset_dir)}")
        try:
            fix_parquet_metadata(parquet_file)
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\nDone!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python fix_parquet_metadata.py <dataset_path>")
        sys.exit(1)

    main(sys.argv[1])
    # python scripts/fix_parquet_metadata.py "/home/jovyan/repo/Pi/data/beat_block_hammer-50ep-agilex-demo_clean"
