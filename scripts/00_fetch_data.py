"""Step 1 — fetch a CELLxGENE dataset and cache it locally.

Usage:
  python scripts/00_fetch_data.py                       # primary RA PBMC -> data/raw.h5ad
  python scripts/00_fetch_data.py <dataset_id> <out>    # any dataset

Primary   RA PBMC       d18736c3-6292-4379-919a-d6d973204c87  (~108,717 cells)
Backup    COVID PBMC    2a498ace-872a-4935-984b-1afa70fd9886  (~340k cells, 3 conditions)
"""
import sys
from pathlib import Path

import cellxgene_census

CENSUS_VERSION = "2025-11-08"
PRIMARY_ID = "d18736c3-6292-4379-919a-d6d973204c87"
ROOT = Path(__file__).resolve().parents[1]


def fetch(dataset_id: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[skip] cache exists: {out} ({out.stat().st_size/1e6:.0f} MB)")
        return
    print(f"[fetch] census {CENSUS_VERSION}, dataset {dataset_id} ...")
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        adata = cellxgene_census.get_anndata(
            census, organism="homo_sapiens",
            obs_value_filter=f"dataset_id == '{dataset_id}'",
        )
    print(f"[ok] {adata.shape[0]} cells x {adata.shape[1]} genes")
    adata.write_h5ad(out)
    print(f"[write] {out} ({out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        fetch(sys.argv[1], Path(sys.argv[2]))
    else:
        fetch(PRIMARY_ID, ROOT / "data" / "raw.h5ad")
