"""Step 1 — fetch the primary CELLxGENE dataset and cache it locally.

RA PBMC scRNA-seq: dataset_id d18736c3-6292-4379-919a-d6d973204c87
(~108,717 cells, 36 donors [18 RA / 18 healthy], 15 immune cell types, single 10x 3' v3 assay).

Writes data/raw.h5ad (gitignored). Idempotent: skips the download if the cache exists.
"""
from pathlib import Path

import cellxgene_census

DATASET_ID = "d18736c3-6292-4379-919a-d6d973204c87"
CENSUS_VERSION = "2025-11-08"  # pinned for reproducibility (per take-home)
OUT = Path(__file__).resolve().parents[1] / "data" / "raw.h5ad"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"[skip] cache already exists: {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")
        return

    print(f"[fetch] census {CENSUS_VERSION}, dataset {DATASET_ID} ...")
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism="homo_sapiens",
            obs_value_filter=f"dataset_id == '{DATASET_ID}'",
        )

    print(f"[ok] fetched AnnData: {adata.shape[0]} cells x {adata.shape[1]} genes")
    adata.write_h5ad(OUT)
    print(f"[write] {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
