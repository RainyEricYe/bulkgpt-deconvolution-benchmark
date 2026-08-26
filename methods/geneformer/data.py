"""Geneformer-specific data functions: gene mapping and rank-value encoding."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from anndata import AnnData


def map_symbol_to_ensembl(
    gene_symbols: list[str],
    model_dir: str | None = None,
) -> Dict[str, str]:
    """Map gene symbols to Ensembl IDs.

    Uses the Geneformer pretrained model's local ``gene_name_id_dict.pkl`` when
    *model_dir* is provided (avoids network I/O).  Symbols not found in the
    local dictionary are looked up via the mygene.info API (requires internet
    access; may require HTTP(S)_PROXY).  Unmappable symbols keep the original
    name.

    Args:
        gene_symbols: List of gene symbols to map.
        model_dir: Path to the Geneformer pretrained model directory containing
            ``dicts/gene_name_id_dict.pkl``.  When given, the local file is
            used as the primary mapping source.

    Returns:
        dict mapping each input symbol -> Ensembl ID (or original symbol if
        unmappable)
    """
    local_map: Dict[str, str] = {}
    if model_dir is not None:
        # The gene_name_id_dict.pkl may be at {model_dir}/dicts/ or
        # at {parent_of_model_dir}/dicts/ (common when model_dir is a
        # at {parent_of_model_dir}/dicts/ (common when model_dir is a
        # subdirectory like "default" or "Geneformer-V2-104M").
        candidates = [
            Path(model_dir) / "dicts" / "gene_name_id_dict.pkl",
            Path(model_dir).parent / "dicts" / "gene_name_id_dict.pkl",
            # Follow symlink (weights/geneformer/default -> external dir)
            Path(model_dir).resolve().parent / "dicts" / "gene_name_id_dict.pkl",
        ]
        local_path = next((p for p in candidates if p.exists()), None)
        if local_path is not None:
            with open(local_path, "rb") as f:
                import pickle
                local_map = pickle.load(f)
            print(f"Loaded local gene mapping: {len(local_map)} entries from {local_path}")
        else:
            print(f"Local gene mapping not found (checked {candidates}), will use mygene.info")

    # Separate known and unknown symbols
    known: set[str] = set()
    unknown: list[str] = []
    for s in gene_symbols:
        k = str(s)
        if k in local_map:
            known.add(k)
        else:
            unknown.append(k)

    mapping: Dict[str, str] = {}
    for k in known:
        mapping[k] = local_map[k]

    if unknown:
        try:
            import mygene

            mg = mygene.MyGeneInfo()
            results = mg.querymany(
                unknown,
                scopes="symbol",
                fields="ensembl.gene",
                species="human",
                returnall=True,
                batch_size=1000,
            )

            for r in results["out"]:
                query = r["query"]
                ensg = r.get("ensembl", {})
                if isinstance(ensg, dict):
                    ensg_id = ensg.get("gene", "")
                elif isinstance(ensg, str):
                    ensg_id = ensg
                else:
                    ensg_id = ""
                mapping[query] = ensg_id if ensg_id else query

        except Exception as exc:
            print(f"mygene.info lookup failed ({exc}), keeping original symbols for {len(unknown)} unmapped genes")

        for q in unknown:
            if q not in mapping:
                mapping[q] = q

    n_mapped = sum(1 for k, v in mapping.items() if v != k and v.startswith("ENSG"))
    n_local = sum(1 for k in known if mapping.get(k, k) != k and mapping[k].startswith("ENSG"))
    n_remote = n_mapped - n_local
    sources = []
    if n_local:
        sources.append(f"{n_local} local")
    if n_remote:
        sources.append(f"{n_remote} via mygene")
    if not sources and not n_mapped:
        sources.append("none")
    print(f"Mapped {n_mapped}/{len(gene_symbols)} gene symbols to Ensembl IDs ({', '.join(sources)})")
    return mapping


import os as _os

def load_geneformer_ensembl_mapping(
    mapping_path: str | None = None,
) -> Dict[str, str]:
    if mapping_path is None:
        # Priority: env var > weights/ > repo-local > legacy
        _root = Path(__file__).resolve().parents[2]  # to_publish/
        _weights_p = _root / "weights" / "geneformer" / "ensembl_mapping_dict_gc104M.pkl"
        _repo_p = _root / "data" / "pretrained" / "geneformer" / "ensembl_mapping_dict_gc104M.pkl"
        mapping_path = _os.environ.get(
            "GENEORMER_MAPPING_PATH",
            str(_weights_p) if _weights_p.exists() else str(_repo_p),
        )
    """Load the Geneformer ensembl mapping dictionary.

    Covers 173k+ entries for the Geneformer V2 gc104M vocabulary.
    """
    import pickle
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)
    print(f"Loaded Geneformer ensembl mapping: {len(mapping)} entries")
    return mapping


def rank_value_encode(
    expr_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank-value encoding: sort genes by expression within each cell.

    For each cell (row), genes are sorted by expression value descending.
    Returns the sorted expression values and per-cell sorting indices
    (to reorder gene IDs correspondingly).

    Returns:
        sorted_expr: (n_cells, n_genes) expression sorted descending per row
        sorted_indices: (n_cells, n_genes) column indices that sort each row
    """
    sorted_indices = np.argsort(-expr_matrix, axis=1)
    sorted_expr = np.take_along_axis(expr_matrix, sorted_indices, axis=1)

    n_genes = expr_matrix.shape[1]
    if n_genes > 0:
        print(f"Rank-value encoded {expr_matrix.shape[0]} cells x {n_genes} genes "
              f"(sorted by expression per cell)")

    return sorted_expr, sorted_indices
