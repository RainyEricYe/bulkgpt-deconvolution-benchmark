"""Gene symbol to Ensembl ID mapping for TranscriptFormer."""
from pathlib import Path
from typing import Dict, List, Optional


def map_symbol_to_ensembl(
    gene_symbols: List[str],
    mapping_path: Optional[str] = None,
) -> Dict[str, str]:
    """Map gene symbols to Ensembl IDs.

    Uses a local pickle mapping first (e.g. Geneformer's gene_name_id_dict.pkl),
    then falls back to the mygene.info API for unmapped symbols.

    Args:
        gene_symbols: List of gene symbols to map.
        mapping_path: Path to a pickle dict mapping symbol -> Ensembl ID.

    Returns:
        dict mapping each input symbol -> Ensembl ID (or original symbol if unmappable)
    """
    local_map: Dict[str, str] = {}
    if mapping_path is None:
        print("No local gene mapping provided — skipping local map, using mygene.info API only")

    if mapping_path:
        p = Path(mapping_path)
        if p.exists():
            import pickle, json
            try:
                with open(p, "rb") as f:
                    local_map = pickle.load(f)
            except Exception:
                with open(p) as f:
                    local_map = {k: v for k, v in json.load(f).items() if v is not None}
            print(f"Loaded local gene mapping: {len(local_map)} entries from {mapping_path}")
        else:
            print(f"Local gene mapping not found: {mapping_path}")

    known: set = set()
    unknown: list = []
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
                unknown, scopes="symbol", fields="ensembl.gene",
                species="human", returnall=True, batch_size=1000,
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
    print(f"Mapped {n_mapped}/{len(gene_symbols)} gene symbols to Ensembl IDs ({', '.join(sources)})" if sources else f"Mapped {n_mapped}/{len(gene_symbols)} gene symbols to Ensembl IDs")
    return mapping
