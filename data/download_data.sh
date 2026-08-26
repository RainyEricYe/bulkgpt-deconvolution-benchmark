#!/usr/bin/env bash
# Download benchmark data from the HuggingFace dataset repo
#   https://huggingface.co/datasets/yeruihku/bulkgpt-data
#
# Downloads:
#   1. Real bulk:   12 H5 files (~10 GB)           -> data/2_real_bulk/
#   2. Pseudo bulk: 30 H5 files (~180 MB)          -> data/1_pseudo_bulk/{cellxgene,tabula_sapiens}/
#
# Usage (from the repository root):
#   bash data/download_data.sh            # everything
#   bash data/download_data.sh real       # real-bulk only
#   bash data/download_data.sh pseudo     # pseudo-bulk only
#   bash data/download_data.sh real sdy67 # single real-bulk dataset (smoke tests)
#
# Needs:  python + huggingface_hub  (pip install huggingface_hub)
#         If the repo is ever made private you also need a token at
#         ~/.cache/huggingface/token

set -euo pipefail
cd "$(dirname "$0")/.."

REPO="yeruihku/bulkgpt-data"
SCOPE="${1:-all}"
ONLY="${2:-}"

real_bulk() {
  local DEST="data/2_real_bulk"
  mkdir -p "$DEST"
  python3 - "$DEST" "${ONLY}" <<'PY'
import pathlib, sys, shutil
from huggingface_hub import hf_hub_download

REPO = "yeruihku/bulkgpt-data"
dest = pathlib.Path(sys.argv[1])
only = sys.argv[2] if len(sys.argv) > 2 else None
names = """sdy67 demixsc_retina sweetwater huuki_myers
altman_Arunachalam altman_Hao altman_TabulaSapiens
finotello_Hao hoek_Hao hoek_purified_Hao linsley_purified_Hao morandini_Hao""".split()
if only:
    names = [n for n in names if n == only]
for name in names:
    local = dest / f"{name}.h5"
    if local.exists() and local.stat().st_size > 0:
        print(f"  skip {name}.h5 (exists)"); continue
    print(f"  download {name}.h5 ...", flush=True)
    p = hf_hub_download(repo_id=REPO, filename=f"2_real_bulk/{name}.h5", repo_type="dataset")
    # hf_hub_download returns a symlink into the HF cache; copy the real
    # bytes so data/ is self-contained and not tied to the cache.
    shutil.copyfile(p, local)
    print(f"    -> {local}")
print("Real-bulk done.")
PY
}

pseudo_bulk() {
  local SRC_DEST="data/1_pseudo_bulk"
  mkdir -p "$SRC_DEST"
  python3 - "$SRC_DEST" <<'PY'
import pathlib, sys, shutil
from huggingface_hub import hf_hub_download

REPO = "yeruihku/bulkgpt-data"
dest = pathlib.Path(sys.argv[1])
cellxgene = """Anterior_Cingulate_Cortex Basal_Zone_Of_Heart Fimbria_Of_Uterine_Tube
Heart_Left_Ventricle Liver Middle_Temporal_Gyrus Primary_Auditory_Cortex
Primary_Somatosensory_Cortex Primary_Visual_Cortex Small_Intestine""".split()
tabula = """Bladder Blood Bone_Marrow Eye Fat Large_Intestine Liver Lung Lymph_Node
Muscle Pancreas Prostate Salivary_Gland Skin Small_Intestine Spleen Thymus
Tongue Trachea Vasculature""".split()
for sub, names in (("cellxgene", cellxgene), ("tabula_sapiens", tabula)):
    out = dest / sub
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        local = out / f"{name}.h5"
        if local.exists() and local.stat().st_size > 0:
            print(f"  skip {name}.h5 (exists)"); continue
        print(f"  download {name}.h5 ...", flush=True)
        p = hf_hub_download(repo_id=REPO, filename=f"1_pseudo_bulk/{sub}/{name}.h5", repo_type="dataset")
        # Copy real bytes (hf_hub_download may return a cache symlink).
        shutil.copyfile(p, local)
        print(f"    -> {local}")
print("Pseudo-bulk done.")
PY
}

case "$SCOPE" in
  all)    echo ">> Real bulk"; real_bulk; echo; echo ">> Pseudo bulk"; pseudo_bulk ;;
  real)   echo ">> Real bulk"; real_bulk ;;
  pseudo) echo ">> Pseudo bulk"; pseudo_bulk ;;
  *)      echo "Usage: $0 [all|real|pseudo] [real-single-dataset]" >&2; exit 2 ;;
esac

echo
echo "Done. Verify:"
echo "  real:   ls data/2_real_bulk/*.h5        (12 files)"
echo "  pseudo: ls data/1_pseudo_bulk/cellxgene/*.h5 + tabula_sapiens/*.h5  (30 files)"