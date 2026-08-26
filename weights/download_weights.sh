#!/usr/bin/env bash
# Download pretrained backbone weights for the frozen-embedding methods.
#
# Populates weights/ with the checkpoints referenced by core/deconv/frozen_paths.py.
# Every path can instead be overridden via its eponymous environment variable,
# so this script is optional — see weights/README.md.
#
# Usage:  bash weights/download_weights.sh   (from the repository root)

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p weights

# scGPT whole-human -----------------------------------------------------------
if [ ! -d weights/scgpt/whole-human ]; then
  echo ">> scGPT whole-human"
  git clone https://huggingface.co/scGPT/whole-human weights/scgpt/whole-human
fi

# Geneformer ----------------------------------------------------------------
if [ ! -d weights/geneformer/default ]; then
  echo ">> Geneformer"
  git clone https://huggingface.co/ctheodoris/Geneformer weights/geneformer/default
fi

# TranscriptFormer ----------------------------------------------------------
if [ ! -d weights/transcriptformer/repo ]; then
  echo ">> TranscriptFormer source (clone into weights/transcriptformer/repo)"
  git clone https://github.com/zcslab/TranscriptFormer weights/transcriptformer/repo
fi
if [ ! -d weights/transcriptformer/tf_sapiens ]; then
  echo ">> TranscriptFormer checkpoint (place in weights/transcriptformer/tf_sapiens)"
  echo "   Not yet automated — see weights/README.md for the HF source."
fi

# BulkFormer ----------------------------------------------------------------
if [ ! -d weights/bulkformer/source ]; then
  echo ">> BulkFormer source"
  git clone https://github.com/CompGenomeMed/bulkformer weights/bulkformer/source
fi

echo
echo "Done. scFoundation and STACK checkpoints are large and not yet automated;"
echo "see weights/README.md for their source locations, or set the env vars:"
echo "  SCFOUNDATION_CKPT, SCFOUNDATION_GENE_INDEX, STACK_CHECKPOINT, STACK_GENELIST"
