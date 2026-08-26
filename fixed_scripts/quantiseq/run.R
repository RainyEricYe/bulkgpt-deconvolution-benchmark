# Fixed quanTIseq run.R
# The original run.R passes bulk and signature to quanTIseq without TPM
# normalization. fixMixture() (in DOCKER_codes.R) does TPM normalization but
# internally calls mapGenes() which reads an HGNC file at
# /opt/quantiseq/deconvolution/HGNC_genenames_20170418.txt that does not exist
# in the DeconBenchmark container. So we apply TPM normalization directly.
args <- DeconUtils::getArgs(c("bulk", "signature"))

suppressMessages(
  suppressWarnings({
      source("/code/DOCKER_codes.R")
    })
)

# Apply TPM normalization directly (bypass fixMixture's mapGenes which
# requires a HGNC file not present in this container)
tpm_normalize <- function(mat) {
    # Do NOT un-log (our data is raw counts/sums, not log2)
    # TPM normalize: each column sums to 1e6
    col_sums <- apply(mat, 2, sum)
    col_sums <- pmax(col_sums, 1)  # avoid division by zero
    t(t(mat) * 1e6 / col_sums)
}

bulk_norm <- tpm_normalize(args$bulk)
sig_norm <- tpm_normalize(args$signature)

P <- quanTIseq(sig_norm, bulk_norm, scaling=rep(1, ncol(sig_norm)), method="lsei")

DeconUtils::writeH5(NULL, P, "quanTIseq")
