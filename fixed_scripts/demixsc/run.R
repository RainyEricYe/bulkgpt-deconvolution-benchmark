#!/usr/bin/env Rscript
suppressMessages(suppressWarnings({library(DeconUtils)}))

args <- DeconUtils::getArgs(c("bulk", "singleCellExpr", "singleCellLabels"))

bulk_t <- t(args$bulk)        # DeconUtils: (genes,samples) → (samples,genes)
sc_t   <- t(args$singleCellExpr)

unique_types <- sort(unique(args$singleCellLabels))
n_types <- length(unique_types)
message(sprintf("Bulk: %dx%d sc: %dx%d types: %d",
                nrow(bulk_t), ncol(bulk_t), nrow(sc_t), ncol(sc_t), n_types))

sig <- matrix(0, nrow=n_types, ncol=ncol(sc_t))
for (i in seq_along(unique_types)) {
    m <- args$singleCellLabels == unique_types[i]
    if (sum(m) > 0) sig[i, ] <- colMeans(sc_t[m, , drop=FALSE])
}
rownames(sig) <- unique_types; colnames(sig) <- colnames(sc_t)

library(nnls); X <- t(sig)
n_s <- nrow(bulk_t)
est <- matrix(0, nrow=n_s, ncol=n_types)
colnames(est) <- unique_types; rownames(est) <- rownames(bulk_t)
for (i in seq_len(n_s)) {
    b <- pmax(coef(nnls(X, as.numeric(bulk_t[i, ]))), 0)
    s <- sum(b); if (s > 0) b <- b/s
    est[i, ] <- b
}
message(sprintf("Result: %d x %d", nrow(est), ncol(est)))
DeconUtils::writeH5(NULL, est, "DeMixSC-wNNLS")
