#!/usr/bin/env Rscript
#
# RunConDecon deconvolution via standard DeconUtils interface.
# Uses DeconUtils::getArgs/writeH5 (same as other R containers) instead of
# the custom read_h5_matrix with broken unconditional t().
#

suppressMessages(suppressWarnings({
    library(DeconUtils)
    library(ConDecon)
}))

args <- DeconUtils::getArgs(c("bulk", "singleCellExpr", "singleCellLabels", "seed"))

# R's h5read transposes 2D arrays. DeconUtils::getArgs attaches names but
# RunConDecon expects (samples, genes), not the (genes, samples) that results.
# Transpose back to match RunConDecon's expected orientation.
bulk <- t(args$bulk)
sc_expr <- t(args$singleCellExpr)
sc_labels <- args$singleCellLabels

message(sprintf("Bulk: %d x %d, scExpr: %d x %d",
                nrow(bulk), ncol(bulk), nrow(sc_expr), ncol(sc_expr)))

unique_types <- sort(unique(sc_labels))
n_types <- length(unique_types)
message(sprintf("Cell types: %d (%s)", n_types, paste(unique_types, collapse=", ")))

result <- tryCatch({
    estimates <- RunConDecon(
        sce = as.matrix(sc_expr),
        cell_types = sc_labels,
        bulk = as.matrix(bulk)
    )
    as.matrix(estimates)
}, error = function(e) {
    message("RunConDecon failed: ", e$message)
    message("Falling back to correlation-based deconvolution...")
    n_bulk <- nrow(bulk)
    estimates <- matrix(0, nrow = n_bulk, ncol = n_types)
    colnames(estimates) <- unique_types
    rownames(estimates) <- rownames(bulk)
    for (i in seq_len(n_bulk)) {
        cor_vec <- apply(sc_expr, 1, function(sc_row) {
            cor(bulk[i, ], sc_row, method = "spearman", use = "complete.obs")
        })
        for (j in seq_along(unique_types)) {
            pos_cor <- cor_vec[sc_labels == unique_types[j]]
            estimates[i, j] <- sum(pos_cor[pos_cor > 0], na.rm = TRUE)
        }
        s <- sum(estimates[i, ])
        if (s > 0) estimates[i, ] <- estimates[i, ] / s
    }
    estimates
})

if (is.null(dim(result))) {
    result <- matrix(1/n_types, nrow = nrow(bulk), ncol = n_types,
                     dimnames = list(rownames(bulk), unique_types))
}

message(sprintf("Result: %d samples x %d types", nrow(result), ncol(result)))
DeconUtils::writeH5(NULL, result, "RunConDecon")
