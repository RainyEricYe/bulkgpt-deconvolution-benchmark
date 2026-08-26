# hspe (dtangle2) deconvolution method
# R h5read transposes 2D data: all matrices are (genes, samples)
args <- DeconUtils::getArgs(c("bulk", "singleCellExpr", "singleCellLabels"))

suppressMessages(suppressWarnings(library(dtangle)))

# Transpose from (genes, samples) to (samples, genes) for dtangle
sc_expr <- t(args$singleCellExpr)  # (n_cells, n_genes)
bulk_expr <- t(args$bulk)          # (n_bulk, n_genes)
Y <- rbind(sc_expr, bulk_expr)     # (n_cells + n_bulk, n_genes)

# Build pure_samples list
sc_labels <- as.character(args$singleCellLabels)
unique_types <- unique(sc_labels)
pure_samples <- lapply(unique_types, function(ct) which(sc_labels == ct))

set.seed(42)

result <- tryCatch({
    dtangle2(Y = Y, pure_samples = pure_samples,
             data_type = "rna-seq", marker_method = "ratio",
             sto = TRUE, verbose = FALSE)
}, error = function(e) {
    dtangle(Y = Y, pure_samples = pure_samples,
            data_type = "rna-seq", marker_method = "ratio")
})

# Extract bulk estimates
n_sc <- nrow(sc_expr)
n_bulk <- nrow(bulk_expr)
estimates <- result$estimates[(n_sc + 1):(n_sc + n_bulk), , drop = FALSE]

# Ensure dimnames are set (dtangle output may lack them)
if (is.null(colnames(estimates))) {
    colnames(estimates) <- unique_types
}
if (is.null(rownames(estimates))) {
    rownames(estimates) <- colnames(args$bulk)
}

# Write output directly
suppressMessages(suppressWarnings(library(rhdf5)))
OUTPUT_PATH <- Sys.getenv("OUTPUT_PATH", "/output/results.h5")
if (file.exists(OUTPUT_PATH)) file.remove(OUTPUT_PATH)
h5createFile(OUTPUT_PATH)
h5createGroup(OUTPUT_PATH, "P")
h5write(t(estimates), OUTPUT_PATH, "P/values")
h5write(colnames(estimates), OUTPUT_PATH, "P/colnames")
h5write(rownames(estimates), OUTPUT_PATH, "P/rownames")
cat("hspe Writing results to ", OUTPUT_PATH, "\n")
