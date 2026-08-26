# Fixed DSA run.R
# DSA's EstimateWeight expects a named list of gene vectors (one per cell type).
# The H5 stores markers as comma-separated strings per cell type -- convert.
args <- DeconUtils::getArgs(c("bulk", "markers", "signature"))

library(DSA)

mix <- args$bulk  # (n_genes, n_samples) in R after h5read transpose

cat("=== DSA Debug ===\n")
cat("Bulk dim:", nrow(mix), "x", ncol(mix), "\n")
cat("Bulk rownames (first 5):", rownames(mix)[1:5], "\n")

# --- Build a proper named list of marker gene vectors ---
marker_list <- NULL

# Markers come from H5 as a named character vector where each element
# is a comma-separated list of gene names for one cell type.
if (!is.null(args$markers) && is.character(args$markers) && !is.null(names(args$markers))) {
    ct_names <- names(args$markers)
    cat("Markers: named character vector with", length(ct_names), "cell types\n")
    marker_list <- lapply(ct_names, function(nm) {
        genes <- strsplit(args$markers[nm], ",")[[1]]
        genes[nzchar(genes)]
    })
    names(marker_list) <- ct_names
    n_total <- sum(sapply(marker_list, length))
    cat("  Total marker genes:", n_total, "\n")
    cat("  First CT markers (up to 5):", marker_list[[1]][1:min(5, length(marker_list[[1]]))], "\n")
}

# If H5 didn't have proper markers, construct from signature matrix
if (is.null(marker_list) && !is.null(args$signature)) {
    cat("Constructing markers from signature matrix...\n")
    sig_mat <- args$signature  # (sig_genes, cell_types)
    cell_types <- colnames(sig_mat)
    n_types <- length(cell_types)
    n_genes <- nrow(sig_mat)
    n_markers_per_type <- min(100, max(1, floor(n_genes / n_types)))
    cat("  Signature dim:", n_genes, "x", n_types, "\n")
    cat("  Markers per type:", n_markers_per_type, "\n")

    marker_list <- list()
    for (ct in cell_types) {
        ct_expr <- sig_mat[, ct]
        other_expr <- rowMeans(sig_mat[, setdiff(colnames(sig_mat), ct), drop = FALSE])
        fc <- ct_expr / (other_expr + 1e-10)
        top_genes <- names(sort(fc, decreasing = TRUE))[1:n_markers_per_type]
        marker_list[[ct]] <- top_genes
    }
}

if (is.null(marker_list) || length(marker_list) == 0) {
    stop("No usable markers found")
}

# --- EstimateWeight with proper named list ---
cat("\n=== EstimateWeight ===\n")
cat("Marker list has", length(marker_list), "cell types\n")
for (nm in names(marker_list)) {
    cat("  ", nm, ":", length(marker_list[[nm]]), "marker genes\n")
}

ew <- tryCatch({
    EstimateWeight(mix, marker_list, method = "LM")
}, error = function(e) {
    cat("EstimateWeight failed:", e$message, "\n")
    NULL
})

if (is.null(ew) || is.null(ew$weight)) {
    stop("EstimateWeight returned NULL or no weight matrix")
}

cat("EstimateWeight succeeded\n")
cat("Weight dim:", nrow(ew$weight), "x", ncol(ew$weight), "\n")
# ew$weight should be (n_cell_types, n_samples)
cat("Weight rownames:", paste(rownames(ew$weight), collapse = ", "), "\n")

# --- Deconvolution ---
# Deconvolution expects: data (n_genes, n_samples), weight (n_samples, n_cell_types)
cat("\n=== Deconvolution ===\n")
prop <- t(ew$weight)  # (n_samples, n_cell_types)
cat("Proportions dim:", nrow(prop), "x", ncol(prop), "\n")

deconv <- Deconvolution(mix, prop)
# deconv has dim (n_genes, n_cell_types) -- cell-type-specific expression (signature)
cat("Deconvolution result dim:", nrow(deconv), "x", ncol(deconv), "\n")

# Set names
colnames(deconv) <- colnames(prop) <- names(marker_list)
rownames(deconv) <- rownames(mix)

# writeH5(S, P, method) where S=signature, P=proportions
cat("\n=== Writing results ===\n")
cat("S (signature) dim:", nrow(deconv), "x", ncol(deconv), "\n")
cat("P (proportions) dim:", nrow(prop), "x", ncol(prop), "\n")
DeconUtils::writeH5(deconv, prop, "DSA")
cat("Done.\n")
