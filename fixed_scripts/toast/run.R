# Fixed TOAST run.R
# TOAST's MDeconv expects SelMarker as a named list of cell-type-specific markers.
# Use TOAST's built-in ChooseMarker() for optimal marker selection.
args <- DeconUtils::getArgs(c("bulk", "markers", "signature"))

suppressMessages(
  suppressWarnings({
    library(TOAST)
  })
)

sig_mat <- args$signature  # (n_genes, n_cell_types) with rownames=genes, colnames=types

# Use TOAST's ChooseMarker to select cell-type-specific markers from signature
sel_marker <- tryCatch({
    ChooseMarker(sig_mat, colnames(sig_mat), nMarkCT = 30, chooseSig = TRUE)
}, error = function(e) {
    # Manual fallback: select top markers by fold-change
    n_genes <- nrow(sig_mat)
    n_types <- ncol(sig_mat)
    n_markers <- min(30, max(5, floor(n_genes / n_types)))
    ct_markers <- list()
    for (ct in colnames(sig_mat)) {
        ct_expr <- sig_mat[, ct]
        other_expr <- rowMeans(sig_mat[, setdiff(colnames(sig_mat), ct), drop = FALSE])
        fc <- ct_expr / (other_expr + 1e-10)
        top_genes <- names(sort(fc, decreasing = TRUE))[1:n_markers]
        ct_markers[[ct]] <- top_genes
    }
    ct_markers
})

estProp_RF <- MDeconv(Ymat = args$bulk, SelMarker = sel_marker,
        epsilon = 1e-3, verbose = FALSE)$H
estProp_RF <- t(estProp_RF)

DeconUtils::writeH5(NULL, estProp_RF, "TOAST")
