# Fixed MCPcounter run.R
# MCPcounter expects markers as a named list (cell type -> character vector of genes).
# The runner passes a flat marker list from select_marker_genes, which causes
# MCPcounter to treat each gene as a separate "population" and return the raw
# expression matrix instead of cell-type scores.
#
# Fix: Use the signature matrix (passed via the "signature" input) to derive
# cell-type-specific marker genes (top 50 highest expressed per cell type).
args <- DeconUtils::getArgs(c("bulk", "signature"))

library(MCPcounter)

args$bulk <- log2(args$bulk + 1)

# Build markers as a named list from the signature matrix
# For each cell type, select the top 50 most highly expressed genes
n_markers <- min(50, nrow(args$signature))
marker_list <- list()
for (ct in colnames(args$signature)) {
    col_expr <- args$signature[, ct]
    top_genes <- names(sort(col_expr, decreasing = TRUE))[1:n_markers]
    marker_list[[ct]] <- top_genes
}

P <- MCPcounter.estimate(args$bulk, marker_list)
DeconUtils::writeH5(NULL, t(P), "MCPcounter")
