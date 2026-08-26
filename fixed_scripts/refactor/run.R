args <- DeconUtils::getArgs(c("bulk", "nCellTypes"))

source("/code/ReFACTor.R")

#P is relative
P <- refactor(args$bulk, args$nCellTypes)$refactor_components

# ── Post-process: Transform PC scores to proportions ──
# ReFACTor outputs PCA component scores (unbounded, e.g. PC1=-38..+53).
# Hungarian matching in evaluate.py maps PC1..PCn to cell types using
# Pearson correlation (invariant under linear per-column transforms).
# Min-max per component + row-normalize converts to [0,1] proportions.
min_vals <- apply(P, 2, min)
max_vals <- apply(P, 2, max)
col_range <- pmax(max_vals - min_vals, 1e-10)
P <- sweep(sweep(P, 2, min_vals, "-"), 2, col_range, "/")
row_sum <- pmax(rowSums(P), 1e-10)
P <- P / row_sum

rownames(P) <- colnames(args$bulk)

DeconUtils::writeH5(NULL, P, "ReFACTor")
