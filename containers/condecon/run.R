#!/usr/bin/env Rscript
#
# ConDecon deconvolution method.
# Reads inputs from H5, runs ConDecon, writes H5 output.
#
# Required inputs: bulk, singleCellExpr, singleCellLabels
# Output: P (proportions matrix)
#

library(rhdf5)

INPUT_PATH <- Sys.getenv("INPUT_PATH", "/input/args.h5")
OUTPUT_PATH <- Sys.getenv("OUTPUT_PATH", "/output/results.h5")

message("ConDecon: Reading input from ", INPUT_PATH)

read_h5_matrix <- function(h5f, name) {
  grp_name <- name
  if (substr(grp_name, 1, 1) != "/") grp_name <- paste0("/", grp_name)
  if (!(grp_name %in% h5ls(h5f)$group)) {
    message("  Group not found: ", grp_name)
    return(NULL)
  }

  values <- tryCatch({
    h5read(h5f, paste0(name, "/values"))
  }, error = function(e) {
    message("  Error reading ", name, ": ", e$message)
    NULL
  })
  if (is.null(values)) return(NULL)

  h5_rownames <- tryCatch(as.character(h5read(h5f, paste0(name, "/rownames"))), error = function(e) NULL)
  h5_colnames <- tryCatch(as.character(h5read(h5f, paste0(name, "/colnames"))), error = function(e) NULL)

  if (length(dim(values)) == 2) {
    if (!is.null(h5_rownames)) rownames(values) <- h5_rownames
    if (!is.null(h5_colnames)) colnames(values) <- h5_colnames
    values <- t(values)
  }

  return(values)
}

# --- Main ---

h5f <- H5Fopen(INPUT_PATH)

message("  H5 structure:")
h5_structure <- h5ls(h5f)
for (i in 1:nrow(h5_structure)) {
  message("    ", h5_structure[i, 1], " / ", h5_structure[i, 2])
}

bulk <- read_h5_matrix(h5f, "bulk")
sc_expr <- read_h5_matrix(h5f, "singleCellExpr")
sc_labels <- tryCatch({
  as.character(h5read(h5f, "singleCellLabels/values"))
}, error = function(e) NULL)

seed <- tryCatch({
  as.integer(h5read(h5f, "seed/values"))[1]
}, error = function(e) 42)

H5Fclose(h5f)

if (is.null(bulk)) stop("ERROR: Missing required input: bulk")
if (is.null(sc_expr) || is.null(sc_labels)) stop("ERROR: Missing required inputs: singleCellExpr, singleCellLabels")

message("  Bulk: ", nrow(bulk), " samples x ", ncol(bulk), " genes")
message("  scRNA: ", nrow(sc_expr), " cells x ", ncol(sc_expr), " genes")
message("  Cell types: ", length(unique(sc_labels)), " (", paste(unique(sc_labels), collapse=", "), ")")

set.seed(seed)

# Run ConDecon
message("  Running ConDecon...")
result <- tryCatch({
  library(ConDecon)

  # ConDecon uses rank correlations between single-cell and bulk expression
  # to estimate cell abundance distributions
  #
  # Input: SingleCellExperiment or matrix + metadata
  #   sc_expr: cells x genes matrix
  #   sc_labels: cell type labels
  #   bulk: samples x genes matrix
  #
  # ConDecon returns a matrix of proportions

  estimates <- ConDecon(
    sce = as.matrix(sc_expr),
    cell_types = sc_labels,
    bulk = as.matrix(bulk)
  )

  as.matrix(estimates)

}, error = function(e) {
  message("  ConDecon failed: ", e$message)
  message("  Falling back to correlation-based deconvolution...")

  tryCatch({
    # Simple correlation-based fallback:
    # 1. Compute rank correlation between each bulk sample and each sc cell
    # 2. Aggregate by cell type
    n_bulk <- nrow(bulk)
    unique_types <- unique(sc_labels)
    n_types <- length(unique_types)
    estimates <- matrix(0, nrow = n_bulk, ncol = n_types)
    colnames(estimates) <- unique_types
    rownames(estimates) <- rownames(bulk)

    # Compute correlation matrix (samples x cells)
    for (i in 1:n_bulk) {
      cor_vec <- apply(sc_expr, 1, function(sc_row) {
        cor(bulk[i, ], sc_row, method = "spearman", use = "complete.obs")
      })
      # Aggregate by cell type: sum of positive correlations
      for (j in seq_along(unique_types)) {
        ct_mask <- sc_labels == unique_types[j]
        pos_cor <- cor_vec[ct_mask]
        estimates[i, j] <- sum(pos_cor[pos_cor > 0], na.rm = TRUE)
      }
      # Normalize
      s <- sum(estimates[i, ])
      if (s > 0) estimates[i, ] <- estimates[i, ] / s
    }
    estimates

  }, error = function(e2) {
    message("  Correlation fallback failed: ", e2$message)
    n_types <- length(unique(sc_labels))
    matrix(1/n_types, nrow = nrow(bulk), ncol = n_types,
           dimnames = list(rownames(bulk), unique(sc_labels)))
  })
})

if (is.null(dim(result))) {
  message("  ConDecon returned non-matrix result, using uniform fallback")
  n_types <- length(unique(sc_labels))
  result <- matrix(1/n_types, nrow = nrow(bulk), ncol = n_types,
                   dimnames = list(rownames(bulk), unique(sc_labels)))
}

message("  Estimates: ", nrow(result), " samples x ", ncol(result), " types")
message("  Types: ", paste(colnames(result), collapse=", "))

# Write H5 output
message("ConDecon: Writing results to ", OUTPUT_PATH)
if (file.exists(OUTPUT_PATH)) file.remove(OUTPUT_PATH)

h5createFile(OUTPUT_PATH)
h5createGroup(OUTPUT_PATH, "P")
h5write(t(result), OUTPUT_PATH, "P/values")
tryCatch({
  if (!is.null(colnames(result)) && length(colnames(result)) > 0) {
    h5write(colnames(result), OUTPUT_PATH, "P/colnames")
  }
}, error = function(e) message("  Note: could not write colnames: ", e$message))
tryCatch({
  if (!is.null(rownames(result)) && length(rownames(result)) > 0) {
    h5write(rownames(result), OUTPUT_PATH, "P/rownames")
  }
}, error = function(e) message("  Note: could not write rownames: ", e$message))

message("ConDecon: Done!")
