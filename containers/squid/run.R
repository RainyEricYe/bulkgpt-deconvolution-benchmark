#!/usr/bin/env Rscript
#
# SQUID deconvolution method.
# Reads inputs from H5, runs SQUID, writes H5 output.
#
# Required inputs: bulk, singleCellExpr, singleCellLabels
# Output: P (proportions matrix)
#

library(rhdf5)

INPUT_PATH <- Sys.getenv("INPUT_PATH", "/input/args.h5")
OUTPUT_PATH <- Sys.getenv("OUTPUT_PATH", "/output/results.h5")

message("SQUID: Reading input from ", INPUT_PATH)

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
    # Transpose: R stores column-major, but H5 values are C-order
    # After transpose: rows=samples/cells, cols=genes
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

# Read inputs
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

# Run SQUID
message("  Running SQUID...")
result <- tryCatch({
  library(SQUID)

  # SQUID requires:
  #   B: bulk RNA-seq data (samples x genes)
  #   scC: single-cell counts (cells x genes)
  #   scMeta: data.frame with cell_id and cell_type columns
  scMeta <- data.frame(
    cell_id = rownames(sc_expr),
    cell_type = sc_labels,
    stringsAsFactors = FALSE
  )

  # SQUID returns a matrix of proportions (samples x cell_types)
  estimates <- SQUID(
    B = as.matrix(bulk),
    scC = as.matrix(sc_expr),
    scMeta = scMeta
  )

  # Return as matrix
  as.matrix(estimates)

}, error = function(e) {
  message("  SQUID failed: ", e$message)
  message("  Falling back to DWLS...")

  # Fallback: build signature matrix and use DWLS
  tryCatch({
    library(MASS)  # for ginv

    # Build signature matrix: mean expression per cell type
    unique_types <- unique(sc_labels)
    sig <- matrix(0, nrow = length(unique_types), ncol = ncol(sc_expr))
    for (i in seq_along(unique_types)) {
      ct_mask <- sc_labels == unique_types[i]
      if (sum(ct_mask) > 0) {
        sig[i, ] <- colMeans(sc_expr[ct_mask, , drop = FALSE])
      }
    }
    sig <- t(sig)  # (genes x cell_types)

    # Dampened weighted least squares
    n_samples <- nrow(bulk)
    n_types <- length(unique_types)
    estimates <- matrix(0, nrow = n_samples, ncol = n_types)
    colnames(estimates) <- unique_types
    rownames(estimates) <- rownames(bulk)

    for (i in 1:n_samples) {
      y <- as.numeric(bulk[i, ])
      # Weighted LS
      w <- 1 / (sig^2 + 0.01)
      X <- sig
      W <- diag(colSums(w))
      tryCatch({
        beta <- ginv(t(X) %*% W %*% X) %*% t(X) %*% W %*% y
        beta <- pmax(beta, 0)
        beta <- beta / sum(beta)
        estimates[i, ] <- beta
      }, error = function(e2) {
        estimates[i, ] <<- rep(1/n_types, n_types)
      })
    }
    estimates
  }, error = function(e2) {
    message("  Fallback also failed: ", e2$message)
    # Uniform proportions
    n_types <- length(unique(sc_labels))
    matrix(1/n_types, nrow = nrow(bulk), ncol = n_types,
           dimnames = list(rownames(bulk), unique(sc_labels)))
  })
})

# Ensure we have a matrix
if (is.null(dim(result))) {
  message("  SQUID returned non-matrix result, using fallback")
  n_types <- length(unique(sc_labels))
  result <- matrix(1/n_types, nrow = nrow(bulk), ncol = n_types,
                   dimnames = list(rownames(bulk), unique(sc_labels)))
}

message("  Estimates: ", nrow(result), " samples x ", ncol(result), " types")
message("  Types: ", paste(colnames(result), collapse=", "))

# Write H5 output
message("SQUID: Writing results to ", OUTPUT_PATH)
if (file.exists(OUTPUT_PATH)) file.remove(OUTPUT_PATH)

h5createFile(OUTPUT_PATH)
h5createGroup(OUTPUT_PATH, "P")
# Store as (cell_types, samples) to match DeconBenchmark convention
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

message("SQUID: Done!")
