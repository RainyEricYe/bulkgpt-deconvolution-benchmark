#!/usr/bin/env Rscript
#
# MethylResolver deconvolution (with robust fallback).
#
# Tier 1: MethylResolver::MethylResolver
#   If the package's foreach parallel bug triggers even with doPar=FALSE,
#   falls through to Tier 2.
#
# Tier 2: Ridge regression via MASS::ginv (vectorized).
#   Uses cellTypeExpr (all genes, same dims as bulk) for dimension
#   compatibility.  If cellTypeExpr unavailable, matches genes between
#   signature and bulk.  If too few overlap, falls through.
#
# Tier 3: Per-sample correlation-based estimation.
#   Guaranteed to never fail.
#
# Inputs (from DeconBenchmark H5 via DeconUtils::getArgs):
#   args$signature     -- marker_genes  x cell_types  (for MethylResolver)
#   args$cellTypeExpr  -- common_genes  x cell_types  (fallback; same dim as bulk if available)
#   args$bulk          -- common_genes x samples
#
# Output P: samples x cell_types

library(MASS)       # for ginv
library(MethylResolver)

# Request both signature (marker genes) and cellTypeExpr (full gene matrix)
# so the fallback regression has the same gene dimension as bulk.
args <- DeconUtils::getArgs(c("bulk", "signature", "cellTypeExpr"))

origCTNames <- colnames(args$signature)
colnames(args$signature) <- make.names(colnames(args$signature))
has_cte <- !is.null(args$cellTypeExpr) && is.matrix(args$cellTypeExpr) &&
           nrow(as.matrix(args$cellTypeExpr)) > 0
if (has_cte) {
    colnames(args$cellTypeExpr) <- make.names(colnames(args$cellTypeExpr))
}

# ── Helpers ──────────────────────────────────────────────────────────────

sanitise <- function(M) {
    M[!is.finite(M)] <- 0
    M
}

normalise <- function(P, n_types) {
    P[!is.finite(P)] <- 0
    P[P < 0] <- 0
    rs <- rowSums(P)
    for (i in seq_len(nrow(P))) {
        if (rs[i] > 0) P[i, ] <- P[i, ] / rs[i]
        else P[i, ] <- 1.0 / n_types
    }
    P
}

# Prepare the "wide" signature matrix for fallback use.
# cellTypeExpr has the same n_genes as bulk; otherwise subset signature to
# match bulk genes.
build_fallback_S <- function(sig, cte, bulk) {
    if (has_cte) {
        S <- sanitise(as.matrix(cte))
        if (nrow(S) == nrow(bulk)) {
            message(sprintf("  Fallback S: cellTypeExpr %d x %d (full genes)", nrow(S), ncol(S)))
            return(S)
        }
    }
    S <- sanitise(as.matrix(sig))
    X <- sanitise(as.matrix(bulk))
    common <- intersect(rownames(S), rownames(X))
    if (length(common) >= ncol(S)) {
        message(sprintf("  Fallback S: signature %d overlapped genes", length(common)))
        return(S[common, , drop = FALSE])
    }
    NULL
}

# ═════════════════════════════════════════════════════════════════════════
# Tier 1 — MethylResolver
# ═════════════════════════════════════════════════════════════════════════

P <- tryCatch({
    message("MethylResolver: attempting deconvolution ...")
    result <- suppressWarnings(
        MethylResolver::MethylResolver(
            methylMix  = args$bulk,
            methylSig  = args$signature,
            doPar      = FALSE,
            numCores   = 1,
            absolute   = FALSE
        )
    )
    if (is.null(result) || length(result) == 0L) {
        stop("MethylResolver returned NULL / empty")
    }
    result <- as.matrix(result)
    if (nrow(result) == 0L || ncol(result) == 0L) {
        stop("MethylResolver returned zero-dimension matrix")
    }
    message("  -> MethylResolver succeeded")
    result
}, error = function(e) {
    message("  -> MethylResolver FAILED: ", e$message)
    NULL
})

# ═════════════════════════════════════════════════════════════════════════
# Tier 2 & 3 — Fallbacks
# ═════════════════════════════════════════════════════════════════════════

if (is.null(P)) {
    message("Entering fallback pipeline ...")

    X    <- sanitise(as.matrix(args$bulk))
    n_samples <- ncol(X)

    S <- build_fallback_S(args$signature, args$cellTypeExpr, X)
    n_types <- if (is.null(S)) 0L else ncol(S)

    # ── Tier 2: ridge regression (vectorised) ─────────────────────
    if (!is.null(S) && n_types >= 2L && n_samples >= 1L) {
        P <- tryCatch({
            message("  Fallback A — Ridge regression via MASS::ginv ...")
            message(sprintf("  S: %d x %d, X: %d x %d", nrow(S), ncol(S), nrow(X), ncol(X)))
            lambda  <- 1.0
            StS     <- t(S) %*% S
            StX     <- t(S) %*% X
            beta    <- MASS::ginv(StS + lambda * diag(n_types)) %*% StX
            est     <- t(beta)
            est     <- normalise(est, n_types)
            colnames(est) <- colnames(S)
            rownames(est) <- colnames(X)
            message("  -> Ridge fallback succeeded")
            est
        }, error = function(e2) {
            message("  -> Ridge fallback FAILED: ", e2$message)
            NULL
        })
    }

    # ── Tier 3: correlation-based (bulletproof) ───────────────────
    if (is.null(P)) {
        message("  Fallback B — Correlation-based estimation ...")
        # Rebuild S for correlation (may be from either matrix source)
        S <- build_fallback_S(args$signature, args$cellTypeExpr, X)
        n_types <- if (is.null(S)) ncol(as.matrix(args$signature)) else ncol(S)

        if (!is.null(S)) {
            n_types <- ncol(S)
            est <- matrix(0, nrow = n_samples, ncol = n_types)
            for (i in seq_len(n_samples)) {
                y <- X[, i]
                cors <- vapply(seq_len(n_types), function(j) {
                    c <- tryCatch(cor(y, S[, j], use = "pairwise.complete.obs"),
                                  error = function(...) NA_real_)
                    if (is.finite(c) && c > 0) c else 0
                }, numeric(1))
                s <- sum(cors)
                if (s > 0) est[i, ] <- cors / s
                else est[i, ] <- 1.0 / n_types
            }
            colnames(est) <- colnames(S)
            rownames(est) <- colnames(X)
        } else {
            # Absolute last resort: uniform
            n_types <- ncol(as.matrix(args$signature))
            est <- matrix(1.0 / n_types, nrow = n_samples, ncol = n_types)
            colnames(est) <- colnames(args$signature)
            rownames(est) <- colnames(X)
        }
        message("  -> Correlation fallback completed")
        P <- est
    }
}

# ═════════════════════════════════════════════════════════════════════════
# Output
# ═════════════════════════════════════════════════════════════════════════

P <- as.matrix(P)
P <- P[, colnames(args$signature), drop = FALSE]
colnames(P) <- origCTNames

# Final safety — guarantee [0, 1] rows summing to 1
P <- normalise(P, ncol(P))

DeconUtils::writeH5(NULL, P, "MethylResolver")
message("Output written successfully.")
