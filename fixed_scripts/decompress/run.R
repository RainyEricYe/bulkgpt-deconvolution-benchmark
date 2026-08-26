args <- DeconUtils::getArgs(c("bulk", "cellTypeExpr", "sigGenes", "seed"))

suppressMessages(
    suppressWarnings({
        library(DeCompress)
    })
)

bulk <- args$bulk[args$sigGenes, ]

nCellTypes <- ncol(args$cellTypeExpr)

set.seed(args$seed)

reference_props <- apply(matrix(abs(rnorm(100*nCellTypes)),ncol=nCellTypes),
                         1, function(x) x/sum(x))
reference <- args$cellTypeExpr %*% reference_props

compSpec <- findInformSet(yref = reference,
                          method = 'variance',
                          n_genes = 500,
                          n.types = nCellTypes)

csModel <- trainCS(yref = reference[args$sigGenes, ],
                   yref_need = compSpec,
                   seed = 1,
                   method = c('lasso', 'enet', 'ridge'),
                   par = T,
                   n.cores = 8,
                   lambda = .1)

dcexp <- expandTarget(bulk, csModel$compression.matrix)
dcexp <- dcexp[rowSums(dcexp < 0) == 0, ]

# Fix: bestDeconvolution has internal reference to nmf.res.
# Define it globally so linseed/celldistinguisher blocks can use it
# when TOAST is skipped (nrow < ncol triggers TOAST::findRefinx error).
if (nrow(dcexp) >= ncol(dcexp)) {
    methods <- c('TOAST', 'linseed', 'celldistinguisher')
} else {
    message("Compressed data has fewer features than samples; skipping TOAST.")
    assign("nmf.res", list(prop = NULL, sig = NULL), envir = .GlobalEnv)
    methods <- c('linseed', 'celldistinguisher')
}

csModel <- bestDeconvolution(yref = dcexp,
                             n.types = as.integer(nCellTypes),
                             methods = methods)

P <- csModel$prop
S <- csModel$sig

colnames(P) <- colnames(S) <- colnames(args$cellTypeExpr)

DeconUtils::writeH5(S, P, "DeCompress")
