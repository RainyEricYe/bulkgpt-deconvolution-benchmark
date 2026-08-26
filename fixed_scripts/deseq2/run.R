args <- DeconUtils::getArgs(c("bulk", "cellTypeExpr"))

suppressMessages(
  suppressWarnings({
      library(DESeq2)
    })
)

# Fix: use shift=1 instead of alpha=0.01 to avoid sqrt(negative) in VST
# which causes L-BFGS-B to fail with "needs finite values of 'fn'"
P <- unmix(args$bulk, args$cellTypeExpr, shift = 1, quiet = T)

DeconUtils::writeH5(NULL, P, "DESeq2")
