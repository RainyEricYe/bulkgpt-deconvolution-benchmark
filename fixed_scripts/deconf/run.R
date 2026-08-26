args <- DeconUtils::getArgs(c("bulk", "nCellTypes", "signature"))

result <- deconf::deconfounding(args$bulk, args$nCellTypes)

S <- result$S$Matrix
P <- result$C$Matrix

rownames(S) <- rownames(args$bulk)
colnames(S) <- colnames(args$signature)
rownames(P) <- colnames(args$signature)
colnames(P) <- colnames(args$bulk)

DeconUtils::writeH5(S, t(P), "deconf")
