args <- DeconUtils::getArgs(c("bulk", "singleCellExpr", "singleCellLabels", "sigGenes"))

suppressMessages(
    suppressWarnings({
        library("deconvSeq")
    })
)

CT <- as.factor(args$singleCellLabels)
design <- model.matrix(~-1+CT)
colnames(design) <- levels(CT)
rownames(design) <- colnames(args$singleCellExpr)

# need count matrix — getdge filters lowly-expressed genes (ncpm.min=1, nsamp.min=4)
dge.celltypes <- getdge(round(args$singleCellExpr), design, ncpm.min=1, nsamp.min=4)

# FIX: filter sigGenes to only those still present in the filtered DGE object.
# getdge removes lowly-expressed genes, so some sigGenes may no longer be in
# rownames(dge.celltypes$counts), causing "subscript out of bounds" in getb0.rnaseq.
sigg_filtered <- args$sigGenes[args$sigGenes %in% rownames(dge.celltypes$counts)]

b0.singlecell <- getb0.rnaseq(dge.celltypes, design, ncpm.min=1, nsamp.min=4, sigg=sigg_filtered)

dge_tissue.sc <- getdge(round(args$bulk), NULL, ncpm.min=1, nsamp.min=4)
P <- getx1.rnaseq(NB0="top_bonferroni", b0.singlecell, dge_tissue.sc)$x1

S <- b0.singlecell$b0

DeconUtils::writeH5(S, P, "deconvSeq")
