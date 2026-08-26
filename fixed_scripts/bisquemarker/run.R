# BisqueMarker deconvolution method
# Fixed: split comma-separated marker strings into individual genes,
#        and filter to only genes present in bulk data
args <- DeconUtils::getArgs(c("bulk", "markers"))

suppressMessages(
    suppressWarnings({
        library(dplyr)
        library(BisqueRNA)
    })
)

bulk.eset <- Biobase::ExpressionSet(assayData = args$bulk)
bulk_genes <- rownames(args$bulk)

if (is.null(names(args$markers))){
    names(args$markers) <- paste0("CellType", 1:length(args$markers))
}

# Split comma-separated marker strings into individual genes and filter
markers <- lapply(names(args$markers), function(ct) {
    # Split the comma-separated string into individual gene names
    genes <- trimws(unlist(strsplit(args$markers[[ct]], ",")))
    # Keep only genes present in the bulk expression matrix
    genes <- genes[genes %in% bulk_genes]
    if (length(genes) == 0) {
        warning(paste0("No markers for cell type ", ct, " overlap with bulk genes"))
        return(NULL)
    }
    data.frame(gene = genes, cluster = ct, stringsAsFactors = FALSE)
})
markers <- do.call(rbind, markers)

if (is.null(markers) || nrow(markers) == 0) {
    stop("No overlapping genes between any markers and bulk.eset")
}

P <- MarkerBasedDecomposition(bulk.eset, markers=markers, weighted=F)$bulk.props

DeconUtils::writeH5(NULL, t(P), "BisqueMarker")
