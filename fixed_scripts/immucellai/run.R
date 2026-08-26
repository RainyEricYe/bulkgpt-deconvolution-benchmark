args <- DeconUtils::getArgs(c("bulk", "signature", "markers"))

suppressMessages(
  suppressWarnings({
          library(e1071)
          library(GSVA)
          library(pracma)
          library(quadprog)
          source("/code/ImmuCellAI.R")
  })
)

# Fix: DeconBenchmark stores markers as comma-separated strings in a single
# array, but ImmuCellAI expects a named list of gene vectors.
# Parse the marker strings into the expected list format.
if (is.array(args$markers) && length(dim(args$markers)) == 1) {
  marker_names <- tryCatch(
    as.character(rhdf5::h5read(Sys.getenv("INPUT_PATH"), "markers/names")),
    error = function(e) paste0("CT", seq_along(args$markers))
  )
  marker_list <- list()
  for (i in seq_along(args$markers)) {
    marker_list[[marker_names[i]]] <- trimws(strsplit(args$markers[i], ",")[[1]])
  }
  args$markers <- marker_list
}

P <- Sample_abundance_calculation(args$bulk, args$markers, args$signature, "rnaSeq", 1)

DeconUtils::writeH5(NULL, P, "ImmuCellAI")
