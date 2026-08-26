args <- DeconUtils::getArgs(c("bulk", "signature"))

suppressMessages(
  suppressWarnings({
      library(LinDeconSeq)
  })
)

fractions <- deconSeq(args$bulk, args$signature, verbose = F)

# Ensure fractions is a matrix (not a vector) with proper dimnames
if (is.null(dim(fractions))) {
    n_types <- length(fractions)
    fractions <- matrix(fractions, nrow = 1)
}

if (is.null(colnames(fractions)) && !is.null(colnames(args$signature))) {
    colnames(fractions) <- colnames(args$signature)
}
if (is.null(rownames(fractions)) && !is.null(colnames(args$bulk))) {
    rownames(fractions) <- colnames(args$bulk)
}

suppressMessages(suppressWarnings(library(rhdf5)))
OUTPUT_PATH <- Sys.getenv("OUTPUT_PATH", "/output/results.h5")
if (file.exists(OUTPUT_PATH)) file.remove(OUTPUT_PATH)

h5createFile(OUTPUT_PATH)
h5createGroup(OUTPUT_PATH, "P")
h5write(t(fractions), OUTPUT_PATH, "P/values")
if (!is.null(colnames(fractions))) {
    h5write(colnames(fractions), OUTPUT_PATH, "P/colnames")
}
if (!is.null(rownames(fractions))) {
    h5write(rownames(fractions), OUTPUT_PATH, "P/rownames")
}

cat("LinDeconSeq Writing results to", OUTPUT_PATH, "\n")
