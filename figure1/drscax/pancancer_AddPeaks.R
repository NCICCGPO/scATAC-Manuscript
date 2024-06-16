library(BiocManager)
library(BSgenome.Hsapiens.UCSC.hg38)
library(ArchR)
library(ggplot2)
library(dplyr)
library(magrittr) # needs to be run every time you start R and want to use %>%

data("geneAnnoHg38")
data("genomeAnnoHg38")
geneAnno <- geneAnnoHg38
genomeAnno <- genomeAnnoHg38
addArchRThreads(12)

#Run This to accesss hidden functions
fn <- unclass(lsf.str(envir = asNamespace("ArchR"), all = TRUE))
for (i in seq_along(fn)) {
    tryCatch({
        eval(parse(text = paste0(fn[i], "<-ArchR:::", fn[i])))
    }, error = function(x) {
    })
}

proj <- loadArchRProject("Path to ArchR Project")
proj <- addReproduciblePeakSet(ArchRProj = proj,force = TRUE, maxPeaks = 1000000,groupBy = "Sample")

saveArchRProject(proj)
