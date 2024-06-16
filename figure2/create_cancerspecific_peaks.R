library(BiocManager)
library(BSgenome.Hsapiens.UCSC.hg38)
library(ArchR)
library(ggplot2)
library(TFBSTools)
library(dplyr)
library(rhdf5)
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


Arrowpath <- "Link to cancer specific arrow files"
ArrowFiles <- list.files(Arrowpath, pattern = ".arrow", full.names = TRUE)
ArrowFiles


#make a new archr project
proj_1<- ArchRProject(
  ArrowFiles = ArrowFiles, 
  geneAnnotation = geneAnno,
  genomeAnnotation = genomeAnno,
  outputDirectory = "Cancer specific archr project"
)

proj_1

#read in the filtered cells for analysis
req_Cells<-read.csv('TableS1.csv')

#merge with cells from cancer and subset only cancer cells
coldata<-getCellColData(proj_1)
nonimmunecells<-rownames(coldata)[rownames(coldata) %in% req_Cells$X]
coldata<-coldata[rownames(coldata) %in% nonimmunecells,]

#subset ArchR Project to only have cancer cells
proj<-subsetCells(proj_1,cellNames=rownames(coldata))
proj

#saving archr Project
saveArchRProject(proj)


#getting reproducible peak 
proj <- addGroupCoverages(ArchRProj = proj, force = TRUE,groupBy = "Sample")
#Call Reproducible Peaks w/ Macs2 
proj <- addReproduciblePeakSet(ArchRProj = proj,force = TRUE, maxPeaks = 1000000,groupBy = "Sample")

peakset<-getPeakSet(proj)

write.csv(data.frame(peakset),'Cancer specific peakset.csv')

#saving the object again
saveArchRProject(proj)




