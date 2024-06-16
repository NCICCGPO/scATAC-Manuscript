library(BiocManager)
library(ArchR)
library(ggplot2)
library(ggplot2)
library(BSgenome.Hsapiens.UCSC.hg38)

data("geneAnnoHg38")
data("genomeAnnoHg38")
geneAnno <- geneAnnoHg38
genomeAnno <- genomeAnnoHg38
addArchRThreads(24)


fn <- unclass(lsf.str(envir = asNamespace("ArchR"), all = TRUE))
  for(i in seq_along(fn)){
    tryCatch({
      eval(parse(text=paste0(fn[i], '<-ArchR:::', fn[i])))
    }, error = function(x){
    })
  }


proj<-loadArchRProject("Path to archr project with cancer cells") 

library(stringr)
for(i in getArrowFiles(proj)){
    #print(i)
    a=strsplit(i, '.', fixed=T)[[1]][1]
    #print(a)
    usename=strsplit(a,'/',fixed=T)[[1]][10]
    print(usename)
    temp<-getMatrixFromArrow(i,useMatrix='PeakMatrix')
    saveRDS(temp,paste('/TCGA/peakrds_sample/',usename,'.rds',sep=''))
}


peakSet<-getPeakSet(proj)
write.csv(data.frame(peakSet),'peakmetadata.csv')

