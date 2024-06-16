# preliminaries
library("Matrix")

dfp <- '/sparsemtx'
mdfp <- '/barcodes'
rds_fp <- 'TCGA/peakrds_specificCancers'

# for loop or functionalize this:
counter <- 0
for (file in Sys.glob(paste(rds_fp, '*.rds', sep=''))){
    counter <- counter + 1
    
    file_basename <- basename(tools::file_path_sans_ext(file))
    
    data <- readRDS(file)
    data # init the obj
    sparse_file <- file.path(dfp, paste(file_basename, 'mtx', sep="."))
    barcodes <- file.path(mdfp, paste(file_basename, 'csv', sep="."))
    write.csv(colData(data), file=barcodes)
    writeMM(assays(data)$PeakMatrix, sparse_file)
    print(paste(counter, file_basename, sep=" : "))
}
