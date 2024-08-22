library(Seurat)
library(ArchR)
library(Cairo)
addArchRGenome("hg38")

# Loading arrowfiles
ArrowFiles <- list.files(path = "data/", full.names = TRUE)

projSKCM <- ArchRProject(
  ArrowFiles = ArrowFiles, 
  outputDirectory = "output/",
  copyArrows = FALSE #This is recommended so that if you modify the Arrow files you have an original copy for later usage.
)

# Annotatin sampleID
bioNames <- gsub("scATAC_SKCM_0852FA43_7577_4456_8033_9A9156A7B258_X005_S07_B1_T1","TCGA-DA-A95Y",projSKCM$Sample)
bioNames <- gsub("scATAC_SKCM_211D9CF4_3348_4DCD_8A01_6827435DDB3D_X004_S07_B1_T1","TCGA-EB-A5KH",bioNames)
bioNames <- gsub("scATAC_SKCM_398F831B_A6C7_40D9_9EC4_16CECA35AEA2_X014_S03_B1_T1","TCGA-D9-A6EA",bioNames)
bioNames <- gsub("scATAC_SKCM_8F708E04_2936_4E85_85C2_D1431003898B_X011_S07_B1_T1","TCGA-EB-A5SH",bioNames)
bioNames <- gsub("scATAC_SKCM_F05B8E69_5AD9_4FCF_8980_1307F35BD173_X007_S07_B1_T1","TCGA-DA-A95W",bioNames)
bioNames <- gsub("scATAC_SKCM_F15664E6_AE19_4B59_971A_8FC8E05CF921_X012_S04_B1_T1","TCGA-D9-A6EC",bioNames)
bioNames <- gsub("scATAC_SKCM_F318F3E5_E6BE_4CBB_977A_ABE202DCE9EA_X009_S07_B1_T1","TCGA-D9-A1JW",bioNames)
bioNames <- gsub("scATAC_SKCM_FDA487D2_5293_4315_9212_3836856CCFFB_X008_S06_B1_T1","TCGA-D3-A8GP",bioNames)
bioNames <- gsub("scATAC_SKCM_FE986D7E_FB8B_4B58_A50C_CAED05FFCAA5_X006_S07_B1_T1","TCGA-D3-A8GM",bioNames)


projSKCM$TCGA_id <- bioNames

# Processing arrowfiles
set.seed(0)
projSKCM <- addIterativeLSI(
  # first round iterative LSI based on tilematrix, with default para, it will carry out estimated LSI
  ArchRProj = projSKCM,
  useMatrix = "TileMatrix",
  name = "IterativeLSI",
  dimsToUse = 1:30,
  varFeatures = 25000,
  force = TRUE
)
projSKCM <- addIterativeLSI(
  ArchRProj = projSKCM,
  useMatrix = "TileMatrix", 
  name = "IterativeLSI", 
  iterations = 2, 
  clusterParams = list( #See Seurat::FindClusters
    resolution = c(0.2), 
    sampleCells = 10000, 
    n.start = 10
  ), 
  varFeatures = 25000, 
  dimsToUse = 1:30,
  force=TRUE
)

# More batch correction (if needed). This process creates a new reducedDims object called “Harmony” in proj object.
projSKCM <- addHarmony(
  ArchRProj = projSKCM,
  reducedDims = "IterativeLSI",
  name = "Harmony",
  groupBy = "TCGA_id",
  force=TRUE
)

projSKCM <- addClusters(
  # add cluster based on LSI using Seurat
  input = projSKCM,
  reducedDims = "Harmony",
  method = "Seurat",
  resolution = 0.1,
  force = TRUE
)

projSKCM <- addUMAP(
  # add embedding
  ArchRProj = projSKCM,
  reducedDims = "Harmony",
  name = "UMAP",
  nNeighbors = 30,
  minDist = 0.5,
  metric = "cosine",
  force = TRUE
)

p1 <- plotEmbedding(ArchRProj = projSKCM, colorBy = "cellColData", name = "TCGA_id", embedding = "UMAP", rastr = TRUE)
p2 <- plotEmbedding(ArchRProj = projSKCM, colorBy = "cellColData", name = "Clusters", embedding = "UMAP", rastr = TRUE)

# Supplementary Figur 2D
pdf(12,10, file="output/UMAP_clusters_SampleID.pdf")
p1
dev.off()

pdf(12,10, file="output/UMAP_clusters_ClusterID.pdf")
p2
dev.off()

# Pseudo bulk ATAC-seq peak for CDKN2A regions according to Clusters (Supplementary Figur 2E)

gr=GRanges(seqnames=c("chr9"),
           ranges=IRanges(start=c(21962752),end=c(22000324))
)

p <- plotBrowserTrack(ArchRProj = projSKCM, region = gr, groupBy = "Clusters", tileSize = 150) # Restricted regions
p <- plotBrowserTrack(ArchRProj = projSKCM, geneSymbol = "CDKN2A", groupBy = "Clusters", tileSize = 250) # CDKN2A genic regions

grid::grid.draw(p)

pdf(4,6, file="output/clusters_genome_browser.pdf")
grid::grid.draw(p)
dev.off()
