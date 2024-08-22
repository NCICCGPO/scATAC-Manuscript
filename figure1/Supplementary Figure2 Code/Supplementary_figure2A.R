library(SCISSOR)

load("pileup_data/CDKN2A_PANCAN_pileup_part_intron.RData")

BLCA.manifest <- read.table("pileup_data/BLCA_manifest.txt",sep="\t",header=T,stringsAsFactors=F)
COAD.manifest <- read.table("pileup_data/COAD_manifest.txt",sep="\t",header=T,stringsAsFactors=F)

pileup.BLCA <- pileupList$BLCA
colnames(pileup.BLCA) <- BLCA.manifest$barcode
pileup.COAD <- pileupList$COAD
colnames(pileup.COAD) <- COAD.manifest$barcode

pdf(6,4,file="output/CDKN2A_BLCA_5.pdf")
plot_pileup(Pileup = pileup.BLCA, cases = "TCGA-DK-A1AD-01A", Ranges = geneRanges)
dev.off()

pdf(6,4,file="output/CDKN2A_COAD_30.pdf")
plot_pileup(Pileup = pileup.COAD, cases = "TCGA-NH-A6GA-01A", Ranges = geneRanges)
dev.off()

