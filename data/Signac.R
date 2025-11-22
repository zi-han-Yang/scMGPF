
library(Signac)
library(sp)
library(SeuratObject)
library(Seurat)
library(reticulate)
library(sceasy)
library(EnsDb.Mmusculus.v79)  # Annotating the mouse genome (assuming version v79, adjust as needed)

library(zellkonverter)
library(rhdf5)
library(anndata)
# Read ATAC data

sce <- readH5AD("E:/data/raw_data/GSE126074_AdBrainCortex/ATAC/ATAC.h5ad")
data.atac <- sce@assays@data@listData[["X"]]

# Filter out the ATAC peaks and retain only the standard chromosomes
grange.counts <- StringToGRanges(rownames(data.atac), sep = c(":", "-"))
grange.use <- seqnames(grange.counts) %in% standardChromosomes(grange.counts)
data.atac <- data.atac[as.vector(grange.use), ]

# Get mouse genome annotations
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Mmusculus.v79)
seqlevelsStyle(annotations) <- 'UCSC'
genome(annotations) <- "mm10"  #  Mouse genome version mm10
# Mouse genome version mm10

# designated fragments could file path (replaced with the actual path; GSE126074 data set may have corresponding fragments could file, such as GSE126074_AdBrainCortex_SNAREseq_chromatin. Fragments could. The TSV. Gz)
frag.file <- "E:/data/raw_data/GSE126074_AdBrainCortex/ATAC/fragments.sort.bed.gz"  

# Do not use fragments files
#frag.file <- NULL
#cat(" The fragments file was not used, and the gene activity matrix was generated relying on the peak count matrix. \n")

# Create a ChromatinAssay object
chrom_assay <- CreateChromatinAssay(
  counts = data.atac,
  sep = c(":", "-"),
  genome = 'mm10',
  fragments = frag.file,
  min.cells = 10,
  annotation = annotations
)

# Create a Seurat object
AdBrainCortex <- CreateSeuratObject(
  counts = chrom_assay,
  assay = "ATAC"
)

# Set the default assay and calculate gene activity
DefaultAssay(AdBrainCortex) <- "ATAC"
gene.activities <- GeneActivity(object = AdBrainCortex)

# Create a gene activity Seurat object and convert it to AnnData format
scgeneactivity <- CreateSeuratObject(counts = gene.activities, assay = "RNA")
sceasy::convertFormat(scgeneactivity, from = "seurat", to = "anndata", outFile = 'E:/data/raw_data/GSE126074_AdBrainCortex/ATAC/ATAC_Signac.h5ad')


