"""
Test dataset for evaluating bioinformatics code models.

Provides curated test examples for:
- Sequence analysis
- Differential expression
- Visualization
- Single-cell analysis
"""

import logging
from typing import Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


# Curated test examples for bioinformatics tasks
BIOINFO_TEST_EXAMPLES = [
    # Python - Biopython sequence analysis
    {
        "instruction": "Write Python code to read a FASTA file and calculate GC content for each sequence.",
        "input": "Use Biopython to read sequences from a FASTA file.",
        "output": '''from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

def calculate_gc_content(fasta_file):
    """Calculate GC content for each sequence in a FASTA file."""
    results = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        gc = gc_fraction(record.seq) * 100
        results[record.id] = round(gc, 2)
    return results

# Example usage
gc_contents = calculate_gc_content("sequences.fasta")
for seq_id, gc in gc_contents.items():
    print(f"{seq_id}: {gc}% GC")''',
        "language": "python",
        "task": "sequence_analysis",
        "packages": ["biopython"],
    },
    # Python - Scanpy single-cell
    {
        "instruction": "Write Python code to load a 10X dataset and perform basic quality control with Scanpy.",
        "input": "The data is in 10X Genomics format.",
        "output": '''import scanpy as sc

def qc_10x_data(data_path, min_genes=200, min_cells=3, max_mito=20):
    """Load 10X data and perform quality control."""
    # Load data
    adata = sc.read_10x_mtx(data_path)
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(
        adata, 
        qc_vars=['mt'], 
        percent_top=None, 
        inplace=True
    )
    
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = adata[adata.obs.pct_counts_mt < max_mito, :]
    
    print(f"After QC: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata

# Example usage
adata = qc_10x_data("filtered_feature_bc_matrix/")''',
        "language": "python",
        "task": "single_cell",
        "packages": ["scanpy"],
    },
    # Python - Pandas for genomics
    {
        "instruction": "Write Python code to parse a VCF file and filter variants by quality score.",
        "input": "Filter variants with QUAL > 30 and read depth > 10.",
        "output": '''import pandas as pd

def parse_vcf(vcf_file):
    """Parse VCF file into a DataFrame."""
    # Read VCF, skip header lines
    with open(vcf_file) as f:
        lines = [l for l in f if not l.startswith('##')]
    
    df = pd.read_csv(
        pd.io.common.StringIO(''.join(lines)),
        sep='\\t',
        dtype={'#CHROM': str, 'POS': int, 'QUAL': float}
    )
    df.columns = df.columns.str.replace('#', '')
    return df

def filter_variants(vcf_file, min_qual=30, min_depth=10):
    """Filter variants by quality and depth."""
    df = parse_vcf(vcf_file)
    
    # Filter by quality
    df = df[df['QUAL'] > min_qual]
    
    # Parse INFO field for depth
    df['DP'] = df['INFO'].str.extract(r'DP=(\\d+)').astype(float)
    df = df[df['DP'] > min_depth]
    
    return df[['CHROM', 'POS', 'REF', 'ALT', 'QUAL', 'DP']]

# Example usage
filtered = filter_variants("variants.vcf", min_qual=30, min_depth=10)
print(f"Filtered variants: {len(filtered)}")''',
        "language": "python",
        "task": "variant_analysis",
        "packages": ["pandas"],
    },
    # R - DESeq2 differential expression
    {
        "instruction": "Write R code to perform differential expression analysis using DESeq2.",
        "input": "Compare treatment vs control groups with batch effect correction.",
        "output": '''library(DESeq2)

#' Perform differential expression analysis with DESeq2
#' @param counts Count matrix with genes as rows
#' @param coldata Sample information with condition and batch
#' @return DESeqResults object
run_deseq2 <- function(counts, coldata) {
    # Create DESeq2 dataset
    dds <- DESeqDataSetFromMatrix(
        countData = counts,
        colData = coldata,
        design = ~ batch + condition
    )
    
    # Filter low counts
    keep <- rowSums(counts(dds) >= 10) >= 3
    dds <- dds[keep, ]
    
    # Run DESeq2
    dds <- DESeq(dds)
    
    # Get results
    res <- results(dds, contrast = c("condition", "treatment", "control"))
    res <- res[order(res$padj), ]
    
    return(res)
}

# Example usage
# counts <- read.csv("counts.csv", row.names = 1)
# coldata <- read.csv("samples.csv", row.names = 1)
# results <- run_deseq2(counts, coldata)
# sig_genes <- subset(results, padj < 0.05 & abs(log2FoldChange) > 1)''',
        "language": "r",
        "task": "differential_expression",
        "packages": ["DESeq2"],
    },
    # R - ComplexHeatmap visualization
    {
        "instruction": "Write R code to create an annotated heatmap of gene expression data using ComplexHeatmap.",
        "input": "Include row and column annotations with clustering.",
        "output": '''library(ComplexHeatmap)
library(circlize)

#' Create annotated expression heatmap
#' @param expr_matrix Expression matrix (genes x samples)
#' @param sample_info Sample annotation data frame
#' @param gene_info Gene annotation data frame
create_expression_heatmap <- function(expr_matrix, sample_info, gene_info) {
    # Scale data
    scaled_mat <- t(scale(t(expr_matrix)))
    
    # Color function
    col_fun <- colorRamp2(c(-2, 0, 2), c("blue", "white", "red"))
    
    # Sample annotation
    ha_col <- HeatmapAnnotation(
        Condition = sample_info$condition,
        Batch = sample_info$batch,
        col = list(
            Condition = c("control" = "#1f77b4", "treatment" = "#ff7f0e"),
            Batch = c("A" = "#2ca02c", "B" = "#d62728", "C" = "#9467bd")
        )
    )
    
    # Gene annotation
    ha_row <- rowAnnotation(
        Pathway = gene_info$pathway,
        LogFC = anno_barplot(gene_info$logFC)
    )
    
    # Create heatmap
    ht <- Heatmap(
        scaled_mat,
        name = "Z-score",
        col = col_fun,
        top_annotation = ha_col,
        left_annotation = ha_row,
        row_names_gp = gpar(fontsize = 8),
        column_names_gp = gpar(fontsize = 10),
        clustering_distance_rows = "euclidean",
        clustering_method_rows = "ward.D2"
    )
    
    return(ht)
}

# Example usage
# pdf("expression_heatmap.pdf", width = 10, height = 12)
# draw(create_expression_heatmap(expr_matrix, sample_info, gene_info))
# dev.off()''',
        "language": "r",
        "task": "visualization",
        "packages": ["ComplexHeatmap"],
    },
    # R - Seurat single-cell
    {
        "instruction": "Write R code to perform clustering and UMAP visualization of single-cell data using Seurat.",
        "input": "Include normalization, variable gene selection, and clustering.",
        "output": '''library(Seurat)

#' Process single-cell data with Seurat
#' @param counts_matrix Raw count matrix
#' @param project_name Name for the Seurat object
#' @return Processed Seurat object
process_scrnaseq <- function(counts_matrix, project_name = "scRNA") {
    # Create Seurat object
    seurat_obj <- CreateSeuratObject(
        counts = counts_matrix,
        project = project_name,
        min.cells = 3,
        min.features = 200
    )
    
    # QC metrics
    seurat_obj[["percent.mt"]] <- PercentageFeatureSet(
        seurat_obj, 
        pattern = "^MT-"
    )
    
    # Filter cells
    seurat_obj <- subset(
        seurat_obj,
        subset = nFeature_RNA > 200 & 
                 nFeature_RNA < 5000 & 
                 percent.mt < 20
    )
    
    # Normalize
    seurat_obj <- NormalizeData(seurat_obj)
    
    # Find variable features
    seurat_obj <- FindVariableFeatures(
        seurat_obj, 
        selection.method = "vst",
        nfeatures = 2000
    )
    
    # Scale and PCA
    seurat_obj <- ScaleData(seurat_obj)
    seurat_obj <- RunPCA(seurat_obj, npcs = 50)
    
    # Clustering
    seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30)
    seurat_obj <- FindClusters(seurat_obj, resolution = 0.5)
    
    # UMAP
    seurat_obj <- RunUMAP(seurat_obj, dims = 1:30)
    
    return(seurat_obj)
}

# Example usage
# seurat <- process_scrnaseq(counts_matrix)
# DimPlot(seurat, reduction = "umap", label = TRUE)''',
        "language": "r",
        "task": "single_cell",
        "packages": ["Seurat"],
    },
    # Python - matplotlib bioinformatics plot
    {
        "instruction": "Write Python code to create a volcano plot for differential expression results.",
        "input": "Highlight significant genes with adjusted p-value < 0.05 and |log2FC| > 1.",
        "output": '''import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def volcano_plot(results_df, pval_col='padj', fc_col='log2FoldChange',
                 pval_threshold=0.05, fc_threshold=1.0, save_path=None):
    """
    Create a volcano plot for differential expression results.
    
    Parameters:
    -----------
    results_df : DataFrame with differential expression results
    pval_col : Column name for adjusted p-values
    fc_col : Column name for log2 fold changes
    pval_threshold : Significance threshold for p-value
    fc_threshold : Threshold for fold change
    save_path : Path to save the figure
    """
    df = results_df.copy()
    df['-log10(padj)'] = -np.log10(df[pval_col])
    
    # Classify genes
    conditions = [
        (df[pval_col] < pval_threshold) & (df[fc_col] > fc_threshold),
        (df[pval_col] < pval_threshold) & (df[fc_col] < -fc_threshold),
    ]
    choices = ['Up', 'Down']
    df['Status'] = np.select(conditions, choices, default='NS')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'Up': '#e74c3c', 'Down': '#3498db', 'NS': '#95a5a6'}
    
    for status, color in colors.items():
        mask = df['Status'] == status
        ax.scatter(
            df.loc[mask, fc_col],
            df.loc[mask, '-log10(padj)'],
            c=color, label=status, alpha=0.6, s=20
        )
    
    # Add threshold lines
    ax.axhline(-np.log10(pval_threshold), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(fc_threshold, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-fc_threshold, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('log2 Fold Change', fontsize=12)
    ax.set_ylabel('-log10(adjusted p-value)', fontsize=12)
    ax.set_title('Volcano Plot', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig, ax

# Example usage
# volcano_plot(de_results, save_path='volcano.png')''',
        "language": "python",
        "task": "visualization",
        "packages": ["matplotlib", "pandas", "numpy"],
    },
    # Python - pysam BAM processing
    {
        "instruction": "Write Python code to calculate read depth at each position from a BAM file.",
        "input": "Use pysam to process the aligned reads.",
        "output": '''import pysam
from collections import defaultdict

def calculate_depth(bam_file, region=None, min_mapq=20):
    """
    Calculate read depth at each position from a BAM file.
    
    Parameters:
    -----------
    bam_file : Path to BAM file (must be indexed)
    region : Optional region string (e.g., 'chr1:1000-2000')
    min_mapq : Minimum mapping quality
    
    Returns:
    --------
    Dictionary with position -> depth mapping
    """
    depth = defaultdict(int)
    
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for pileupcolumn in bam.pileup(region=region, min_mapping_quality=min_mapq):
            depth[pileupcolumn.pos] = pileupcolumn.n
    
    return dict(depth)

def get_coverage_stats(bam_file, region=None, min_mapq=20):
    """Calculate coverage statistics for a BAM file."""
    depth = calculate_depth(bam_file, region, min_mapq)
    
    if not depth:
        return {"mean": 0, "median": 0, "max": 0, "min": 0}
    
    depths = list(depth.values())
    
    return {
        "mean": sum(depths) / len(depths),
        "median": sorted(depths)[len(depths) // 2],
        "max": max(depths),
        "min": min(depths),
        "positions": len(depths)
    }

# Example usage
# stats = get_coverage_stats("aligned.bam", region="chr1:1-1000000")
# print(f"Mean depth: {stats['mean']:.2f}x")''',
        "language": "python",
        "task": "alignment",
        "packages": ["pysam"],
    },
]


class BioinfoTestDataset:
    """
    Curated test dataset for bioinformatics code evaluation.
    """

    def __init__(self, examples: Optional[list[dict]] = None):
        self.examples = examples or BIOINFO_TEST_EXAMPLES

    def get_dataset(self) -> Dataset:
        """Get the test dataset as a Hugging Face Dataset."""
        return Dataset.from_list(self.examples)

    def get_by_task(self, task: str) -> Dataset:
        """Get examples for a specific task."""
        filtered = [ex for ex in self.examples if ex.get("task") == task]
        return Dataset.from_list(filtered)

    def get_by_language(self, language: str) -> Dataset:
        """Get examples for a specific language."""
        filtered = [ex for ex in self.examples if ex.get("language") == language]
        return Dataset.from_list(filtered)

    def get_tasks(self) -> list[str]:
        """Get list of available tasks."""
        return list(set(ex.get("task", "unknown") for ex in self.examples))

    def add_example(
        self,
        instruction: str,
        output: str,
        language: str,
        task: str,
        input_context: str = "",
        packages: Optional[list[str]] = None,
    ):
        """Add a new test example."""
        self.examples.append({
            "instruction": instruction,
            "input": input_context,
            "output": output,
            "language": language,
            "task": task,
            "packages": packages or [],
        })


def create_test_dataset(
    num_examples: Optional[int] = None,
    tasks: Optional[list[str]] = None,
    languages: Optional[list[str]] = None,
) -> Dataset:
    """
    Create a test dataset for evaluation.

    Args:
        num_examples: Maximum number of examples
        tasks: Filter to specific tasks
        languages: Filter to specific languages

    Returns:
        Hugging Face Dataset
    """
    examples = BIOINFO_TEST_EXAMPLES.copy()

    # Filter by task
    if tasks:
        examples = [ex for ex in examples if ex.get("task") in tasks]

    # Filter by language
    if languages:
        examples = [ex for ex in examples if ex.get("language") in languages]

    # Limit examples
    if num_examples and num_examples < len(examples):
        examples = examples[:num_examples]

    logger.info(f"Created test dataset with {len(examples)} examples")
    return Dataset.from_list(examples)

