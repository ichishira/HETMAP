# HEAMTP : Quantitative Assessment of Histological Intratumoral Heterogeneity Using Spatial Feature Clustering with Multiple Foundation Models

## Abstract
Intratumoral heterogeneity (ITH) plays a central role in cancer evolution, treatment resistance, and patient outcomes. Current methods for ITH assessment rely on costly genomic or transcriptomic profiling or on subjective morphological evaluation, limiting their clinical applicability. Here we present HETMAP (HETerogeneity Mapping through Multiple AI Pathology models), a fully automated computational framework for objective quantification of morphological heterogeneity from standard H&E slides. HETMAP integrates three complementary pathology foundation models for epithelial segmentation, morphological feature embedding, and tumor region identification, combined with spatially constrained clustering to delineate tumor subclones.

Validation against single-cell spatial transcriptomics in gastric cancer demonstrated high concordance between HETMAP clusters and gene expression–defined subpopulations, confirming biological relevance. In The Cancer Genome Atlas and independent validation cohorts, higher HETMAP-derived heterogeneity was consistently associated with poor survival outcomes (e.g., gastric adenocarcinoma, HR=3.02, p=0.009; breast cancer, HR=2.14, p=0.03). Comprehensive ablation studies further established the necessity of epithelial masking, spatial smoothing, and tumor-specific filtering for prognostic performance. Importantly, HETMAP features enhanced prognostic prediction when integrated with embeddings from state-of-the-art foundation models.

By enabling cost-effective and scalable quantification of ITH from routinely available slides, HETMAP bridges conventional pathology with computational oncology. This framework provides robust biomarkers of tumor heterogeneity that may facilitate large-scale retrospective studies and support precision oncology by improving patient stratification for therapy. 

## Graphical Abstract

![Graphical Abstract](graphical_abstract.png)


## Workflow

The workflow consists of the following three steps, each with its expected directory structure for inputs and outputs.

### 1. Patch Extraction from WSIs

This step extracts smaller image patches from the large WSI files at a consistent magnification.

- **Directory:** `TCGA_wsi_patch/`
- **Process:**
    1.  The `make_files_for_patch.ipynb` notebook reads slide information to generate lists of samples to process.
    2.  The `create_patches_fp.py` script from the `CLAM` repository is used for patch extraction.

- **Example Directory Structure & Command:**

  - **Input:** A directory containing your WSI files.
    ```
    /path/to/wsi_slides/
    ├── sample_A.svs
    └── sample_B.svs
    ```

  - **Command:** The `--patch_size` should be adjusted based on the slide's magnification (e.g., 224 for 20x, 448 for 40x).
    ```bash
    python ../CLAM/create_patches_fp.py \
        --source /path/to/wsi_slides/ \
        --save_dir /path/to/output_patches/ \
        --patch_size 224 \
        --seg --patch --stitch
    ```

  - **Output:** The script will generate a directory containing `.h5` files with patch coordinates.
    ```
    /path/to/output_patches/
    └── patches/
        ├── sample_A.h5
        └── sample_B.h5
    ```

### 2. Embedding with UNI

This step generates a feature vector (embedding) for each patch using the UNI pathology foundation model.

- **Directory:** `UNI_encode_with_mask/` (or `UNI_encode_wo_mask/`)
- **Process:**
    1.  Create a file named `samples_to_encode.txt` listing the paths to the patch `.h5` files you want to process.
    2.  Run the `encode_all.sh` script to generate embeddings.

- **Example Directory Structure & Command:**

  - **Input:** The directory containing the `encode_all.sh` script and a text file listing samples.
    ```
    UNI_encode_with_mask/
    ├── samples_to_encode.txt
    ├── encode_all.sh
    └── UNI_embedding.py
    ```

  - **Command:**
    ```bash
    cd UNI_encode_with_mask/
    ./encode_all.sh
    ```

  - **Output:** The script creates a `features` directory containing the embeddings for each sample.
    ```
    UNI_encode_with_mask/
    └── features/
        ├── sample_A/
        │   ├── features.npy
        │   └── coords.npy
        └── sample_B/
            ...
    ```

### 3. Patch Clustering with HETMAP

The final step clusters the patch embeddings to identify and visualize spatially distinct, heterogeneous regions.

- **Directory:** `hetero_cluster/`
- **Process:**
    1.  The main clustering logic is in `hetero_cluster.py`.
    2.  To run on multiple samples, adapt the `cluster_all.sh` script.

- **Example Directory Structure & Command:**

  - **Input:** Paths to directories containing slides, patches, and features are provided as arguments.
    - `-slidedir /path/to/your/slides/`
    - `-h5dir /path/to/h5/patches/`
    - `-featuredir /path/to/encoded/features/`

  - **Command (for a single sample):**
    ```bash
    python ./hetero_cluster/hetero_cluster.py \
        -slidedir /path/to/your/slides/ \
        -h5dir /path/to/h5/patches/ \
        -featuredir /path/to/encoded/features/ \
        -sample YOUR_SAMPLE_ID \
        -maxcluster 15 \
        -savedir output_clustering_results \
        -seed 314
    ```
  - **Output:** A directory containing clustering results and visualizations for each sample.
    ```
    output_clustering_results/
    └── YOUR_SAMPLE_ID/
        ├── optimized_state.npy
        ├── clustering_optimized.png
        └── he.png
    ```

- **Downstream Analysis:**

  - **Correlation with Prognosis:** The `prog_est_clonenum_corr_chief_all_STAD_tot.ipynb` notebook provides an example of how to correlate the HETMAP heterogeneity metric with clinical outcomes.
    - **Process:**
      1.  **Loads Data:** The notebook loads the clustering results and clinical/prognostic data for the cohort (e.g., from `prog_data/stad_tcga_pan_can_atlas_2018_clinical_data.tsv`).
      2.  **Calculates Heterogeneity Score:** It calculates a heterogeneity score for each tumor based on the number of distinct clusters.
      3.  **Performs Survival Analysis:** Using the `lifelines` library, it divides patients into "High" and "Low" heterogeneity groups to generate Kaplan-Meier survival curves and assess the prognostic impact.

  - **Correlation with Pathologist Assessment:** The `histrogy_analysis_ICGC.ipynb` notebook demonstrates that the subclone counts derived from HETMAP show a significant correlation with those evaluated by expert pathologists, confirming the clinical relevance of the automated assessment.

  - **Integration with Other Models:** The framework's prognostic prediction capabilities can be further enhanced by integrating HETMAP features with other models, such as TITAN.
