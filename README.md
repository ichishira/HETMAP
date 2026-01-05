# HEAMTP : Quantitative Assessment of Histological Intratumoral Heterogeneity Using Spatial Feature Clustering with Multiple Foundation Models

## Abstract
Intratumoral heterogeneity (ITH) plays a central role in cancer evolution, treatment resistance, and patient outcomes. Current methods for ITH assessment rely on costly genomic or transcriptomic profiling or on subjective morphological evaluation, limiting their clinical applicability. Here we present HETMAP (HETerogeneity Mapping through Multiple AI Pathology models), a fully automated computational framework for objective quantification of morphological heterogeneity from standard H&E slides. HETMAP integrates three complementary pathology foundation models for epithelial segmentation, morphological feature embedding, and tumor region identification, combined with spatially constrained clustering to delineate tumor subclones.

Validation against single-cell spatial transcriptomics in gastric cancer demonstrated high concordance between HETMAP clusters and gene expression–defined subpopulations, confirming biological relevance. In The Cancer Genome Atlas and independent validation cohorts, higher HETMAP-derived heterogeneity was consistently associated with poor survival outcomes (e.g., gastric adenocarcinoma, HR=3.02, p=0.009; breast cancer, HR=2.14, p=0.03). Comprehensive ablation studies further established the necessity of epithelial masking, spatial smoothing, and tumor-specific filtering for prognostic performance. Importantly, HETMAP features enhanced prognostic prediction when integrated with embeddings from state-of-the-art foundation models.

By enabling cost-effective and scalable quantification of ITH from routinely available slides, HETMAP bridges conventional pathology with computational oncology. This framework provides robust biomarkers of tumor heterogeneity that may facilitate large-scale retrospective studies and support precision oncology by improving patient stratification for therapy. 

## Graphical Abstract

![Graphical Abstract](graphical_abstract.png)


## Workflow

The workflow consists of the following three steps.

### 1. Patch Extraction from WSIs

This step extracts patches from WSIs while maintaining a consistent magnification.

- **Directory:** `TCGA_wsi_patch/`
- **Process:**
    1.  The `make_files_for_patch.ipynb` notebook reads the magnification level from the slides and creates text files listing the samples to be processed. Patches are extracted at a size of 224x224 pixels for 20x samples and 448x448 pixels for 40x samples.
    2.  Execute the patching script from the `CLAM` repository. The `--patch_size` should be adjusted based on the slide's magnification.

- **Example Command:**
    ```bash
    python ../CLAM/create_patches_fp.py \
        --source /path/to/wsi_slides/ \
        --save_dir /path/to/output_patches/ \
        --patch_size 224 \
        --seg --patch --stitch
    ```

### 2. Embedding with UNI

This step generates feature embeddings for the extracted patches using the UNI model.

- **Directory:** `UNI_encode_with_mask/` (or `UNI_encode_wo_mask/` for ablation studies)
- **Process:**
    1.  Only patches where at least 1/4 of the area consists of epithelial components are encoded.
    2.  List the paths to the samples you wish to encode in a file named `samples_to_encode.txt`.
    3.  Run the `encode_all.sh` script to generate embeddings for all specified samples.

- **Example Command:**
    ```bash
    cd UNI_encode_with_mask/
    ./encode_all.sh
    ```

### 3. Patch Clustering with HETMAP

This step clusters the patch embeddings to identify and analyze tumor heterogeneity.

- **Directory:** `hetero_cluster/`
- **Process:**
    1.  The main clustering logic is handled by `hetero_cluster.py`.
    2.  To run clustering for multiple samples in parallel, you can adapt the `cluster_all.sh` script.

- **Example Command (for a single sample):**
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
- **Downstream Analysis:**

  - **Correlation with Prognosis:** The `prog_est_clonenum_corr_chief_all_STAD_tot.ipynb` notebook provides an example of how to correlate the HETMAP heterogeneity metric with clinical outcomes.
    - **Process:**
      1.  **Loads Data:** The notebook loads the clustering results and clinical/prognostic data for the cohort (e.g., from `prog_data/stad_tcga_pan_can_atlas_2018_clinical_data.tsv`).
      2.  **Calculates Heterogeneity Score:** It calculates a heterogeneity score for each tumor based on the number of distinct clusters.
      3.  **Performs Survival Analysis:** Using the `lifelines` library, it divides patients into "High" and "Low" heterogeneity groups to generate Kaplan-Meier survival curves and assess the prognostic impact.

  - **Correlation with Pathologist Assessment:** The `histrogy_analysis_ICGC.ipynb` notebook demonstrates that the subclone counts derived from HETMAP show a significant correlation with those evaluated by expert pathologists, confirming the clinical relevance of the automated assessment.

  - **Integration with Other Models:** The framework's prognostic prediction capabilities can be further enhanced by integrating HETMAP features with other models, such as TITAN.
