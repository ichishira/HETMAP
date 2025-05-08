# project for estimation of heterogeneity from WSI

## Repository Workflow: Extracting Patches, Embedding with UNI, and Clustering with Hetero_cluster

This repository implements a workflow consisting of the following steps: extracting patches from Whole Slide Images (WSIs), generating embeddings for each patch using the pathology image foundation model UNI, and clustering these patches using hetero_cluster.

### 1. Patch Extraction from WSIs

The `TCGA_wsi_patch` directory contains the code for extracting patches from WSIs. To maintain consistent magnification, 20x samples are extracted at a size of 224x224 pixels, and 40x samples are extracted at 448x448 pixels. The `make_files_for_patch.ipynb` notebook extracts the magnification level from the slides and outputs the samples for each magnification into separate text files. This code also includes the creation of symbolic links.

Executing the commands listed in `exec_patching.txt` in the terminal will initiate the patching process.

### 2. Embedding with UNI

The embedding of patches is implemented in `UNI_encode_with_mask.py` (and `UNI_encode_wo_mask.py` for ablation studies). Only patches where at least 1/4 of the area consists of epithelial components are encoded. To encode specific samples, the paths to these samples should be listed in `samples_to_encode.txt`. Running the `encode_all.sh` script will then encode all the specified samples.

### 3. Patch Clustering with Hetero_cluster

The `hetero_cluster` directory contains the code for clustering the extracted patches. The functions within the `func` subdirectory are imported and executed by `hetero_cluster.py`. `hetero_cluster_wo_potts.py` is used for ablation studies. To cluster specific samples, the paths to these samples should be listed in `samples_to_cluster.txt`. Executing `cluster_all.sh` will perform parallel clustering for multiple samples.

The `prog_est_clonenum_corr_chief_all.ipynb` notebook loads the `prog_data` and analyzes the impact of the cluster number on prognosis.

