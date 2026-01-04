set -euo pipefail

# 基本ディレクトリのパス
SLIDEDIR="/wsi/analysis/TCGA_TMA_analysis/data/WSI/STAD/"
H5DIR="/data/shirasuna/work/hetero_path/TCGA_wsi_patch/patched_STAD_mag20/patches/"
FEATUREDIR="/data/shirasuna/work/hetero_path/UNI_encode_with_mask/features_STAD_mag20/"

SAMPLES_FILE="STAD_mag20_all.txt"
SCRIPT="/data/shirasuna/work/hetero_path/hetero_cluster/hetero_cluster.py"  # フルパスを指定
SAVE_DIR="STAD_clustering_pca80var_integrate_mag20_init20_8"
LOG_FILE="error_log_init20.txt"

max_run_files=30

# ログファイルをクリア
> "$LOG_FILE"

# サンプルを順番に処理
for SAMPLE in $(cat "$SAMPLES_FILE"); do
  # 現在のジョブ数がmax_run_files未満の場合に実行
  while [ $(jobs | wc -l) -ge "$max_run_files" ]; do
    sleep 5
  done

  # 並列処理を実行
  echo "Running sample: $SAMPLE"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python "$SCRIPT" -slidedir "$SLIDEDIR" -h5dir "$H5DIR" -featuredir "$FEATUREDIR" -sample "$SAMPLE" -maxcluster 20 -savedir "$SAVE_DIR" -seed 314 || {
    echo "$SAMPLE" | tee -a "$LOG_FILE"
  } &
done

# 全てのジョブが終了するまで待機
wait

echo "All jobs completed (errors are logged in $LOG_FILE)"

