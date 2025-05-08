#!/bin/bash

SAMPLES_FILE="samples_to_encode.txt"
CANCER_TYPE="BRCA"
SCRIPT="UNI_embedding.py"
LOG_FILE="error_log.txt"
SAVE_DIR="features_BRCA_mag40"
LEVEL=40

# テキストファイルが存在しない場合は終了
if [ ! -f "$SAMPLES_FILE" ]; then
    echo "Error: $SAMPLES_FILE not found!"
    exit 1
fi

# ログファイルをクリア
> "$LOG_FILE"

# サンプルごとにスクリプトを実行
while IFS= read -r sample; do

    # 実行＆エラーが出たらログに記録
    python "$SCRIPT" -sample "$sample" -cancertype "$CANCER_TYPE" -savedir "$SAVE_DIR" -level "$LEVEL" || {
        echo "Error occurred with sample: $sample" | tee -a "$LOG_FILE"
        continue  # エラー時は次のサンプルへスキップ
    }

done < "$SAMPLES_FILE"

echo "All samples processed (skipped errors)."
echo "Errors are logged in $LOG_FILE"

