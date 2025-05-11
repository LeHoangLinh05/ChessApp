#!/bin/bash
#
python train.py \
D:/AI/Chess/archive/nodes5000pv2_UHO.binpack \
D:/AI/Chess/archive/nodes5000pv2_UHO.binpack \
 --gpus 1 \
 --threads 2 \
 --batch-size 8096 \
 --progress_bar_refresh_rate 20 \
 --smart-fen-skipping \
 --random-fen-skipping 10 \
 --features=HalfKP^ \
 --lambda=1.0 \
 --max_epochs=300
#TRAIN_SCRIPT="D:/AI/nnue-pytorch/train.py"
#
#TRAIN_DATA_FILES=("D:/AI/Chess/archive/nodes5000pv2_UHO.binpack") # Hoặc nhiều file
## VAL_DATA_FILES=("../data/your_validation_file.binpack") # Tùy chọn
#VAL_DATA_FILES=()
#
#OUTPUT_DIR="D:/AI/nnue-pytorch/output/training_run_$(date +%Y%m%d_%H%M%S)" # Sử dụng dấu / thay vì \ cho tính tương thích
#GPU_CONFIG="1" # Hoặc "0"
#TORCH_THREADS=-1 # Tự động
#NUM_DATALOADER_WORKERS=1
#BATCH_SIZE=8096
#MAX_EPOCHS=300
#
## === FEATURES ===
## Lựa chọn 1: Sử dụng feature mặc định (rất tốt)
#FEATURES_NAME="HalfKAv2_hm^"
## Lựa chọn 2: Nếu "HalfKP^" là một tên hợp lệ được định nghĩa
## FEATURES_NAME="HalfKP^"
## Lựa chọn 3: Một feature block khác từ get_available_feature_blocks_names()
## FEATURES_NAME="SomeOtherValidFeatureBlockName"
#
#NNUE_LAMBDA=1.0
#LEARNING_RATE=8.75e-4 # Hoặc giá trị bạn muốn thử
#
#RANDOM_FEN_SKIPPING_VAL=10
#
## === XÂY DỰNG LỆNH ===
#CMD="python ${TRAIN_SCRIPT}" # Bắt đầu lệnh với python và tên script
#
## Thêm các tệp dữ liệu huấn luyện
#for f in "${TRAIN_DATA_FILES[@]}"; do
#  CMD+=" \"${f}\"" # Nối tên tệp dữ liệu (được đặt trong dấu ngoặc kép) VÀO SAU tên script
#done
#
## Thêm các tệp dữ liệu kiểm định nếu có
#if [ ${#VAL_DATA_FILES[@]} -gt 0 ]; then
#  CMD+=" --validation-data" # Thêm cờ
#  for f in "${VAL_DATA_FILES[@]}"; do
#    CMD+=" \"${f}\"" # Nối tên tệp dữ liệu kiểm định
#  done
#fi
#
## Thêm các tham số tùy chọn khác
#PYTHON_ARGS+=(
#  "--default_root_dir" "${OUTPUT_DIR}"
#  "--gpus" "${GPU_CONFIG}"
#  "--threads" "${TORCH_THREADS}"
#  "--num-workers" "${NUM_DATALOADER_WORKERS}"
#  "--batch-size" "${BATCH_SIZE}"
#  "--max_epochs" "${MAX_EPOCHS}"
#  "--features" "${FEATURES_NAME}"
#  "--lambda=${NNUE_LAMBDA}" # Hoặc "--lambda" "${NNUE_LAMBDA}"
#  "--lr=${LEARNING_RATE}"   # Hoặc "--lr" "${LEARNING_RATE}"
#  "--random-fen-skipping" "${RANDOM_FEN_SKIPPING_VAL}"
#  "--progress_bar_refresh_rate" "20"
#)
#
#echo "Executing command:"
## Sử dụng printf để hiển thị lệnh một cách an toàn hơn, đặc biệt nếu có ký tự đặc biệt
#printf "%q " python "${PYTHON_ARGS[@]}" # Lệnh này sẽ in ra lệnh gần giống như cách nó được thực thi
#echo # Thêm dòng mới
#
## Thực thi lệnh python với mảng các đối số
#python "${PYTHON_ARGS[@]}"
#
#echo "Training script finished."