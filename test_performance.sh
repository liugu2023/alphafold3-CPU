#!/bin/bash

# 检查输入文件是否存在
INPUT_FILE="/archive/liugu/input/processed/37aa_2JO9.json"
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE does not exist!"
    exit 1
fi

# 基准目录
BASE_DIR="/archive/liugu/output/performance_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BASE_DIR
echo "Created base directory: $BASE_DIR"

# 测试不同的 recycles 次数
for recycles in 3 5 8 10; do
    echo "Testing with num_recycles = $recycles"
    
    # 创建输出目录
    OUTPUT_DIR="$BASE_DIR/recycles_${recycles}"
    mkdir -p $OUTPUT_DIR
    echo "Created output directory: $OUTPUT_DIR"
    
    TIME_LOG="$OUTPUT_DIR/time_log.txt"
    FULL_LOG="$OUTPUT_DIR/full_log.txt"
    
    # 记录开始时间
    echo "=== Test with num_recycles = $recycles ===" > $TIME_LOG
    echo "Start time: $(date)" >> $TIME_LOG
    START_TIME=$(date +%s)
    
    # 运行 AlphaFold
    python run_alphafold.py \
      --json_path=$INPUT_FILE \
      --output_dir=$OUTPUT_DIR \
      --model_dir=/archive/liugu/model \
      --db_dir=/archive/plato/task3/database \
      --num_recycles=$recycles \
      --num_diffusion_samples=5 \
      --jackhmmer_n_cpu=30 \
      --nhmmer_n_cpu=30 \
      2>&1 | tee $FULL_LOG
      
    # 记录结束时间和运行时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "End time: $(date)" >> $TIME_LOG
    echo "Total runtime: $DURATION seconds" >> $TIME_LOG
    
    # 提取推理时间
    INFERENCE_TIME=$(grep "Running model inference with seed 1 took" $FULL_LOG | awk '{print $8}')
    if [ -n "$INFERENCE_TIME" ]; then
        echo "num_recycles=$recycles: $INFERENCE_TIME seconds" >> "$BASE_DIR/comparison.txt"
    else
        echo "Warning: Could not extract inference time for num_recycles=$recycles"
        echo "num_recycles=$recycles: ERROR" >> "$BASE_DIR/comparison.txt"
    fi
    
    echo "Completed test with num_recycles = $recycles"
    echo "----------------------------------------"
done

# 打印比较结果
echo -e "\nPerformance comparison:"
if [ -f "$BASE_DIR/comparison.txt" ]; then
    cat "$BASE_DIR/comparison.txt"
else
    echo "No comparison results available"
fi

echo -e "\nTest results are saved in: $BASE_DIR" 