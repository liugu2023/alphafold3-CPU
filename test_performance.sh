#!/bin/bash

# 基准目录
BASE_DIR="/archive/liugu/output/performance_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BASE_DIR

# 测试不同的 recycles 次数
for recycles in 3 5 8 10; do
    echo "Testing with num_recycles = $recycles"
    
    OUTPUT_DIR="$BASE_DIR/recycles_${recycles}"
    TIME_LOG="$OUTPUT_DIR/time_log.txt"
    
    python run_alphafold.py \
      --json_path=/archive/liugu/input/process/37aa_2JO9.json \
      --output_dir=$OUTPUT_DIR \
      --model_dir=/archive/liugu/model \
      --db_dir=/archive/plato/task3/database \
      --num_recycles=$recycles \
      --num_diffusion_samples=5 \
      --jackhmmer_n_cpu=30 \
      --nhmmer_n_cpu=30 \
      2>&1 | tee "$OUTPUT_DIR/full_log.txt"
      
    # 提取推理时间
    INFERENCE_TIME=$(grep "Running model inference with seed 1 took" "$OUTPUT_DIR/full_log.txt" | awk '{print $8}')
    echo "num_recycles=$recycles: $INFERENCE_TIME seconds" >> "$BASE_DIR/comparison.txt"
done

# 打印比较结果
echo "Performance comparison:"
cat "$BASE_DIR/comparison.txt" 