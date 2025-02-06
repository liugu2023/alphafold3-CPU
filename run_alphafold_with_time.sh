#!/bin/bash

# 设置运行时间日志文件
TIME_LOG="/archive/liugu/output/time_log_$(date +%Y%m%d_%H%M%S).txt"
FULL_LOG="/archive/liugu/output/alphafold_$(date +%Y%m%d_%H%M%S).log"

# 记录开始时间
echo "=== AlphaFold Run Time Log ===" > $TIME_LOG
echo "Start time: $(date)" >> $TIME_LOG
START_TIME=$(date +%s)

# 运行 AlphaFold 并同时记录输出
python run_alphafold.py \
  --json_path=/archive/liugu/input/process/37aa_2JO9.json \
  --output_dir=/archive/liugu/output \
  --model_dir=/archive/liugu/model \
  --db_dir=/archive/plato/task3/database \
  --num_recycles=10 \
  --num_diffusion_samples=5 \
  --jackhmmer_n_cpu=30 \
  --nhmmer_n_cpu=30 \
  2>&1 | tee $FULL_LOG

# 记录结束时间和总运行时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 转换总时间为更易读的格式
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

# 写入时间信息
echo "End time: $(date)" >> $TIME_LOG
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> $TIME_LOG
echo "Total seconds: ${DURATION}" >> $TIME_LOG

# 从完整日志中提取关键时间信息
echo -e "\nDetailed timing breakdown:" >> $TIME_LOG
grep "took.*seconds" $FULL_LOG >> $TIME_LOG

# 提取并排序耗时最长的5个步骤
echo -e "\nTop 5 most time-consuming steps:" >> $TIME_LOG
grep "took.*seconds" $FULL_LOG | 
  sed 's/.*\([0-9]\+\.[0-9]\+\) seconds.*/\1 &/' |  # 将秒数移到行首以便排序
  sort -rn |  # 按数字降序排序
  head -n 5 |  # 取前5行
  sed 's/^[0-9.]\+ //' >> $TIME_LOG  # 删除行首的秒数

# 打印时间信息到屏幕
echo "----------------------------------------"
echo "Run completed!"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo -e "\nTop 5 most time-consuming steps:"
grep "took.*seconds" $FULL_LOG | 
  sed 's/.*\([0-9]\+\.[0-9]\+\) seconds.*/\1 &/' |
  sort -rn |
  head -n 5 |
  sed 's/^[0-9.]\+ //' |
  while IFS= read -r line; do
    echo "  $line"
  done
echo -e "\nDetailed timing information saved to: $TIME_LOG"
echo "Full log saved to: $FULL_LOG"
echo "----------------------------------------" 