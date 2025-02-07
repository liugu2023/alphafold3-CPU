## SpecStory

## AlphaFold 3 CPU Migration Guide (2025/2/7 19:23:47)

_**User**_

迁移到CPU上

---

_**Assistant**_

我将帮助修改代码以在CPU上运行AlphaFold 3。主要需要修改以下几个部分:

1. 移除GPU相关的检查和配置
2. 将设备设置为CPU
3. 调整性能相关的参数

以下是具体修改:

```python:run_alphafold.py
// ... existing code ...

# 移除GPU设备相关的flag
_GPU_DEVICE = flags.DEFINE_integer(
    'gpu_device',
    None,  # 改为None默认值
    'Deprecated. AlphaFold 3 now runs on CPU only.',
)

// ... existing code ...

def main(_):
  if _JAX_COMPILATION_CACHE_DIR.value is not None:
    jax.config.update(
        'jax_compilation_cache_dir', _JAX_COMPILATION_CACHE_DIR.value
    )

  if _JSON_PATH.value is None == _INPUT_DIR.value is None:
    raise ValueError(
        'Exactly one of --json_path or --input_dir must be specified.'
    )

  if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
    raise ValueError(
        'At least one of --run_inference or --run_data_pipeline must be'
        ' set to true.'
    )

  // ... existing code ...

  if _RUN_INFERENCE.value:
    # 使用CPU设备
    devices = jax.local_devices(backend='cpu')
    print(f'Using CPU device: {devices[0]}')

    print('Building model from scratch...')
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation='xla',  # 在CPU上使用XLA实现
            num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
            num_recycles=_NUM_RECYCLES.value,
            return_embeddings=_SAVE_EMBEDDINGS.value,
        ),
        device=devices[0],  # 使用第一个CPU设备
        model_dir=pathlib.Path(MODEL_DIR.value),
    )
    # 检查模型参数是否可以加载
    print('Checking that model parameters can be loaded...')
    _ = model_runner.model_params
  else:
    model_runner = None

  // ... existing code ...
```

主要修改说明:

1. 移除了GPU设备相关的flag,将`_GPU_DEVICE`的默认值改为None

2. 移除了GPU计算能力的检查代码

3. 在创建ModelRunner时:
   - 使用`jax.local_devices(backend='cpu')`获取CPU设备
   - 将flash attention实现设置为'xla'(CPU兼容)
   - 使用第一个CPU设备

4. 保留了其他功能不变,包括:
   - 数据管道配置
   - 模型参数加载
   - 输入输出处理
   - 多seed支持等

需要注意:

1. 在CPU上运行会显著降低性能,推理时间会大幅增加

2. 内存使用可能会增加,建议适当减小batch size或其他参数

3. 某些优化功能(如flash attention)在CPU上可能无法使用最优实现

4. 建议在测试或开发环境使用CPU版本,生产环境仍推荐使用GPU版本

这些修改使AlphaFold 3可以在CPU上运行,但性能会受到影响。如果可能的话,仍建议使用GPU来获得更好的性能。

---

_**User**_

给出运行命令

---

_**Assistant**_

我来提供在CPU上运行AlphaFold 3的命令。以下是基本的运行命令和一些常用选项:

```bash
python run_alphafold.py \
  --json_path=/path/to/input.json \
  --output_dir=/path/to/output \
  --model_dir=/path/to/model_params \
  --db_dir=/path/to/databases \
  --flash_attention_implementation=xla \
  --num_recycles=3 \
  --num_diffusion_samples=1
```

主要参数说明:

1. 必需参数:
```bash
--json_path          # 输入JSON文件路径(与--input_dir二选一)
--output_dir         # 输出目录
```

2. 重要路径参数:
```bash
--model_dir          # 模型参数目录
--db_dir             # 数据库目录
```

3. CPU优化参数:
```bash
--flash_attention_implementation=xla  # 使用XLA实现
--num_recycles=3                     # 减少循环次数以加快运行
--num_diffusion_samples=1            # 减少扩散采样数量
```

4. 可选的性能调优参数:
```bash
--jackhmmer_n_cpu=4   # 限制MSA搜索的CPU使用数量
--nhmmer_n_cpu=4      # 限制RNA MSA搜索的CPU使用数量
```

5. 其他有用的选项:
```bash
--run_data_pipeline=true   # 是否运行数据预处理管道
--run_inference=true       # 是否运行推理
--save_embeddings=false    # 是否保存嵌入向量
```

使用建议:

1. 由于在CPU上运行,建议:
   - 减少`num_recycles`(默认10)
   - 减少`num_diffusion_samples`(默认5)
   - 适当限制CPU核心使用数量

2. 对于测试目的,可以使用最小配置:
```bash
python run_alphafold.py \
  --json_path=test_input.json \
  --output_dir=./output \
  --model_dir=./models \
  --db_dir=./databases \
  --num_recycles=1 \
  --num_diffusion_samples=1
```

3. 对于生产用途,可以使用更多资源:
```bash
python run_alphafold.py \
  --json_path=prod_input.json \
  --output_dir=./results \
  --model_dir=./models \
  --db_dir=./databases \
  --num_recycles=5 \
  --num_diffusion_samples=3 \
  --jackhmmer_n_cpu=8 \
  --nhmmer_n_cpu=8
```

请注意:
- 在CPU上运行会显著增加计算时间
- 建议先用小数据集测试
- 根据实际CPU性能和内存大小调整参数
- 确保有足够的磁盘空间存储输出结果

---

_**User**_

输入：/archive/liugu/iutput/processed    输出：/archive/liugu/output    模型权重：/archive/liugu/model    数据库：/archive/plato/task3/database

---

_**Assistant**_

根据您提供的具体路径，我来给出运行命令：

```bash
python run_alphafold.py \
  --input_dir=/archive/liugu/iutput/processed \
  --output_dir=/archive/liugu/output \
  --model_dir=/archive/liugu/model \
  --db_dir=/archive/plato/task3/database \
  --flash_attention_implementation=xla \
  --num_recycles=3 \
  --num_diffusion_samples=1 \
  --jackhmmer_n_cpu=8 \
  --nhmmer_n_cpu=8
```

参数说明：
1. `--input_dir`: 指定输入目录为 `/archive/liugu/iutput/processed`
2. `--output_dir`: 指定输出目录为 `/archive/liugu/output`
3. `--model_dir`: 指定模型权重目录为 `/archive/liugu/model`
4. `--db_dir`: 指定数据库目录为 `/archive/plato/task3/database`

性能优化参数：
- `--flash_attention_implementation=xla`: 使用CPU兼容的XLA实现
- `--num_recycles=3`: 设置较小的循环次数以加快运行
- `--num_diffusion_samples=1`: 减少扩散采样数量
- `--jackhmmer_n_cpu=8`: 设置MSA搜索的CPU核心数
- `--nhmmer_n_cpu=8`: 设置RNA MSA搜索的CPU核心数

建议：
1. 首先用小数据集测试运行是否正常
2. 如果运行速度太慢，可以进一步减少`num_recycles`和CPU使用数量
3. 确保输出目录有足够的磁盘空间
4. 建议添加日志输出，方便追踪运行状态：

```bash
python run_alphafold.py \
  --input_dir=/archive/liugu/iutput/processed \
  --output_dir=/archive/liugu/output \
  --model_dir=/archive/liugu/model \
  --db_dir=/archive/plato/task3/database \
  --flash_attention_implementation=xla \
  --num_recycles=3 \
  --num_diffusion_samples=1 \
  --jackhmmer_n_cpu=8 \
  --nhmmer_n_cpu=8 2>&1 | tee alphafold_run.log
```

这样会将输出同时打印到屏幕和保存到`alphafold_run.log`文件中。

---

_**User**_

Traceback (most recent call last):
  File "/archive/liugu/alphafold3-CPU/run_alphafold.py", line 715, in <module>
Using CPU device: TFRT_CPU_0
Building model from scratch...
Checking that model parameters can be loaded...
    app.run(main)
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/absl/app.py", line 308, in run
    _run_main(main, args)
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/absl/app.py", line 254, in _run_main
    sys.exit(main(argv))
             ^^^^^^^^^^
  File "/archive/liugu/alphafold3-CPU/run_alphafold.py", line 696, in main
    for fold_input in fold_inputs:
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/alphafold3/common/folding_input.py", line 1353, in load_fold_inputs_from_path
    with open(json_path, 'r') as f:
         ^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/archive/liugu/input/process/37aa_2JO9.json'

---

_**Assistant**_



看起来输入路径有错误。根据错误信息，系统在尝试访问 `/archive/liugu/input/process/37aa_2JO

---

_**User**_

Traceback (most recent call last):
  File "/archive/liugu/alphafold3-CPU/run_alphafold.py", line 715, in <module>
Using CPU device: TFRT_CPU_0
Building model from scratch...
Checking that model parameters can be loaded...
    app.run(main)
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/absl/app.py", line 308, in run
    _run_main(main, args)
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/absl/app.py", line 254, in _run_main
    sys.exit(main(argv))
             ^^^^^^^^^^
  File "/archive/liugu/alphafold3-CPU/run_alphafold.py", line 702, in main
    data_pipeline_config=data_pipeline_config,
                         ^^^^^^^^^^^^^^^^^^^^
NameError: name 'data_pipeline_config' is not defined

---

_**Assistant**_



---



---

