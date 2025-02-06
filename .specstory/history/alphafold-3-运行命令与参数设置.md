## SpecStory

## AlphaFold 3 运行命令与参数设置 (2025/2/6 17:32:56)

_**User**_

给出运行命令

---

_**Assistant**_

我将为您提供运行 AlphaFold 3 的命令。根据代码分析,这是一个针对 CPU 优化的版本。

以下是基本的运行命令:

```bash
python run_alphafold.py \
  --json_path=/path/to/input.json \
  --output_dir=/path/to/output \
  --model_dir=/path/to/models \
  --db_dir=/path/to/databases
```

主要参数说明:

1. 必需参数:
- `--output_dir`: 输出目录路径(必填)
- `--json_path` 或 `--input_dir`: 二选一
  - `--json_path`: 单个输入 JSON 文件路径
  - `--input_dir`: 包含多个输入 JSON 文件的目录路径

2. 重要可选参数:
- `--model_dir`: 模型文件目录,默认为 `$HOME/models`
- `--db_dir`: 数据库目录,默认为 `$HOME/public_databases`
- `--num_recycles`: 推理时的循环次数,默认为 10
- `--num_diffusion_samples`: 生成的扩散样本数,默认为 5
- `--num_seeds`: 使用的随机种子数量

3. 性能相关参数:
- `--jackhmmer_n_cpu`: Jackhmmer 使用的 CPU 核心数,默认为 min(cpu_count, 8)
- `--nhmmer_n_cpu`: Nhmmer 使用的 CPU 核心数,默认为 min(cpu_count, 8)

一个完整的示例命令:

```bash
python run_alphafold.py \
  --json_path=/path/to/input.json \
  --output_dir=/path/to/output \
  --model_dir=/path/to/models \
  --db_dir=/path/to/databases \
  --num_recycles=10 \
  --num_diffusion_samples=5 \
  --num_seeds=3 \
  --jackhmmer_n_cpu=8 \
  --nhmmer_n_cpu=8
```

注意事项:
1. 这是针对 CPU 优化的版本,已移除 GPU 相关配置
2. 需要确保已安装所有依赖包
3. 需要下载模型文件和数据库文件
4. 输入 JSON 文件需要符合 AlphaFold 3 的格式要求
5. 输出目录会自动创建,如果目录非空会自动创建新目录

您可以根据实际需求调整这些参数。如果您需要了解更多参数的详细信息,可以运行:

```bash
python run_alphafold.py --help
```

---

_**User**_

输入：/archive/liugu/input/process    输出：/archive/liugu/output    模型权重：/archive/liugu/model    数据库：/archive/plato/task3/database

---

_**Assistant**_

根据您提供的具体路径，以下是运行命令：

```bash
python run_alphafold.py \
  --input_dir=/archive/liugu/input/process \
  --output_dir=/archive/liugu/output \
  --model_dir=/archive/liugu/model \
  --db_dir=/archive/plato/task3/database \
  --num_recycles=10 \
  --num_diffusion_samples=5 \
  --jackhmmer_n_cpu=8 \
  --nhmmer_n_cpu=8
```

主要参数说明：
1. `--input_dir=/archive/liugu/input/process` - 指定包含输入 JSON 文件的目录
2. `--output_dir=/archive/liugu/output` - 指定输出结果的目录
3. `--model_dir=/archive/liugu/model` - 指定模型权重文件所在目录
4. `--db_dir=/archive/plato/task3/database` - 指定数据库文件所在目录

这个命令会：
1. 从 `/archive/liugu/input/process` 读取所有输入 JSON 文件
2. 将结果输出到 `/archive/liugu/output` 
3. 使用 `/archive/liugu/model` 中的模型权重
4. 使用 `/archive/plato/task3/database` 中的数据库文件
5. 每个结构预测使用 10 次循环
6. 每个输入生成 5 个扩散样本
7. Jackhmmer 和 Nhmmer 各使用 8 个 CPU 核心

注意事项：
1. 请确保所有指定的目录都存在且有正确的读写权限
2. 输出目录如果已存在且非空，程序会自动创建一个带时间戳的新目录
3. 这是 CPU 版本，会自动使用可用的 CPU 资源
4. 确保数据库目录包含所有必需的数据库文件

---

_**User**_

我要一共分配60个CPU核心

---

_**Assistant**_



根据代码分析，主要的 CPU 密集型任务是 Jackhmmer 和 Nhmmer 的 MSA 搜索。我们可以合理分配这 60 个 CPU 核心。以下是优化后的运行命令：

```bash
python run_alphafold.py \
  --input_dir=/archive/liugu/input/process \
  --output_dir=/archive/liugu/output \
  --model_dir=/archive/liugu/model \
  --db_dir=/archive/plato/task3/database \
  --num_recycles=10 \
  --num_diffusion_samples=5 \
  --jackhmmer_n_cpu=30 \
  --nhmmer_n_cpu=30
```

CPU 核心分配说明：
1. `--jackhmmer_n_cpu=30` - 分配 30 个核心给 Jackhmmer 进行蛋白质 MSA 搜索
2. `--nhmmer_n_cpu=30` - 分配 30 个核心给 Nhmmer 进行 RNA MSA 搜索

这样的分配方案：
1. 总共使用 60 个 CPU 核心
2. 在 Jackhmmer 和 Nhmmer 之间平均分配资源
3. 两个工具可以并行运行，充分利用计算资源
4. 避免单个工具占用过多资源导致系统负载不均衡

注意事项：
1. 原代码中默认将 Jackhmmer 和 Nhmmer 的 CPU 核心数限制为 min(cpu_count, 8)，我们修改了这个限制以充分利用 60 个核心
2. 如果发现系统负载过高，可以适当调整分配比例，比如：
   ```bash
   --jackhmmer_n_cpu=25 --nhmmer_n_cpu=25
   ```
   这样会预留一些核心给系统其他进程
3. 如果主要处理蛋白质结构预测，可以增加 Jackhmmer 的核心分配：
   ```bash
   --jackhmmer_n_cpu=40 --nhmmer_n_cpu=20
   ```

您可以根据实际任务类型和系统负载情况，适当调整这两个参数的值，但总和建议不超过 60。

---

_**User**_

更改为输入文件37aa_2JO9.json

---

_**Assistant**_



好的，我将命令更改为使用单个输入 JSON 文件。以下是更新后的运行命令：

```bash
python run_alphafold.py \
  --json_path=/archive/liugu/input/process/37aa_2JO9.json \
  --output_dir=/archive/liugu/output \
  --model_dir=/archive/liugu/model \
  --db_dir=/archive/plato/task3/database \
  --num_recycles=10 \
  --num_diffusion_samples=5 \
  --jackhmmer_n_cpu=30 \
  --nhmmer_n_cpu=30
```

主要改动：
1. 将 `--input_dir` 改为 `--json_path`
2. 指定具体的输入文件 `37aa_2JO9.json`

其他参数保持不变：
1. 输出目录：`/archive/liugu/output`
2. 模型目录：`/archive/liugu/model`
3. 数据库目录：`/archive/plato/task3/database`
4. CPU 分配：共 60 个核心，Jackhmmer 和 Nhmmer 各 30 个
5. 每个结构预测使用 10 次循环
6. 每个输入生成 5 个扩散样本

注意事项：
1. 请确保 `37aa_2JO9.json` 文件存在且格式正确
2. 输出将保存在 `/archive/liugu/output/37aa_2JO9` 目录下（基于输入文件名自动创建）
3. 如果输出目录已存在且非空，程序会自动创建一个带时间戳的新目录

---

_**User**_

I0206 17:51:47.570915 140308961452736 xla_bridge.py:895] Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
I0206 17:51:47.572212 140308961452736 xla_bridge.py:895] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
Traceback (most recent call last):
  File "/archive/liugu/alphafold3-CPU/run_alphafold.py", line 789, in <module>
    app.run(main)
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/absl/app.py", line 308, in run
    _run_main(main, args)
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/absl/app.py", line 254, in _run_main
    sys.exit(main(argv))
             ^^^^^^^^^^
  File "/archive/liugu/alphafold3-CPU/run_alphafold.py", line 677, in main
    gpu_devices = jax.local_devices(backend='gpu')
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/jax/_src/xla_bridge.py", line 1141, in local_devices
    process_index = get_backend(backend).process_index()
                    ^^^^^^^^^^^^^^^^^^^^
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/jax/_src/xla_bridge.py", line 1017, in get_backend
    return _get_backend_uncached(platform)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/jax/_src/xla_bridge.py", line 998, in _get_backend_uncached
    platform = canonicalize_platform(platform)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/jax/_src/xla_bridge.py", line 801, in canonicalize_platform
    raise RuntimeError(f"Unknown backend: '{platform}' requested, but no "
RuntimeError: Unknown backend: 'gpu' requested, but no platforms that are instances of gpu are present. Platforms present are: cpu

---

_**Assistant**_



看起来代码在尝试使用 GPU 设备时出错了。让我修改代码以正确使用 CPU。需要修改 `run_alphafold.py` 文件中的相关部分：

```python:run_alphafold.py
# ... existing code ...

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

  if _INPUT_DIR.value is not None:
    fold_inputs = folding_input.load_fold_inputs_from_dir(
        pathlib.Path(_INPUT_DIR.value)
    )
  elif _JSON_PATH.value is not None:
    fold_inputs = folding_input.load_fold_inputs_from_path(
        pathlib.Path(_JSON_PATH.value)
    )
  else:
    raise AssertionError(
        'Exactly one of --json_path or --input_dir must be specified.'
    )

  # Make sure we can create the output directory before running anything.
  try:
    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
  except OSError as e:
    print(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
    raise

  if _RUN_INFERENCE.value:
    # 使用 CPU 设备
    devices = jax.local_devices(backend='cpu')
    print(f'Using CPU device: {devices[0]}')
    
    print('Building model from scratch...')
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation='xla',  # CPU 使用 xla
            num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
            num_recycles=_NUM_RECYCLES.value,
            return_embeddings=_SAVE_EMBEDDINGS.value,
        ),
        device=devices[0],  # 使用第一个 CPU 设备
        model_dir=pathlib.Path(MODEL_DIR.value),
    )
    # 检查模型参数
    print('Checking we can load the model parameters...')
    _ = model_runner.model_params
  else:
    model_runner = None

  # ... rest of the code ...
```

主要修改：
1. 移除了 GPU 相关的代码
2. 直接使用 CPU 设备
3. 将 flash attention 实现设置为 'xla'（CPU 兼容）

现在您可以使用相同的命令运行：

```bash
python run_alphafold.py \
  --json_path=/archive/liugu/input/process/37aa_2JO9.json \
  --output_dir=/archive/liugu/output \
  --model_dir=/archive/liugu/model \
  --db_dir=/archive/plato/task3/database \
  --num_recycles=10 \
  --num_diffusion_samples=5 \
  --jackhmmer_n_cpu=30 \
  --nhmmer_n_cpu=30
```

这些修改应该能解决 GPU 相关的错误，让程序正确地在 CPU 上运行。程序会：
1. 自动检测并使用可用的 CPU 设备
2. 使用 XLA 作为 attention 实现
3. 正确分配 CPU 核心给 Jackhmmer 和 Nhmmer

---

_**User**_

再看看

---

_**Assistant**_

我注意到需要修改 `run_alphafold.py` 中的设备初始化部分。让我们修改相关代码：

```python:run_alphafold.py
# ... existing code ...

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

  if _INPUT_DIR.value is not None:
    fold_inputs = folding_input.load_fold_inputs_from_dir(
        pathlib.Path(_INPUT_DIR.value)
    )
  elif _JSON_PATH.value is not None:
    fold_inputs = folding_input.load_fold_inputs_from_path(
        pathlib.Path(_JSON_PATH.value)
    )
  else:
    raise AssertionError(
        'Exactly one of --json_path or --input_dir must be specified.'
    )

  try:
    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
  except OSError as e:
    print(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
    raise

  if _RUN_INFERENCE.value:
    # 直接使用 CPU 设备
    device = jax.devices('cpu')[0]
    print(f'Using CPU device: {device}')
    
    print('Building model from scratch...')
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=_FLASH_ATTENTION_IMPLEMENTATION.value,
            num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
            num_recycles=_NUM_RECYCLES.value,
            return_embeddings=_SAVE_EMBEDDINGS.value,
        ),
        device=device,
        model_dir=pathlib.Path(MODEL_DIR.value),
    )
    print('Checking we can load the model parameters...')
    _ = model_runner.model_params
  else:
    model_runner = None

  # ... rest of the code ...
```

主要修改：
1. 移除了 `gpu_devices` 相关代码
2. 使用 `jax.devices('cpu')[0]` 直接获取 CPU 设备
3. 保留了原有的 flash attention 实现配置

现在可以继续使用相同的运行命令：

```bash
python run_alphafold.py \
  --json_path=/archive/liugu/input/process/37aa_2JO9.json \
  --output_dir=/archive/liugu/output \
  --model_dir=/archive/liugu/model \
  --db_dir=/archive/plato/task3/database \
  --num_recycles=10 \
  --num_diffusion_samples=5 \
  --jackhmmer_n_cpu=30 \
  --nhmmer_n_cpu=30
```

这样的修改应该能解决 GPU 后端初始化的错误，让程序正确地在 CPU 上运行。程序会：
1. 直接使用 CPU 设备而不尝试初始化 GPU
2. 正确配置模型运行环境
3. 按指定分配 CPU 核心给 MSA 搜索工具

---

_**User**_

将运行时输出保存到log文件

---

_**Assistant**_

我将修改运行命令，将输出重定向到日志文件：

```bash
python run_alphafold.py \
  --json_path=/archive/liugu/input/process/37aa_2JO9.json \
  --output_dir=/archive/liugu/output \
  --model_dir=/archive/liugu/model \
  --db_dir=/archive/plato/task3/database \
  --num_recycles=10 \
  --num_diffusion_samples=5 \
  --jackhmmer_n_cpu=30 \
  --nhmmer_n_cpu=30 \
  2>&1 | tee /archive/liugu/output/alphafold_$(date +%Y%m%d_%H%M%S).log
```

主要改动：
1. 添加了 `2>&1 | tee /archive/liugu/output/alphafold_$(date +%Y%m%d_%H%M%S).log`
   - `2>&1` 将标准错误也重定向到标准输出
   - `| tee` 将输出同时显示在屏幕上并写入文件
   - 日志文件名包含时间戳，格式为 `alphafold_YYYYMMDD_HHMMSS.log`

这样：
1. 所有输出（包括错误信息）都会被保存到日志文件中
2. 同时您也能在终端看到实时输出
3. 日志文件会保存在输出目录下，便于后续查看
4. 每次运行都会生成一个新的日志文件，不会覆盖之前的日志

如果您想将日志保存到其他位置，可以修改日志文件路径，例如：
```bash
2>&1 | tee /path/to/your/log/directory/alphafold_$(date +%Y%m%d_%H%M%S).log
```

---

_**User**_

I0206 17:54:45.839266 140105580143296 xla_bridge.py:895] Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
I0206 17:54:45.840124 140105580143296 xla_bridge.py:895] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
Using CPU device: TFRT_CPU_0
Building model from scratch...
Checking we can load the model parameters...
Traceback (most recent call last):
  File "/archive/liugu/alphafold3-CPU/run_alphafold.py", line 711, in <module>
    app.run(main)
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/absl/app.py", line 308, in run
    _run_main(main, args)
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/absl/app.py", line 254, in _run_main
    sys.exit(main(argv))
             ^^^^^^^^^^
  File "/archive/liugu/alphafold3-CPU/run_alphafold.py", line 698, in main
    data_pipeline_config=data_pipeline_config,
                         ^^^^^^^^^^^^^^^^^^^^
NameError: name 'data_pipeline_config' is not defined

---



---

