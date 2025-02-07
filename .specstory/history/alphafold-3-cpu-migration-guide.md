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



---

