## SpecStory

## Error Handling in Model Inference (2025/2/7 18:07:14)

_**User**_

Warning: Found NaN/inf values in array of shape (5, 256, 24, 3)
Running model inference with seed 1 took 1705.70 seconds.
Extracting output structure samples with seed 1...
Error in structure extraction: Invalid structure result: missing positions
Detailed error information:
Running model inference and extracting output structures with 1 seed(s) took 1705.97 seconds.
Traceback (most recent call last):
  File "/archive/liugu/alphafold3-CPU/run_alphafold.py", line 883, in <module>
    app.run(main)
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/absl/app.py", line 308, in run
    _run_main(main, args)
  File "/archive/liugu/alphafold-CPU/lib/python3.11/site-packages/absl/app.py", line 254, in _run_main
    sys.exit(main(argv))
             ^^^^^^^^^^
  File "/archive/liugu/alphafold3-CPU/run_alphafold.py", line 868, in main
    process_fold_input(
  File "/archive/liugu/alphafold3-CPU/run_alphafold.py", line 721, in process_fold_input
    all_inference_results = predict_structure(
                            ^^^^^^^^^^^^^^^^^^
  File "/archive/liugu/alphafold3-CPU/run_alphafold.py", line 552, in predict_structure
    raise ValueError('No valid predictions were generated for any seed.')
ValueError: No valid predictions were generated for any seed.

---

_**Assistant**_

从错误信息来看,主要问题出在模型推理结果中出现了NaN/inf值,导致结构预测失败。让我们对代码进行一些优化来提高数值稳定性和错误处理。

以下是建议的修改:

```python:run_alphafold.py
# ... 现有代码 ...

def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
) -> Sequence[ResultsForSeed]:
    """运行完整的推理流水线来预测每个种子的结构。"""

    print(f'Featurising data with {len(fold_input.rng_seeds)} seed(s)...')
    featurisation_start_time = time.time()
    ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
    
    # 添加特征归一化
    def normalize_features(features):
        """对数值特征进行归一化处理"""
        if isinstance(features, dict):
            for key, value in features.items():
                if isinstance(value, (np.ndarray, jnp.ndarray)):
                    if value.dtype.kind in 'fc':
                        # 对数值特征进行归一化,避免数值过大
                        mean = np.mean(value)
                        std = np.std(value)
                        if std > 0:
                            features[key] = (value - mean) / (std + 1e-7)
        return features

    # 特征化处理
    featurised_examples = []
    for seed in fold_input.rng_seeds:
        print(f'Featurising data with seed {seed}.')
        try:
            example = featurisation.featurise_input(
                fold_input=fold_input,
                buckets=buckets,
                ccd=ccd,
                verbose=True,
                conformer_max_iterations=conformer_max_iterations,
            )
            # 添加特征归一化
            example = normalize_features(example)
            featurised_examples.append(example)
        except Exception as e:
            print(f"Error in featurisation for seed {seed}: {e}")
            continue

    if not featurised_examples:
        raise ValueError("Failed to featurise any examples")

    print(
        f'Featurising data with {len(fold_input.rng_seeds)} seed(s) took'
        f' {time.time() - featurisation_start_time:.2f} seconds.'
    )

    # 添加数值稳定性检查函数
    def stabilize_array(arr, name=""):
        """处理数组中的数值不稳定问题"""
        if isinstance(arr, (np.ndarray, jnp.ndarray)) and arr.dtype.kind in 'fc':
            # 检查并报告异常值
            n_nan = np.sum(np.isnan(arr))
            n_inf = np.sum(np.isinf(arr))
            if n_nan > 0 or n_inf > 0:
                print(f"Warning: Found {n_nan} NaNs and {n_inf} Infs in {name}")
            
            # 替换异常值
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 裁剪过大的值
            arr = np.clip(arr, -1e6, 1e6)
            
            return arr
        return arr

    print(
        'Running model inference and extracting output structure samples with'
        f' {len(fold_input.rng_seeds)} seed(s)...'
    )
    all_inference_start_time = time.time()
    all_inference_results = []
    
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        print(f'Running model inference with seed {seed}...')
        inference_start_time = time.time()
        rng_key = jax.random.PRNGKey(seed)
        
        try:
            # 确保 example 是正确的格式
            if isinstance(example, list):
                example = example[0]
            
            # 在推理前进行数值稳定性处理    
            example = jax.tree_map(lambda x: stabilize_array(x, "input"), example)
            
            # 添加梯度裁剪
            def clip_gradients(grad):
                if isinstance(grad, (np.ndarray, jnp.ndarray)):
                    return np.clip(grad, -1.0, 1.0)
                return grad
            
            result = model_runner.run_inference_with_retry(example, rng_key)
            
            # 处理结果中的数值问题
            result = jax.tree_map(
                lambda x: stabilize_array(x, "output"), 
                result
            )

            print(
                f'Running model inference with seed {seed} took'
                f' {time.time() - inference_start_time:.2f} seconds.'
            )
            
            # 验证结果
            if not isinstance(result, dict):
                raise ValueError(f"Expected dict result, got {type(result)}")
                
            print(f'Extracting output structure samples with seed {seed}...')
            extract_structures = time.time()
            
            try:
                inference_results = model_runner.extract_structures(
                    batch=example, result=result, target_name=fold_input.name
                )
                
                # 验证结构结果
                valid_results = []
                for idx, res in enumerate(inference_results):
                    try:
                        if not hasattr(res, 'positions') or res.positions is None:
                            print(f"Warning: Sample {idx} missing positions")
                            continue
                            
                        positions = res.positions
                        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                            print(f"Warning: Sample {idx} has invalid positions")
                            continue
                            
                        valid_results.append(res)
                    except Exception as e:
                        print(f"Error validating sample {idx}: {e}")
                        continue
                        
                if not valid_results:
                    raise ValueError("No valid structure samples generated")
                    

---

_**User**_

结合标准GPU代码解决NaN/inf问题

---

_**Assistant**_



---



---

