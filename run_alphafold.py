# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 structure prediction script.

AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

To request access to the AlphaFold 3 model parameters, follow the process set
out at https://github.com/google-deepmind/alphafold3. You may only use these
if received directly from Google. Use is subject to terms of use available at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""

import os
# 禁用ROCM和TPU检测
os.environ['JAX_PLATFORMS'] = 'cpu'  # 只使用CPU
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用CUDA
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'  # 强制使用CPU

# 设置CPU优化参数
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX2'

from collections.abc import Callable, Sequence
import csv
import dataclasses
import datetime
import functools
import multiprocessing
import os
import pathlib
import shutil
import string
import textwrap
import time
import typing
from typing import overload, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
import psutil
import gc
from contextlib import contextmanager
import hashlib
import json
import queue

from absl import app
from absl import flags
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.constants import chemical_components
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.jax.attention import attention
from alphafold3.model import features
from alphafold3.model import model
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import utils
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np

# 在导入语句后，主程序开始前设置
multiprocessing.set_start_method('spawn', force=True)

_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
_DEFAULT_MODEL_DIR = _HOME_DIR / 'models'
_DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'


# Input and output paths.
_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to a directory where the results will be saved.',
)
MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    _DEFAULT_MODEL_DIR.as_posix(),
    'Path to the model to use for inference.',
)

# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    True,
    'Whether to run the data pipeline on the fold inputs.',
)
_RUN_INFERENCE = flags.DEFINE_bool(
    'run_inference',
    True,
    'Whether to run inference on the fold inputs.',
)

# Binary paths.
_JACKHMMER_BINARY_PATH = flags.DEFINE_string(
    'jackhmmer_binary_path',
    shutil.which('jackhmmer'),
    'Path to the Jackhmmer binary.',
)
_NHMMER_BINARY_PATH = flags.DEFINE_string(
    'nhmmer_binary_path',
    shutil.which('nhmmer'),
    'Path to the Nhmmer binary.',
)
_HMMALIGN_BINARY_PATH = flags.DEFINE_string(
    'hmmalign_binary_path',
    shutil.which('hmmalign'),
    'Path to the Hmmalign binary.',
)
_HMMSEARCH_BINARY_PATH = flags.DEFINE_string(
    'hmmsearch_binary_path',
    shutil.which('hmmsearch'),
    'Path to the Hmmsearch binary.',
)
_HMMBUILD_BINARY_PATH = flags.DEFINE_string(
    'hmmbuild_binary_path',
    shutil.which('hmmbuild'),
    'Path to the Hmmbuild binary.',
)

# Database paths.
DB_DIR = flags.DEFINE_multi_string(
    'db_dir',
    (_DEFAULT_DB_DIR.as_posix(),),
    'Path to the directory containing the databases. Can be specified multiple'
    ' times to search multiple directories in order.',
)

_SMALL_BFD_DATABASE_PATH = flags.DEFINE_string(
    'small_bfd_database_path',
    '${DB_DIR}/bfd-first_non_consensus_sequences.fasta',
    'Small BFD database path, used for protein MSA search.',
)
_MGNIFY_DATABASE_PATH = flags.DEFINE_string(
    'mgnify_database_path',
    '${DB_DIR}/mgy_clusters_2022_05.fa',
    'Mgnify database path, used for protein MSA search.',
)
_UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = flags.DEFINE_string(
    'uniprot_cluster_annot_database_path',
    '${DB_DIR}/uniprot_all_2021_04.fa',
    'UniProt database path, used for protein paired MSA search.',
)
_UNIREF90_DATABASE_PATH = flags.DEFINE_string(
    'uniref90_database_path',
    '${DB_DIR}/uniref90_2022_05.fa',
    'UniRef90 database path, used for MSA search. The MSA obtained by '
    'searching it is used to construct the profile for template search.',
)
_NTRNA_DATABASE_PATH = flags.DEFINE_string(
    'ntrna_database_path',
    '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
    'NT-RNA database path, used for RNA MSA search.',
)
_RFAM_DATABASE_PATH = flags.DEFINE_string(
    'rfam_database_path',
    '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
    'Rfam database path, used for RNA MSA search.',
)
_RNA_CENTRAL_DATABASE_PATH = flags.DEFINE_string(
    'rna_central_database_path',
    '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    'RNAcentral database path, used for RNA MSA search.',
)
_PDB_DATABASE_PATH = flags.DEFINE_string(
    'pdb_database_path',
    '${DB_DIR}/mmcif_files',
    'PDB database directory with mmCIF files path, used for template search.',
)
_SEQRES_DATABASE_PATH = flags.DEFINE_string(
    'seqres_database_path',
    '${DB_DIR}/pdb_seqres_2022_09_28.fasta',
    'PDB sequence database path, used for template search.',
)

# Number of CPUs to use for MSA tools.
_JACKHMMER_N_CPU = flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Jackhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)
_NHMMER_N_CPU = flags.DEFINE_integer(
    'nhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Nhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)

# Template search configuration.
_MAX_TEMPLATE_DATE = flags.DEFINE_string(
    'max_template_date',
    '2021-09-30',  # By default, use the date from the AlphaFold 3 paper.
    'Maximum template release date to consider. Format: YYYY-MM-DD. All '
    'templates released after this date will be ignored.',
)

_CONFORMER_MAX_ITERATIONS = flags.DEFINE_integer(
    'conformer_max_iterations',
    None,  # Default to RDKit default parameters value.
    'Optional override for maximum number of iterations to run for RDKit '
    'conformer search.',
)

# JAX inference performance tuning.
_JAX_COMPILATION_CACHE_DIR = flags.DEFINE_string(
    'jax_compilation_cache_dir',
    None,
    'Path to a directory for the JAX compilation cache.',
)
_BUCKETS = flags.DEFINE_list(
    'buckets',
    # pyformat: disable
    ['256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072',
     '3584', '4096', '4608', '5120'],
    # pyformat: enable
    'Strictly increasing order of token sizes for which to cache compilations.'
    ' For any input with more tokens than the largest bucket size, a new bucket'
    ' is created for exactly that number of tokens.',
)
_NUM_RECYCLES = flags.DEFINE_integer(
    'num_recycles',
    10,
    'Number of recycles to use during inference.',
    lower_bound=1,
)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    'num_diffusion_samples',
    5,
    'Number of diffusion samples to generate.',
    lower_bound=1,
)
_NUM_SEEDS = flags.DEFINE_integer(
    'num_seeds',
    None,
    'Number of seeds to use for inference. If set, only a single seed must be'
    ' provided in the input JSON. AlphaFold 3 will then generate random seeds'
    ' in sequence, starting from the single seed specified in the input JSON.'
    ' The full input JSON produced by AlphaFold 3 will include the generated'
    ' random seeds. If not set, AlphaFold 3 will use the seeds as provided in'
    ' the input JSON.',
    lower_bound=1,
)

# Output controls.
_SAVE_EMBEDDINGS = flags.DEFINE_bool(
    'save_embeddings',
    False,
    'Whether to save the final trunk single and pair embeddings in the output.',
)


def make_model_config(
    *,
    flash_attention_implementation: attention.Implementation = 'triton',
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
) -> model.Model.Config:
  """Returns a model config with some defaults overridden."""
  config = model.Model.Config()
  config.global_config.flash_attention_implementation = (
      flash_attention_implementation
  )
  config.heads.diffusion.eval.num_samples = num_diffusion_samples
  config.num_recycles = num_recycles
  config.return_embeddings = return_embeddings
  return config


class CacheManager:
    """管理计算结果缓存"""
    def __init__(self, cache_dir: Optional[pathlib.Path] = None):
        self._memory_cache: Dict[str, Any] = {}
        self._cache_dir = cache_dir or pathlib.Path('/tmp/alphafold_cache')
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        def make_hashable(obj):
            if isinstance(obj, np.ndarray):
                # 对于numpy数组，使用其形状和内容的哈希值
                return f"array_shape={obj.shape}_hash={hash(obj.tobytes())}"
            if isinstance(obj, (dict, list, set)):
                # 递归处理嵌套结构
                if isinstance(obj, dict):
                    return f"dict_{hash(tuple((k, make_hashable(v)) for k, v in sorted(obj.items())))}"
                return f"{type(obj).__name__}_{hash(tuple(make_hashable(x) for x in obj))}"
            if hasattr(obj, '__dict__'):
                # 处理自定义对象
                return f"{obj.__class__.__name__}_{hash(str(obj.__dict__))}"
            return str(obj)
        
        # 组合所有参数的哈希值
        key_parts = []
        for arg in args:
            try:
                key_parts.append(make_hashable(arg))
            except Exception as e:
                print(f"Warning: Failed to hash argument {type(arg)}: {e}")
                key_parts.append(f"unhashable_{type(arg).__name__}")
        
        for k, v in sorted(kwargs.items()):
            try:
                key_parts.append(f"{k}={make_hashable(v)}")
            except Exception as e:
                print(f"Warning: Failed to hash kwarg {k}: {e}")
                key_parts.append(f"{k}=unhashable_{type(v).__name__}")
        
        # 生成最终的哈希值
        combined = "_".join(key_parts)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存的结果"""
        # 先检查内存缓存
        if key in self._memory_cache:
            self._hits += 1
            return self._memory_cache[key]
        
        # 检查磁盘缓存
        cache_file = self._cache_dir / f"{key}.npz"
        if cache_file.exists():
            self._hits += 1
            try:
                with np.load(cache_file, allow_pickle=True) as data:
                    result = {k: data[k] for k in data.files}
                    self._memory_cache[key] = result
                    return result
            except Exception as e:
                print(f"Failed to load cache: {e}")
        
        self._misses += 1
        return None
    
    def put(self, key: str, value: Any):
        """存储结果到缓存"""
        try:
            self._memory_cache[key] = value
            cache_file = self._cache_dir / f"{key}.npz"
            
            # 确保值是可以被npz存储的格式
            save_dict = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    save_dict[k] = v
                elif isinstance(v, (int, float, bool, str)):
                    save_dict[k] = np.array([v])
                else:
                    print(f"Warning: Skipping non-serializable value for key {k}")
            
            if save_dict:
                np.savez_compressed(cache_file, **save_dict)
            
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def get_stats(self) -> Tuple[int, int]:
        """返回缓存命中和未命中次数"""
        return self._hits, self._misses


class ModelRunner:
    """添加缓存机制的ModelRunner"""
    def __init__(
        self,
        config: model.Model.Config,
        device: jax.Device,
        model_dir: pathlib.Path,
    ):
        self._model_config = config
        self._device = device
        self._model_dir = model_dir
        self._cache_manager = CacheManager()
        self._param_cache = {}
    
    @functools.lru_cache(maxsize=32)
    def _get_cached_params(self, param_key: str):
        """缓存模型参数"""
        if param_key not in self._param_cache:
            self._param_cache[param_key] = params.get_model_haiku_params(
                model_dir=self._model_dir
            )
        return self._param_cache[param_key]
    
    @functools.cached_property
    def _model(self) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
        """Loads model parameters and returns a jitted model forward pass."""
        @hk.transform
        def forward_fn(batch):
            return model.Model(self._model_config)(batch)

        return functools.partial(
            jax.jit(forward_fn.apply, device=self._device), 
            self._get_cached_params('')
        )
    
    def run_inference(
        self,
        featurised_example: features.BatchDict,
        rng_key: jnp.ndarray,
    ) -> model.ModelResult:
        """运行推理，添加结果缓存"""
        cache_key = self._cache_manager._get_cache_key(
            featurised_example,
            rng_key,
            self._model_config
        )
        
        # 检查缓存
        cached_result = self._cache_manager.get(cache_key)
        if cached_result is not None:
            print("Using cached result")
            return cached_result
        
        # 计算新结果
        featurised_example = jax.device_put(
            jax.tree_util.tree_map(
                jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
            ),
            self._device,
        )

        result = self._model(rng_key, featurised_example)
        result = self.check_output_numerics(result)
        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
            result,
        )
        result = dict(result)
        identifier = self._get_cached_params('')['__meta__']['__identifier__'].tobytes()
        result['__identifier__'] = identifier
        
        # 最后再次检查关键坐标数据
        if 'structure_module' in result:
            for key in ['final_atom_positions', 'final_atom_mask']:
                if key in result['structure_module']:
                    result['structure_module'][key] = self.fix_numerics(
                        result['structure_module'][key], 
                        f'structure_module.{key}'
                    )
        
        # 缓存结果
        self._cache_manager.put(cache_key, result)
        
        return result

    @staticmethod
    def fix_numerics(value, key=""):
        """修复数值问题的通用方法"""
        if isinstance(value, (np.ndarray, jnp.ndarray)):
          if np.any(np.isnan(value)) or np.any(np.isinf(value)):
            print(f"Warning: Found NaN/inf in output {key}")
            print(f"Shape: {value.shape}")
            print(f"NaN count: {np.isnan(value).sum()}")
            print(f"Inf count: {np.isinf(value).sum()}")
            
            # 对于坐标相关的键，使用更保守的替换值
            if any(coord in key.lower() for coord in ['coord', 'pos', 'xyz', 'position']):
              # 使用邻近的有效值填充
              mask = np.isnan(value) | np.isinf(value)
              if mask.any():
                valid_values = value[~mask]
                if len(valid_values) > 0:
                  # 使用有效值的平均值
                  fill_value = np.mean(valid_values)
                else:
                  # 如果没有有效值，使用0
                  fill_value = 0.0
                value = np.where(mask, fill_value, value)
            else:
              # 对于其他值使用标准替换
              value = np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
          
          # 确保数值在合理范围内
          value = np.clip(value, -1e6, 1e6)
        return value

    def check_output_numerics(self, result):
        """检查并修复输出中的数值问题"""
        def process_dict(d, parent_key=""):
          for key, value in d.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
              d[key] = process_dict(value, full_key)
            else:
              d[key] = self.fix_numerics(value, full_key)
          return d

        return process_dict(result)

    def extract_structures(
        self,
        batch: features.BatchDict,
        result: model.ModelResult,
        target_name: str,
    ) -> list[model.InferenceResult]:
        """Generates structures from model outputs."""
        return list(
            model.Model.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )

    def extract_embeddings(
        self,
        result: model.ModelResult,
    ) -> dict[str, np.ndarray] | None:
        """Extracts embeddings from model outputs."""
        embeddings = {}
        if 'single_embeddings' in result:
          embeddings['single_embeddings'] = result['single_embeddings']
        if 'pair_embeddings' in result:
          embeddings['pair_embeddings'] = result['pair_embeddings']
        return embeddings or None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
  """Stores the inference results (diffusion samples) for a single seed.

  Attributes:
    seed: The seed used to generate the samples.
    inference_results: The inference results, one per sample.
    full_fold_input: The fold input that must also include the results of
      running the data pipeline - MSA and templates.
    embeddings: The final trunk single and pair embeddings, if requested.
  """

  seed: int
  inference_results: Sequence[model.InferenceResult]
  full_fold_input: folding_input.Input
  embeddings: dict[str, np.ndarray] | None = None


def create_model_runner(model_config, model_dir):
    """在每个进程中创建ModelRunner，添加计算优化"""
    # 设置JAX优化选项
    jax.config.update('jax_enable_x64', False)  # 使用float32以提高性能
    jax.config.update('jax_default_matmul_precision', 'bfloat16')  # 使用bfloat16加速矩阵运算
    
    # 设置CPU线程数
    num_physical_cores = psutil.cpu_count(logical=False)  # 物理核心数
    threads_per_core = 2  # 每个核心的线程数
    
    # 为每个进程分配合适的线程数
    process_threads = max(1, num_physical_cores // 2)
    
    # 设置线程亲和性和线程数
    os.environ['OMP_NUM_THREADS'] = str(process_threads)
    os.environ['MKL_NUM_THREADS'] = str(process_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(process_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(process_threads)
    
    # 启用Intel MKL优化
    os.environ['MKL_DEBUG_CPU_TYPE'] = '5'  # 启用高级指令集
    os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX2'  # 使用AVX2指令集
    
    # 设置CPU亲和性策略
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    
    # 优化内存分配
    os.environ['MKL_DYNAMIC'] = 'FALSE'  # 禁用动态调整以提高稳定性
    
    print(f"Optimized thread settings:")
    print(f"Physical cores: {num_physical_cores}")
    print(f"Threads per process: {process_threads}")
    
    # 使用CPU设备
    devices = jax.local_devices(backend='cpu')
    print(f'Creating model runner with CPU device: {devices[0]}')
    
    # 创建优化后的ModelRunner
    model_runner = ModelRunner(
        config=model_config,
        device=devices[0],
        model_dir=model_dir,
    )
    
    return model_runner


@contextmanager
def memory_tracker(operation_name: str):
    """跟踪操作前后的内存使用情况"""
    process = psutil.Process()
    start_mem = process.memory_info().rss
    start_time = time.time()
    try:
        yield
    finally:
        end_mem = process.memory_info().rss
        end_time = time.time()
        print(f"\n{operation_name} Memory Usage:")
        print(f"Start: {start_mem / 1024 / 1024:.2f} MB")
        print(f"End: {end_mem / 1024 / 1024:.2f} MB")
        print(f"Diff: {(end_mem - start_mem) / 1024 / 1024:.2f} MB")
        print(f"Time: {end_time - start_time:.2f} seconds")


def predict_structure_for_seed(
    seed: int,
    featurised_example: features.BatchDict,
    model_config: model.Model.Config,
    model_dir: pathlib.Path,
    fold_input: folding_input.Input,
) -> ResultsForSeed:
    """处理单个seed的预测，添加缓存优化"""
    
    with memory_tracker("Model Runner Creation"):
        model_runner = create_model_runner(model_config, model_dir)
    
    # 清理不需要的缓存
    gc.collect()
    
    with memory_tracker("Model Inference"):
        print(f'Running model inference with seed {seed}...')
        inference_start_time = time.time()
        
        # 使用确定性的随机数生成
        rng_key = jax.random.PRNGKey(seed)
        
        # 运行推理，使用缓存
        result = model_runner.run_inference(featurised_example, rng_key)
        
        # 打印缓存统计
        hits, misses = model_runner._cache_manager.get_stats()
        print(f"Cache stats - hits: {hits}, misses: {misses}")
        
        print(
            f'Running model inference with seed {seed} took'
            f' {time.time() - inference_start_time:.2f} seconds.'
        )
    
    # 提取需要的数据后释放大型中间结果
    with memory_tracker("Structure Extraction"):
        print(f'Extracting output structure samples with seed {seed}...')
        extract_structures = time.time()
        
        # 使用优化后的结构提取
        inference_results = model_runner.extract_structures(
            batch=featurised_example, 
            result=result,
            target_name=fold_input.name
        )
        
        print(
            f'Extracting {len(inference_results)} output structure samples with'
            f' seed {seed} took {time.time() - extract_structures:.2f} seconds.'
        )
        
        embeddings = model_runner.extract_embeddings(result)
        
        # 释放大型中间结果
        del result
        del model_runner
        gc.collect()
    
    return ResultsForSeed(
        seed=seed,
        inference_results=inference_results,
        full_fold_input=fold_input,
        embeddings=embeddings,
    )


def worker_process(
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    model_config: model.Model.Config,
    model_dir: pathlib.Path,
):
    """工作进程处理函数"""
    try:
        while True:
            try:
                # 非阻塞方式获取任务
                task = task_queue.get_nowait()
            except queue.Empty:
                # 没有任务时退出
                break
            
            seed, featurised_example, fold_input = task
            
            try:
                with memory_tracker(f"Worker Process {os.getpid()}"):
                    result = predict_structure_for_seed(
                        seed=seed,
                        featurised_example=featurised_example,
                        model_config=model_config,
                        model_dir=model_dir,
                        fold_input=fold_input,
                    )
                    result_queue.put((seed, result))
            except Exception as e:
                print(f"Error in worker {os.getpid()} processing seed {seed}: {e}")
                result_queue.put((seed, e))
            
            # 主动清理内存
            gc.collect()
            
    except Exception as e:
        print(f"Worker process {os.getpid()} failed: {e}")
        raise

def predict_structure(
    fold_input: folding_input.Input,
    model_config: model.Model.Config,
    model_dir: pathlib.Path,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
) -> Sequence[ResultsForSeed]:
    """并行运行推理管道来预测每个seed的结构，添加动态负载均衡"""
    
    with memory_tracker("Data Featurisation"):
        print(f'Featurising data with {len(fold_input.rng_seeds)} seed(s)...')
        featurisation_start_time = time.time()
        ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
        featurised_examples = featurisation.featurise_input(
            fold_input=fold_input,
            buckets=buckets,
            ccd=ccd,
            verbose=True,
            conformer_max_iterations=conformer_max_iterations,
        )
        print(
            f'Featurising data with {len(fold_input.rng_seeds)} seed(s) took'
            f' {time.time() - featurisation_start_time:.2f} seconds.'
        )
    
    # 根据系统资源动态调整进程数
    available_memory = psutil.virtual_memory().available
    memory_per_process = 2 * 1024 * 1024 * 1024  # 估计每个进程2GB内存
    max_processes_by_memory = max(1, available_memory // memory_per_process)
    
    num_cpus = multiprocessing.cpu_count()
    num_workers = min(
        len(fold_input.rng_seeds),
        max(1, num_cpus - 1),
        max_processes_by_memory
    )
    
    print(
        f'Running parallel model inference with {num_workers} workers for'
        f' {len(fold_input.rng_seeds)} seed(s)...'
    )
    print(f'Available memory: {available_memory / 1024 / 1024:.2f} MB')
    print(f'Estimated memory per process: {memory_per_process / 1024 / 1024:.2f} MB')
    
    # 创建任务队列和结果队列
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    
    # 将任务放入队列
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        task_queue.put((seed, example, fold_input))
    
    # 创建并启动工作进程
    processes = []
    for _ in range(num_workers):
        p = multiprocessing.Process(
            target=worker_process,
            args=(task_queue, result_queue, model_config, model_dir)
        )
        p.start()
        processes.append(p)
    
    # 收集结果
    all_inference_results = []
    completed_seeds = set()
    total_seeds = len(fold_input.rng_seeds)
    
    start_time = time.time()
    last_progress_time = start_time
    
    try:
        while len(completed_seeds) < total_seeds:
            try:
                seed, result = result_queue.get(timeout=300)  # 5分钟超时
                
                if isinstance(result, Exception):
                    print(f"Error processing seed {seed}: {result}")
                    continue
                
                all_inference_results.append(result)
                completed_seeds.add(seed)
                
                # 每30秒打印一次进度
                current_time = time.time()
                if current_time - last_progress_time > 30:
                    elapsed = current_time - start_time
                    completed = len(completed_seeds)
                    remaining = total_seeds - completed
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = remaining / rate if rate > 0 else 0
                    
                    print(f"Progress: {completed}/{total_seeds} seeds completed")
                    print(f"Rate: {rate:.2f} seeds/second")
                    print(f"ETA: {eta/60:.1f} minutes")
                    
                    last_progress_time = current_time
                
            except queue.Empty:
                print("Warning: No results received for 5 minutes")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
        for p in processes:
            p.terminate()
    finally:
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        # 清理队列
        while not task_queue.empty():
            try:
                task_queue.get_nowait()
            except queue.Empty:
                break
        
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                break
    
    # 按seed排序结果
    all_inference_results.sort(key=lambda x: x.seed)
    
    print(
        'Running parallel model inference and extracting output structures with'
        f' {len(fold_input.rng_seeds)} seed(s) took'
        f' {time.time() - start_time:.2f} seconds.'
    )
    
    return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
  """Writes the input JSON to the output directory."""
  os.makedirs(output_dir, exist_ok=True)
  path = os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json')
  print(f'Writing model input JSON to {path}')
  with open(path, 'wt') as f:
    f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
  """Writes outputs to the specified output directory."""
  ranking_scores = []
  max_ranking_score = None
  max_ranking_result = None

  output_terms = (
      pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
  ).read_text()

  os.makedirs(output_dir, exist_ok=True)
  for results_for_seed in all_inference_results:
    seed = results_for_seed.seed
    for sample_idx, result in enumerate(results_for_seed.inference_results):
      sample_dir = os.path.join(output_dir, f'seed-{seed}_sample-{sample_idx}')
      os.makedirs(sample_dir, exist_ok=True)
      post_processing.write_output(
          inference_result=result, output_dir=sample_dir
      )
      ranking_score = float(result.metadata['ranking_score'])
      ranking_scores.append((seed, sample_idx, ranking_score))
      if max_ranking_score is None or ranking_score > max_ranking_score:
        max_ranking_score = ranking_score
        max_ranking_result = result

    if embeddings := results_for_seed.embeddings:
      embeddings_dir = os.path.join(output_dir, f'seed-{seed}_embeddings')
      os.makedirs(embeddings_dir, exist_ok=True)
      post_processing.write_embeddings(
          embeddings=embeddings, output_dir=embeddings_dir
      )

  if max_ranking_result is not None:  # True iff ranking_scores non-empty.
    post_processing.write_output(
        inference_result=max_ranking_result,
        output_dir=output_dir,
        # The output terms of use are the same for all seeds/samples.
        terms_of_use=output_terms,
        name=job_name,
    )
    # Save csv of ranking scores with seeds and sample indices, to allow easier
    # comparison of ranking scores across different runs.
    with open(os.path.join(output_dir, 'ranking_scores.csv'), 'wt') as f:
      writer = csv.writer(f)
      writer.writerow(['seed', 'sample', 'ranking_score'])
      writer.writerows(ranking_scores)


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_config: None,
    model_dir: None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> folding_input.Input:
    ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_config: model.Model.Config,
    model_dir: pathlib.Path,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
    ...


def replace_db_dir(path_with_db_dir: str, db_dirs: Sequence[str]) -> str:
  """Replaces the DB_DIR placeholder in a path with the given DB_DIR."""
  template = string.Template(path_with_db_dir)
  if 'DB_DIR' in template.get_identifiers():
    for db_dir in db_dirs:
      path = template.substitute(DB_DIR=db_dir)
      if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f'{path_with_db_dir} with ${{DB_DIR}} not found in any of {db_dirs}.'
    )
  if not os.path.exists(path_with_db_dir):
    raise FileNotFoundError(f'{path_with_db_dir} does not exist.')
  return path_with_db_dir


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_config: model.Model.Config | None,
    model_dir: pathlib.Path | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
) -> folding_input.Input | Sequence[ResultsForSeed]:
    """Runs data pipeline and/or inference on a single fold input.

    Args:
        fold_input: Fold input to process.
        data_pipeline_config: Data pipeline config to use. If None, skip the data
            pipeline.
        model_config: Model configuration to use. If None, skip inference.
        model_dir: Path to model directory. Required if model_config is provided.
        output_dir: Output directory to write to.
        buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
            of the model. If None, calculate the appropriate bucket size from the
            number of tokens.
        conformer_max_iterations: Optional override for maximum number of iterations
            to run for RDKit conformer search.

    Returns:
        The processed fold input, or the inference results for each seed.

    Raises:
        ValueError: If the fold input has no chains.
    """
    print(f'\nRunning fold job {fold_input.name}...')

    if not fold_input.chains:
        raise ValueError('Fold input has no chains.')

    if os.path.exists(output_dir) and os.listdir(output_dir):
        new_output_dir = (
            f'{output_dir}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        print(
            f'Output will be written in {new_output_dir} since {output_dir} is'
            ' non-empty.'
        )
        output_dir = new_output_dir
    else:
        print(f'Output will be written in {output_dir}')

    if data_pipeline_config is None:
        print('Skipping data pipeline...')
    else:
        print('Running data pipeline...')
        fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

    write_fold_input_json(fold_input, output_dir)
    if model_config is None:
        print('Skipping model inference...')
        output = fold_input
    else:
        print(
            f'Predicting 3D structure for {fold_input.name} with'
            f' {len(fold_input.rng_seeds)} seed(s)...'
        )
        all_inference_results = predict_structure(
            fold_input=fold_input,
            model_config=model_config,
            model_dir=model_dir,
            buckets=buckets,
            conformer_max_iterations=conformer_max_iterations,
        )
        print(f'Writing outputs with {len(fold_input.rng_seeds)} seed(s)...')
        write_outputs(
            all_inference_results=all_inference_results,
            output_dir=output_dir,
            job_name=fold_input.sanitised_name(),
        )
        output = all_inference_results

    print(f'Fold job {fold_input.name} done, output written to {output_dir}\n')
    return output


def split_fold_input(fold_input: folding_input.Input, max_tokens: int = 1000) -> list[folding_input.Input]:
    """将大型输入分割成多个小份"""
    total_tokens = sum(len(chain.sequence) for chain in fold_input.chains)
    if total_tokens <= max_tokens:
        return [fold_input]
    
    # 计算需要分成几份
    num_splits = (total_tokens + max_tokens - 1) // max_tokens
    print(f"Input size {total_tokens} tokens exceeds {max_tokens}, splitting into {num_splits} parts")
    
    # 分割chains
    splits = []
    current_tokens = 0
    current_chains = []
    
    for chain in fold_input.chains:
        chain_tokens = len(chain.sequence)
        if current_tokens + chain_tokens > max_tokens and current_chains:
            # 创建新的fold_input，只传递必要的参数
            split_input = folding_input.Input(
                name=f"{fold_input.name}_part{len(splits)}",
                chains=current_chains.copy(),
                rng_seeds=fold_input.rng_seeds,
                user_ccd=fold_input.user_ccd if hasattr(fold_input, 'user_ccd') else None,
            )
            splits.append(split_input)
            current_chains = []
            current_tokens = 0
        
        current_chains.append(chain)
        current_tokens += chain_tokens
    
    # 添加最后一部分
    if current_chains:
        split_input = folding_input.Input(
            name=f"{fold_input.name}_part{len(splits)}",
            chains=current_chains,
            rng_seeds=fold_input.rng_seeds,
            user_ccd=fold_input.user_ccd if hasattr(fold_input, 'user_ccd') else None,
        )
        splits.append(split_input)
    
    return splits

def process_fold_input_parallel(
    fold_inputs: list[folding_input.Input],
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_config: model.Model.Config | None,
    model_dir: pathlib.Path | None,
    output_dir: str,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
    max_workers: int = 80,  # 默认使用80个核心
) -> None:
    """并行处理多个fold inputs"""
    
    # 创建进程池
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        for fold_input in fold_inputs:
            future = executor.submit(
                process_fold_input,
                fold_input=fold_input,
                data_pipeline_config=data_pipeline_config,
                model_config=model_config,
                model_dir=model_dir,
                output_dir=os.path.join(output_dir, fold_input.sanitised_name()),
                buckets=buckets,
                conformer_max_iterations=conformer_max_iterations,
            )
            futures.append((fold_input.name, future))
        
        # 等待所有任务完成
        for name, future in futures:
            try:
                future.result()
                print(f"Completed processing {name}")
            except Exception as e:
                print(f"Error processing {name}: {e}")

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

  notice = textwrap.wrap(
      'Running AlphaFold 3. Please note that standard AlphaFold 3 model'
      ' parameters are only available under terms of use provided at'
      ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
      ' If you do not agree to these terms and are using AlphaFold 3 derived'
      ' model parameters, cancel execution of AlphaFold 3 inference with'
      ' CTRL-C, and do not use the model parameters.',
      break_long_words=False,
      break_on_hyphens=False,
      width=80,
  )
  print('\n' + '\n'.join(notice) + '\n')

  if _RUN_DATA_PIPELINE.value:
    expand_path = lambda x: replace_db_dir(x, DB_DIR.value)
    max_template_date = datetime.date.fromisoformat(_MAX_TEMPLATE_DATE.value)
    data_pipeline_config = pipeline.DataPipelineConfig(
        jackhmmer_binary_path=_JACKHMMER_BINARY_PATH.value,
        nhmmer_binary_path=_NHMMER_BINARY_PATH.value,
        hmmalign_binary_path=_HMMALIGN_BINARY_PATH.value,
        hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH.value,
        hmmbuild_binary_path=_HMMBUILD_BINARY_PATH.value,
        small_bfd_database_path=expand_path(_SMALL_BFD_DATABASE_PATH.value),
        mgnify_database_path=expand_path(_MGNIFY_DATABASE_PATH.value),
        uniprot_cluster_annot_database_path=expand_path(
            _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH.value
        ),
        uniref90_database_path=expand_path(_UNIREF90_DATABASE_PATH.value),
        ntrna_database_path=expand_path(_NTRNA_DATABASE_PATH.value),
        rfam_database_path=expand_path(_RFAM_DATABASE_PATH.value),
        rna_central_database_path=expand_path(_RNA_CENTRAL_DATABASE_PATH.value),
        pdb_database_path=expand_path(_PDB_DATABASE_PATH.value),
        seqres_database_path=expand_path(_SEQRES_DATABASE_PATH.value),
        jackhmmer_n_cpu=_JACKHMMER_N_CPU.value,
        nhmmer_n_cpu=_NHMMER_N_CPU.value,
        max_template_date=max_template_date,
    )
  else:
    data_pipeline_config = None

  if _RUN_INFERENCE.value:
    # 创建模型配置
    model_config = make_model_config(
        flash_attention_implementation='xla',  # CPU只支持XLA实现
        num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
        num_recycles=_NUM_RECYCLES.value,
        return_embeddings=_SAVE_EMBEDDINGS.value,
    )
    model_dir = pathlib.Path(MODEL_DIR.value)
    
    # 检查模型参数是否可以加载
    print('Checking that model parameters can be loaded...')
    test_runner = create_model_runner(model_config, model_dir)
    _ = test_runner._get_cached_params('')
  else:
    model_config = None
    model_dir = None

  # 在处理fold_inputs之前添加分割逻辑
  all_split_inputs = []
  for fold_input in fold_inputs:
      if _NUM_SEEDS.value is not None:
          print(f'Expanding fold job {fold_input.name} to {_NUM_SEEDS.value} seeds')
          fold_input = fold_input.with_multiple_seeds(_NUM_SEEDS.value)
      
      # 分割大型输入
      split_inputs = split_fold_input(fold_input, max_tokens=1000)
      all_split_inputs.extend(split_inputs)
  
  # 使用并行处理
  num_workers = min(80, len(all_split_inputs))  # 最多使用80个核心
  print(f"Processing {len(all_split_inputs)} fold inputs using {num_workers} workers")
  
  process_fold_input_parallel(
      fold_inputs=all_split_inputs,
      data_pipeline_config=data_pipeline_config,
      model_config=model_config,
      model_dir=model_dir,
      output_dir=_OUTPUT_DIR.value,
      buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
      conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
      max_workers=num_workers,
  )

  print(f'Done running {len(all_split_inputs)} fold jobs.')


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)