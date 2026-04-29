`nanoverl/config.py`

### 问题 1: 关于构建逻辑与未定义键的处理

*   **构建逻辑**: config.py  会先使用 config.py  将 `.json` 或 `.yaml` 文件读取为一个**普通的 Python 字典**。然后，它把这个字典传递给 config.py 。在 config.py  内部，它会对每个子配置块（如 `DataConfig`, `ModelConfig` 等）调用 config.py 。
*   **未定义的键会怎样？**: 现在会**直接报错**，不会被悄悄忽略。
    原因是 `_coerce_dataclass()` 会先计算输入 mapping 中不属于目标 dataclass 的字段：
    ```python
    field_names = set(cls.__dataclass_fields__)
    unknown_fields = sorted(set(value) - field_names)
    if unknown_fields:
        raise ConfigError(...)
    ```
    这样做的好处是配置写错时能尽早暴露。例如把 `"mini_batch_size"` 错写成 `"ppo_mini_batch_size"`，现在会在解析配置阶段失败，而不是默默走默认值。
    同理，当前没有公开的 `"ray"` 配置块；未来真正接入并行 infra 时，会重新设计明确的 runtime / placement 配置。

### 问题 2: 必填项与非必填项，以及示例

*   **如何体现必填和非必填**:
    在 Python 的 `@dataclass` 中，如果在声明字段时**没有赋予默认值**（如 `path: str`），那么它就是**必填项**（不传会引发 TypeError）。如果在声明时**赋予了默认值**（如 `use_remove_padding: bool = True`），那么它就是**选填项**。

    **但是，在这个 config.py 的实现中，所有的字段都提供了默认值！** (你可以在代码中看到，连 `ModelConfig` 的路径都有默认值 `path: str = "debug-model"`)。这意味着从代码解析（Parse）的层面来看，**完全没有任何配置是强制必填的**，你甚至传一个空的 `{}` 都能成功解析（它会全部走默认调试配置）。

    **真正的约束来自于后面的逻辑校验（`validate` 方法）**:
    虽然你能解析出来，但在执行 `cfg.validate()` 时，会有很多一致性校验和限制。例如，如果没有正确配置路径，后续模型加载肯定会失败。

### 全包含的示例 JSON 配置文件

为了方便你理解和复制，这里提供一个包含**所有可用配置项**的 JSON 示例。请注意注释中关于哪些项通常需要修改（逻辑上的“必填”）的说明：

```json
{
  "data": {
    // 【必须修改】你的训练数据和验证数据路径（JSONL 格式）
    "train_path": "your_data/train.jsonl",
    "val_path": "your_data/val.jsonl",
    
    // 【必须根据显存修改】全局或数据生成时的批次大小
    "train_batch_size": 16,
    "val_batch_size": 4,
    
    // 【选填】是否打乱数据，以及随机种子
    "shuffle": true,
    "seed": 42,
    
    // 【必须根据长度修改】最大输入Prompt长度和最大生成长度
    "max_prompt_length": 512,
    "max_response_length": 256
  },
  "model": {
    // 【必须修改】底座模型的路径（本地路径或 HuggingFace ID）
    "path": "meta-llama/Llama-3.2-1B",
    
    // 【选填】如果不填，会自动和 path 保持一致 (在 _apply_derived_defaults 中实现)
    "tokenizer_path": null,
    
    // 【选填】如果你希望 Actor 和 Critic 用不同的底座，在这里指定。通常不填。
    "critic_path": null,
    
    // 【选填】各种加载选项
    "trust_remote_code": false,
    "use_remove_padding": true,
    "dtype": "bfloat16", // 建议修改为 bfloat16
    "chat_template_path": null,
    "attn_implementation": "flash_attention_2" // 如果支持，强烈建议改用 flash attention
  },
  "algorithm": {
    // 【必须修改】使用的算法。可选: "ppo", "grpo", "rloo"
    "name": "ppo",
    
    // 【选填】优势估计方法，如果不填会根据 name 自动推导 (ppo -> gae, grpo -> grpo, rloo -> rloo)
    "advantage_estimator": "gae",
    
    // 【选填】RL 算法超参数
    "gamma": 1.0,
    "lam": 1.0,
    "use_kl_in_reward": false,
    "kl_penalty": "kl",
    "kl_coef": 0.05,
    "norm_adv_by_std_in_grpo": true
  },
  "actor": {
    // 【必须修改】后端引擎，实际训练必须改成 "hf" (HuggingFace) 或 "fsdp"
    "backend": "hf",
    "device": "cuda:0",
    
    // 【选填】策略 Loss 和更新参数
    "policy_loss": "ppo_clip",
    "mini_batch_size": 4, // 必须能被 micro_batch_size 整除
    "update_epochs": 1,
    "micro_batch_size": 2, // 必须设置以防 OOM
    "clip_ratio": 0.2,
    "clip_ratio_low": null,
    "clip_ratio_high": null,
    "clip_ratio_c": 3.0,
    "entropy_coeff": 0.001,
    "record_entropy": true,
    "use_kl_loss": false,
    "kl_loss_coef": 0.001,
    "loss_agg_mode": "token-mean",
    
    // 【选填】优化器参数
    "update_step_size": 0.02,
    "lr": 1e-6,
    "weight_decay": 0.01,
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "max_grad_norm": 1.0,
    "shuffle": true
  },
  "critic": {
    // 【必须修改】如果用 PPO，后端必须改成和 Actor 一致的 "hf" 或 "fsdp"
    "backend": "hf",
    "device": "cuda:0",
    
    // 【选填】是否启用 Critic (PPO 必须 true，GRPO/RLOO 可以 false)
    "enable": true,
    
    // 【选填】相关超参数，同 Actor
    "mini_batch_size": 4,
    "update_epochs": 1,
    "micro_batch_size": 2,
    "cliprange_value": 0.5,
    "loss_agg_mode": "token-mean",
    "lr": 5e-6,
    "weight_decay": 0.01,
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "max_grad_norm": 1.0,
    "shuffle": true
  },
  "reference": {
    // 【必须修改】如果开启，后端引擎必须跟 Actor 一致
    "backend": "hf",
    "device": "cuda:0",
    "enable": true, // 通常在 RLHF 中都需要参考模型
    "fixed_kl_offset": -0.15
  },
  "rollout": {
    // 【必须修改】用于生成回答的后端。如果用 HF 训练，这里写 "hf" 或 "vllm"
    "backend": "hf",
    "device": "cuda:0",
    "response_length": 256,
    
    // 【选填】训练和验证时的采样参数
    "train": {
      "temperature": 1.0,
      "top_p": 1.0,
      "top_k": -1,
      "do_sample": true,
      "n": 1 // 如果用 GRPO/RLOO，这里必须大于 1 (比如 4 或 8)
    },
    "validation": {
      "temperature": 0.0,
      "do_sample": false,
      "n": 1
    },
    "balance_by_length": false,
    "gpu_memory_utilization": 0.5, // 仅 VLLM 用
    "tensor_model_parallel_size": 1, // 仅 VLLM 用
    "enforce_eager": false,
    "engine_kwargs": {}
  },
  "reward": {
    // 【必须修改】定义你是如何打分的
    "type": "python", // 或部署的 reward_model 等
    "function_path": "your_reward_script.py",
    "function_name": "compute_reward",
    "reward_key": "data_source"
  },
  "trainer": {
    // 【必须修改】主要控制运行时和步数
    "total_epochs": 1,
    "total_training_steps": null, // 可以指定总步数代替 Epoch
    
    // 【选填】各种日志和保存频率
    "validate_before_train": true,
    "validate_only": false,
    "test_freq": 100, // 多少步验证一次
    "save_freq": 500, // 多少步保存一次检查点
    "critic_warmup": 0,
    "balance_batch": false,
    "train_dump_freq": 10,
    "validation_dump_freq": 10,
    "dump_max_rows": 8, // 打印生成内容的行数
    
    // 【必须修改】影响日志展示（如 wandb）和本地保存路径
    "project_name": "My_RLHF_Project",
    "experiment_name": "Llama3_PPO_Run1",
    "default_local_dir": "./checkpoints",
    
    "log_optimizer_steps": false,
    "loggers": ["console", "wandb"] // 如果需要 wandb 就加上
  }
}
```
