"""
Configuration loader and validator.
محمّل وموثّق الإعدادات
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


@dataclass
class ModelConfig:
    base_model: str = "microsoft/Phi-4-multimodal-instruct"
    quantization_enabled: bool = True
    quantization_bits: int = 4
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    double_quant: bool = True
    output_dir: str = "./models/fine-tuned"
    adapter_dir: str = "./models/adapters"
    merged_dir: str = "./models/merged"
    max_length: int = 4096
    max_new_tokens: int = 2048


@dataclass
class AudioConfig:
    model: str = "openai/whisper-large-v3"
    language: str = "ar"
    device: str = "cuda"
    compute_type: str = "float16"


@dataclass
class LoRAConfig:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50
    save_total_limit: int = 3
    dataset_path: str = "./data/processed"
    train_split: float = 0.9
    seed: int = 42


@dataclass
class GRPOConfig:
    enabled: bool = False
    num_generations: int = 8
    max_completion_length: int = 1024
    max_prompt_length: int = 512
    learning_rate: float = 1e-5
    temperature: float = 0.7
    warmup_ratio: float = 0.1
    beta: float = 0.0                          # KL coeff (0.0 = modern standard)
    epsilon: float = 0.2                       # PPO clip range
    loss_type: str = "dapo"                    # grpo, dapo, dr_grpo, bnpo
    scale_rewards: str = "batch"               # false, true, "batch"
    mask_truncated_completions: bool = True
    reasoning_format: str = "xml"              # xml, think, none
    reward_functions: list = field(default_factory=lambda: [
        "accuracy", "format", "coherence"
    ])


@dataclass
class RAGConfig:
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    embedding_dim: int = 1024
    chunk_size: int = 512
    chunk_overlap: int = 50
    vector_store: str = "chromadb"
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_table: str = "documents"
    chromadb_persist_dir: str = "./data/embeddings/chromadb"
    chromadb_collection: str = "knowledge_base"
    top_k: int = 5
    similarity_threshold: float = 0.7
    rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: list = field(default_factory=lambda: ["*"])
    api_key: str = ""
    rate_limit: int = 60
    max_upload_size: int = 52428800


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR} with environment variable values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var, "")
    return value


def _resolve_dict(d: dict) -> dict:
    """Recursively resolve environment variables in dict."""
    resolved = {}
    for k, v in d.items():
        if isinstance(v, dict):
            resolved[k] = _resolve_dict(v)
        elif isinstance(v, str):
            resolved[k] = _resolve_env_vars(v)
        else:
            resolved[k] = v
    return resolved


def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """Load and validate configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found: {path}, using defaults")
        return AppConfig()

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    raw = _resolve_dict(raw)
    cfg = AppConfig()

    # Model config
    if "model" in raw:
        m = raw["model"]
        cfg.model = ModelConfig(
            base_model=m.get("base_model", cfg.model.base_model),
            quantization_enabled=m.get("quantization", {}).get("enabled", True),
            quantization_bits=m.get("quantization", {}).get("bits", 4),
            quant_type=m.get("quantization", {}).get("quant_type", "nf4"),
            compute_dtype=m.get("quantization", {}).get("compute_dtype", "bfloat16"),
            double_quant=m.get("quantization", {}).get("double_quant", True),
            output_dir=m.get("output_dir", cfg.model.output_dir),
            adapter_dir=m.get("adapter_dir", cfg.model.adapter_dir),
            merged_dir=m.get("merged_dir", cfg.model.merged_dir),
            max_length=m.get("max_length", cfg.model.max_length),
            max_new_tokens=m.get("max_new_tokens", cfg.model.max_new_tokens),
        )

    # Audio config
    if "audio" in raw:
        a = raw["audio"]
        cfg.audio = AudioConfig(
            model=a.get("model", cfg.audio.model),
            language=a.get("language", cfg.audio.language),
            device=a.get("device", cfg.audio.device),
            compute_type=a.get("compute_type", cfg.audio.compute_type),
        )

    # LoRA config
    if "lora" in raw:
        lo = raw["lora"]
        cfg.lora = LoRAConfig(
            r=lo.get("r", cfg.lora.r),
            alpha=lo.get("alpha", cfg.lora.alpha),
            dropout=lo.get("dropout", cfg.lora.dropout),
            bias=lo.get("bias", cfg.lora.bias),
            target_modules=lo.get("target_modules", cfg.lora.target_modules),
            task_type=lo.get("task_type", cfg.lora.task_type),
        )

    # Training config
    if "training" in raw:
        t = raw["training"]
        cfg.training = TrainingConfig(
            epochs=t.get("epochs", cfg.training.epochs),
            batch_size=t.get("batch_size", cfg.training.batch_size),
            gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
            learning_rate=t.get("learning_rate", cfg.training.learning_rate),
            weight_decay=t.get("weight_decay", cfg.training.weight_decay),
            warmup_ratio=t.get("warmup_ratio", cfg.training.warmup_ratio),
            lr_scheduler=t.get("lr_scheduler", cfg.training.lr_scheduler),
            fp16=t.get("fp16", cfg.training.fp16),
            bf16=t.get("bf16", cfg.training.bf16),
            gradient_checkpointing=t.get("gradient_checkpointing", True),
            max_grad_norm=t.get("max_grad_norm", cfg.training.max_grad_norm),
            logging_steps=t.get("logging_steps", cfg.training.logging_steps),
            save_steps=t.get("save_steps", cfg.training.save_steps),
            eval_steps=t.get("eval_steps", cfg.training.eval_steps),
            save_total_limit=t.get("save_total_limit", cfg.training.save_total_limit),
            dataset_path=t.get("dataset_path", cfg.training.dataset_path),
            train_split=t.get("train_split", cfg.training.train_split),
            seed=t.get("seed", cfg.training.seed),
        )

    # RAG config
    if "rag" in raw:
        r = raw["rag"]
        sub = r.get("supabase", {})
        chroma = r.get("chromadb", {})
        ret = r.get("retrieval", {})
        cfg.rag = RAGConfig(
            embedding_model=r.get("embedding_model", cfg.rag.embedding_model),
            embedding_dim=r.get("embedding_dim", cfg.rag.embedding_dim),
            chunk_size=r.get("chunk_size", cfg.rag.chunk_size),
            chunk_overlap=r.get("chunk_overlap", cfg.rag.chunk_overlap),
            vector_store=r.get("vector_store", cfg.rag.vector_store),
            supabase_url=sub.get("url", ""),
            supabase_key=sub.get("key", ""),
            supabase_table=sub.get("table_name", "documents"),
            chromadb_persist_dir=chroma.get("persist_dir", cfg.rag.chromadb_persist_dir),
            chromadb_collection=chroma.get("collection_name", cfg.rag.chromadb_collection),
            top_k=ret.get("top_k", cfg.rag.top_k),
            similarity_threshold=ret.get("similarity_threshold", 0.7),
            rerank=ret.get("rerank", True),
            rerank_model=ret.get("rerank_model", cfg.rag.rerank_model),
        )

    # Server config
    if "server" in raw:
        s = raw["server"]
        cfg.server = ServerConfig(
            host=s.get("host", cfg.server.host),
            port=s.get("port", cfg.server.port),
            workers=s.get("workers", cfg.server.workers),
            cors_origins=s.get("cors_origins", cfg.server.cors_origins),
            api_key=s.get("api_key", ""),
            rate_limit=s.get("rate_limit", cfg.server.rate_limit),
            max_upload_size=s.get("max_upload_size", cfg.server.max_upload_size),
        )

    # GRPO config
    if "grpo" in raw:
        g = raw["grpo"]
        cfg.grpo = GRPOConfig(
            enabled=g.get("enabled", cfg.grpo.enabled),
            num_generations=g.get("num_generations", cfg.grpo.num_generations),
            max_completion_length=g.get("max_completion_length", cfg.grpo.max_completion_length),
            max_prompt_length=g.get("max_prompt_length", cfg.grpo.max_prompt_length),
            learning_rate=g.get("learning_rate", cfg.grpo.learning_rate),
            temperature=g.get("temperature", cfg.grpo.temperature),
            warmup_ratio=g.get("warmup_ratio", cfg.grpo.warmup_ratio),
            beta=g.get("beta", cfg.grpo.beta),
            epsilon=g.get("epsilon", cfg.grpo.epsilon),
            loss_type=g.get("loss_type", cfg.grpo.loss_type),
            scale_rewards=g.get("scale_rewards", cfg.grpo.scale_rewards),
            mask_truncated_completions=g.get("mask_truncated_completions", cfg.grpo.mask_truncated_completions),
            reasoning_format=g.get("reasoning_format", cfg.grpo.reasoning_format),
            reward_functions=g.get("reward_functions", cfg.grpo.reward_functions),
        )

    logger.info(f"Configuration loaded: model={cfg.model.base_model}")
    return cfg
