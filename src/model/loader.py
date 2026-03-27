"""
Model Loader — Load base model with QLoRA quantization and LoRA adapters.
محمّل النموذج — تحميل النموذج الأساسي مع التكميم وإضافة طبقات LoRA
"""
import torch
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
    TaskType,
)
from src.utils.config import AppConfig


def get_quantization_config(cfg: AppConfig) -> Optional[BitsAndBytesConfig]:
    """Build BitsAndBytes quantization config."""
    if not cfg.model.quantization_enabled:
        return None

    compute_dtype = getattr(torch, cfg.model.compute_dtype, torch.bfloat16)

    if cfg.model.quantization_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.model.quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg.model.double_quant,
            llm_int8_enable_fp32_cpu_offload=True,  # CRITICAL: Allows offloading to RAM when VRAM is full
        )
    elif cfg.model.quantization_bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        logger.warning(f"Unsupported quantization bits: {cfg.model.quantization_bits}")
        return None


def get_lora_config(cfg: AppConfig) -> LoraConfig:
    """
    Build LoRA configuration.
    Updated: includes modules_to_save for lm_head + embed_tokens
    when new special tokens are added (reasoning tags, etc.)
    """
    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
    }
    # Avoid applying PEFT to the Vision Encoder (which already has its own LoRA inside the model).
    # This prevents the 'only Tensors of floating point dtype can require gradients' crash.
    targets = "|".join(cfg.lora.target_modules)
    target_regex = fr".*model\.layers.*(?:{targets})"

    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        target_modules=target_regex,
        task_type=task_type_map.get(cfg.lora.task_type, TaskType.CAUSAL_LM),
        # Save embedding layers when fine-tuning with new special tokens
        modules_to_save=["lm_head", "embed_tokens"],
    )


def load_base_model(
    cfg: AppConfig,
    for_training: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Optional[AutoProcessor]]:
    """
    Load the base multimodal model with optional quantization.

    Args:
        cfg: Application configuration
        for_training: If True, prepare model for k-bit training

    Returns:
        Tuple of (model, tokenizer, processor)
    """
    logger.info(f"Loading base model: {cfg.model.base_model}")

    quant_config = get_quantization_config(cfg)
    compute_dtype = getattr(torch, cfg.model.compute_dtype, torch.bfloat16)

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(cfg.model.base_model, trust_remote_code=True)

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": compute_dtype,
    }

    # Flash Attention 2 — 2x faster, less VRAM (requires Ampere+ GPU)
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        config._attn_implementation = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")
    except ImportError:
        model_kwargs["attn_implementation"] = "eager"
        config._attn_implementation = "eager"
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "eager"
        logger.info("Flash Attention 2 not available — using default attention")

    if quant_config:
        model_kwargs["quantization_config"] = quant_config
        logger.info(f"Quantization enabled: {cfg.model.quantization_bits}-bit {cfg.model.quant_type}")

    # Fix Microsoft Phi-4 PEFT initialization bug: Phi4MMModel lacks prepare_inputs_for_generation
    # which causes PeftModel to crash when the model internally instantiates LoRA for its vision encoder.
    added_dummy_prep = False
    if not hasattr(torch.nn.Module, "prepare_inputs_for_generation"):
        setattr(torch.nn.Module, "prepare_inputs_for_generation", lambda self, *args, **kwargs: None)
        added_dummy_prep = True

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        config=config,
        **model_kwargs,
    )

    if added_dummy_prep:
        delattr(torch.nn.Module, "prepare_inputs_for_generation")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load processor (for multimodal inputs — images, audio)
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            cfg.model.base_model,
            trust_remote_code=True,
        )
        logger.info("Multimodal processor loaded successfully")
    except Exception as e:
        logger.warning(f"No multimodal processor found: {e}")

    # Prepare for training
    if for_training:
        if quant_config:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=cfg.training.gradient_checkpointing,
            )
        elif cfg.training.gradient_checkpointing:
            model.gradient_checkpointing_enable()

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: {param_count / 1e9:.2f}B params, {trainable / 1e6:.2f}M trainable")

    return model, tokenizer, processor


def load_model_with_lora(
    cfg: AppConfig,
    for_training: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Optional[AutoProcessor]]:
    """
    Load base model and apply LoRA adapters.

    Args:
        cfg: Application configuration
        for_training: If True, create new LoRA. If False, load existing adapter.

    Returns:
        Tuple of (peft_model, tokenizer, processor)
    """
    model, tokenizer, processor = load_base_model(cfg, for_training=for_training)

    if for_training:
        # Apply new LoRA adapters
        lora_config = get_lora_config(cfg)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("New LoRA adapters applied for training")
    else:
        # Load existing trained adapter
        adapter_path = Path(cfg.model.adapter_dir)
        if adapter_path.exists():
            model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                is_trainable=False,
            )
            logger.info(f"LoRA adapter loaded from: {adapter_path}")
        else:
            logger.warning(f"No adapter found at {adapter_path}, using base model")

    return model, tokenizer, processor


def load_inference_model(
    cfg: AppConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Optional[AutoProcessor]]:
    """
    Load the final production model for inference.
    Tries merged model first, then adapter, then base.
    """
    merged_path = Path(cfg.model.merged_dir)
    adapter_path = Path(cfg.model.adapter_dir)

    if merged_path.exists() and any(merged_path.iterdir()):
        logger.info(f"Loading merged model from: {merged_path}")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }
        # For inference, optionally quantize for speed
        if cfg.model.quantization_enabled:
            model_kwargs["quantization_config"] = get_quantization_config(cfg)

        model = AutoModelForCausalLM.from_pretrained(str(merged_path), **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(str(merged_path), trust_remote_code=True)
        processor = None
        try:
            processor = AutoProcessor.from_pretrained(str(merged_path), trust_remote_code=True)
        except Exception:
            pass
        return model, tokenizer, processor

    elif adapter_path.exists() and any(adapter_path.iterdir()):
        logger.info("Loading base model + LoRA adapter for inference")
        return load_model_with_lora(cfg, for_training=False)

    else:
        logger.info("No fine-tuned model found, loading base model")
        return load_base_model(cfg, for_training=False)


def merge_and_save(cfg: AppConfig) -> None:
    """Merge LoRA adapter into base model and save for deployment."""
    logger.info("Merging LoRA adapter into base model...")
    model, tokenizer, processor = load_model_with_lora(cfg, for_training=False)

    merged_model = model.merge_and_unload()
    output_path = Path(cfg.model.merged_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    if processor:
        processor.save_pretrained(str(output_path))

    logger.info(f"Merged model saved to: {output_path}")
