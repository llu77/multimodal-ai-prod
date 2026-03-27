"""
Training Pipeline — Multimodal QLoRA Fine-tuning + GRPO Reinforcement Learning.
خط أنابيب التدريب — الضبط الدقيق متعدد الوسائط بـ QLoRA + التعلم بالتعزيز GRPO

FIX (TODO-1): Complete rewrite of SFT training.
Old version: extracted text only, discarded images → text-only training
New version: uses MultimodalDataset + MultimodalDataCollator → real multimodal training

Architecture:
    JSONL (text + base64 images)
        → MultimodalDataset (loads images, tokenizes with processor)
        → MultimodalDataCollator (pads batches, handles mixed text/image)
        → Trainer (computes loss on assistant tokens only)
"""
import json
from pathlib import Path
from loguru import logger

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from src.model.loader import load_model_with_lora
from src.data.multimodal_dataset import (
    MultimodalDataset,
    MultimodalDataCollator,
    validate_dataset,
)
from src.utils.config import AppConfig, load_config


def run_sft_training(cfg: AppConfig) -> str:
    """
    Run Supervised Fine-Tuning (SFT) with QLoRA — MULTIMODAL.
    تشغيل الضبط الدقيق المُشرف عليه مع QLoRA — متعدد الوسائط

    FIX: Old version used SFTTrainer with text-only formatting.
    New version uses standard Trainer with:
    - MultimodalDataset: loads images + text, processes through model's processor
    - MultimodalDataCollator: pads variable-length multimodal batches
    - Label masking: loss only on assistant response tokens

    Returns: Path to saved adapter
    """
    logger.info("=" * 60)
    logger.info("Starting Multimodal SFT Training / بدء التدريب متعدد الوسائط")
    logger.info("=" * 60)

    # ── Load model with LoRA ──
    model, tokenizer, processor = load_model_with_lora(cfg, for_training=True)

    if processor is None:
        logger.warning(
            "No multimodal processor found for this model. "
            "Image training will fall back to text-only. "
            "This is expected for text-only models like Mistral/Llama."
        )

    # ── Load dataset ──
    dataset_path = Path(cfg.training.dataset_path)
    train_file = dataset_path / "train.jsonl"

    if not train_file.exists():
        logger.error(f"Training data not found: {train_file}")
        logger.info("Run: python -m src.data.prepare_dataset --create-sample --config config/config.yaml")
        raise FileNotFoundError(f"No training data at {train_file}")

    # Build multimodal dataset
    full_dataset = MultimodalDataset(
        data_path=str(train_file),
        tokenizer=tokenizer,
        processor=processor,
        max_length=cfg.model.max_length,
        data_root=str(dataset_path),
    )

    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty. Add samples to train.jsonl first.")

    # Validate a few samples
    logger.info("Validating dataset samples...")
    stats = validate_dataset(full_dataset, num_samples=min(3, len(full_dataset)))
    if stats["errors"]:
        logger.warning(f"Dataset has {len(stats['errors'])} errors — check data quality")

    # ── Split train/val ──
    split_idx = int(len(full_dataset) * cfg.training.train_split)
    # Use torch random_split for proper Dataset splitting
    train_size = split_idx
    val_size = len(full_dataset) - split_idx

    if val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.training.seed),
        )
    else:
        train_dataset = full_dataset
        val_dataset = None

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset) if val_dataset else 0} samples")

    # ── Data Collator ──
    collator = MultimodalDataCollator(
        tokenizer=tokenizer,
        max_length=cfg.model.max_length,
    )

    # ── Output directory ──
    output_dir = Path(cfg.model.adapter_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training arguments ──
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        max_grad_norm=cfg.training.max_grad_norm,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=cfg.training.eval_steps if val_dataset else None,
        save_total_limit=cfg.training.save_total_limit,
        seed=cfg.training.seed,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,  # CRITICAL: keep pixel_values and other image columns
        dataloader_pin_memory=True,
    )

    # ── Initialize Trainer ──
    # Using standard Trainer (not SFTTrainer) because:
    # 1. SFTTrainer expects a "text" field — we have multimodal tensors
    # 2. We already handle tokenization in MultimodalDataset
    # 3. We already handle label masking in MultimodalDataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    # ── Train ──
    logger.info("Training started...")
    logger.info(f"  Model: {cfg.model.base_model}")
    logger.info(f"  LoRA rank: {cfg.lora.r}, alpha: {cfg.lora.alpha}")
    logger.info(f"  Batch: {cfg.training.batch_size} × {cfg.training.gradient_accumulation_steps} grad_accum")
    logger.info(f"  Epochs: {cfg.training.epochs}, LR: {cfg.training.learning_rate}")
    logger.info(f"  Multimodal: {'Yes (processor loaded)' if processor else 'No (text-only)'}")

    try:
        train_result = trainer.train()
    except torch.cuda.OutOfMemoryError:
        logger.error(
            f"GPU OOM during training. Try:\n"
            f"  1. Reduce batch_size in config (current: {cfg.training.batch_size})\n"
            f"  2. Reduce max_length (current: {cfg.model.max_length})\n"
            f"  3. Increase gradient_accumulation_steps\n"
            f"  4. Use a smaller model"
        )
        raise

    # ── Save adapter ──
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if processor:
        try:
            processor.save_pretrained(str(output_dir))
        except Exception:
            pass  # Not all processors support save_pretrained

    # ── Log metrics ──
    metrics = train_result.metrics
    logger.info("Training complete!")
    logger.info(f"  Loss: {metrics.get('train_loss', 'N/A')}")
    logger.info(f"  Runtime: {metrics.get('train_runtime', 0):.0f}s")
    logger.info(f"  Samples/sec: {metrics.get('train_samples_per_second', 0):.2f}")
    logger.info(f"  Adapter saved to: {output_dir}")

    return str(output_dir)


def run_grpo_training(cfg: AppConfig) -> str:
    """
    Run GRPO (Group Relative Policy Optimization) for reasoning improvement.
    تشغيل GRPO لتحسين الاستدلال

    FIX: Complete rewrite of reward functions.
    Old rewards were heuristic (length, keywords) — easily gamed.
    New rewards are VERIFIABLE following DeepSeek R1's RLVR pattern:
    - Accuracy: compare against ground-truth answers
    - Format: structured output validation
    - Coherence: language consistency check
    """
    if not cfg.grpo.enabled:
        logger.info("GRPO training is disabled in config")
        return ""

    logger.info("=" * 60)
    logger.info("Starting GRPO Training / بدء تدريب GRPO")
    logger.info("=" * 60)

    from trl import GRPOTrainer, GRPOConfig as TRLGRPOConfig

    # Load the SFT-trained model
    model, tokenizer, processor = load_model_with_lora(cfg, for_training=False)

    # GRPOTrainer requires left padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Import extracted reward functions (testable independently)
    from src.training.rewards import (
        accuracy_reward, format_reward, coherence_reward, xml_format_reward,
    )

    # ── Reasoning System Prompt (Nemotron-style) ──
    REASONING_SYSTEM_PROMPTS = {
        "xml": (
            "أنت مساعد طبي ذكي. أجب باللغة العربية بالتنسيق التالي:\n"
            "<reasoning>\nتحليلك خطوة بخطوة\n</reasoning>\n"
            "<answer>\nإجابتك النهائية\n</answer>"
        ),
        "think": (
            "أنت مساعد طبي ذكي. فكّر خطوة بخطوة قبل الإجابة.\n"
            "استخدم <think>...</think> للتفكير ثم أعط إجابتك."
        ),
        "none": "أنت مساعد طبي ذكي. أجب باللغة العربية بدقة.",
    }

    reasoning_fmt = cfg.grpo.reasoning_format
    system_prompt = REASONING_SYSTEM_PROMPTS.get(reasoning_fmt, REASONING_SYSTEM_PROMPTS["none"])

    # Load training prompts — MUST include 'answer' for verifiable rewards
    dataset_path = Path(cfg.training.dataset_path)
    prompts_file = dataset_path / "grpo_prompts.jsonl"

    if not prompts_file.exists():
        logger.warning(f"GRPO prompts not found: {prompts_file}")
        logger.info("Creating sample GRPO prompts with verifiable answers...")
        sample_prompts = [
            {"prompt": "ما هو العضو المسؤول عن ضخ الدم في جسم الإنسان؟", "answer": "القلب"},
            {"prompt": "كم عدد عظام الجسم البشري البالغ؟", "answer": "206"},
            {"prompt": "ما هو الهرمون المسؤول عن تنظيم مستوى السكر في الدم؟", "answer": "الأنسولين"},
            {"prompt": "ما الفرق الرئيسي بين الشريان والوريد؟", "answer": "الشريان ينقل الدم من القلب والوريد ينقل الدم إلى القلب"},
            {"prompt": "ما هو أكبر عضو في جسم الإنسان؟", "answer": "الجلد"},
            {"prompt": "ما هي وظيفة الكلى الرئيسية؟", "answer": "تصفية الدم وإنتاج البول"},
            {"prompt": "ما هو الناقل العصبي المسؤول عن السعادة؟", "answer": "السيروتونين"},
            {"prompt": "كم عدد فقرات العمود الفقري؟", "answer": "33"},
            {"prompt": "ما الغدة المسؤولة عن تنظيم الأيض؟", "answer": "الغدة الدرقية"},
            {"prompt": "ما أصغر عظمة في جسم الإنسان؟", "answer": "الركاب في الأذن الوسطى"},
        ]
        prompts_file.parent.mkdir(parents=True, exist_ok=True)
        with open(prompts_file, "w", encoding="utf-8") as f:
            for p in sample_prompts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    # Convert to conversational format for GRPOTrainer
    # GRPOTrainer expects "prompt" column with chat messages format
    formatted_prompts = []
    for p in prompts:
        formatted_prompts.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p["prompt"]},
            ],
            "answer": p["answer"],  # Kept for accuracy_reward
        })

    prompt_dataset = Dataset.from_list(formatted_prompts)
    logger.info(f"GRPO dataset: {len(prompt_dataset)} prompts, format={reasoning_fmt}")

    # Select reward functions based on reasoning format
    reward_fns = [accuracy_reward, format_reward, coherence_reward]
    if reasoning_fmt == "xml":
        reward_fns.append(xml_format_reward)
        logger.info("Added XML format reward (Nemotron-style reasoning)")

    # ── GRPO Config (Nemotron/DAPO 2025 best practices) ──
    grpo_output = str(Path(cfg.model.adapter_dir) / "grpo")

    grpo_kwargs = {
        "output_dir": grpo_output,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": cfg.grpo.learning_rate,
        "warmup_ratio": cfg.grpo.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "num_generations": cfg.grpo.num_generations,
        "max_completion_length": cfg.grpo.max_completion_length,
        "max_prompt_length": cfg.grpo.max_prompt_length,
        "temperature": cfg.grpo.temperature,
        # Nemotron/DAPO improvements:
        "beta": cfg.grpo.beta,                                       # 0.0 = no KL (modern standard)
        "loss_type": cfg.grpo.loss_type,                             # "dapo" recommended
        "mask_truncated_completions": cfg.grpo.mask_truncated_completions,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_grad_norm": 0.1,                                        # Tighter clipping for RL stability
        "logging_steps": 1,
        "save_steps": 50,
        "report_to": "none",
        "remove_unused_columns": False,                               # Keep 'answer' column for rewards
    }

    # scale_rewards: handle string/bool from config
    sr = cfg.grpo.scale_rewards
    if sr == "batch":
        grpo_kwargs["scale_rewards"] = "batch"
    elif sr in (True, "true", "True"):
        grpo_kwargs["scale_rewards"] = True
    else:
        grpo_kwargs["scale_rewards"] = False

    grpo_config = TRLGRPOConfig(**grpo_kwargs)

    # ── Initialize GRPO Trainer ──
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=prompt_dataset,
        reward_funcs=reward_fns,
    )

    logger.info("GRPO Training started (Nemotron/DAPO mode)...")
    logger.info(f"  loss_type={cfg.grpo.loss_type}, beta={cfg.grpo.beta}")
    logger.info(f"  scale_rewards={cfg.grpo.scale_rewards}, mask_truncated={cfg.grpo.mask_truncated_completions}")
    logger.info(f"  num_generations={cfg.grpo.num_generations}, temperature={cfg.grpo.temperature}")
    trainer.train()

    # Save
    trainer.save_model(grpo_output)
    tokenizer.save_pretrained(grpo_output)
    logger.info(f"GRPO adapter saved to: {grpo_output}")

    return grpo_output



def main():
    """CLI entry point for training."""
    import argparse
    parser = argparse.ArgumentParser(description="Training Pipeline — تدريب النموذج")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stage", choices=["sft", "grpo", "all"], default="sft")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.stage in ("sft", "all"):
        run_sft_training(cfg)

    if args.stage in ("grpo", "all"):
        run_grpo_training(cfg)


if __name__ == "__main__":
    main()