"""
Multimodal Data Preparation Pipeline.
خط أنابيب تحضير البيانات متعددة الوسائط
"""
import json
import base64
from pathlib import Path
from typing import Optional
from loguru import logger
from datasets import Dataset, DatasetDict


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encode image file to base64 string."""
    path = Path(image_path)
    if not path.exists():
        logger.warning(f"Image not found: {image_path}")
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_multimodal_conversation(
    text_query: str,
    response: str,
    image_path: Optional[str] = None,
    audio_transcript: Optional[str] = None,
    system_prompt: str = "أنت مساعد ذكي متعدد الوسائط. تفهم النصوص والصور والأصوات وتقدم إجابات دقيقة ومفيدة.",
) -> dict:
    """
    Build a single multimodal conversation sample.

    Returns dict in chat format compatible with most multimodal models.
    """
    user_content = []

    # System message
    messages = [{"role": "system", "content": system_prompt}]

    # Build user message with multimodal content
    if image_path:
        img_b64 = encode_image_to_base64(image_path)
        if img_b64:
            user_content.append({
                "type": "image",
                "image": img_b64,
            })

    if audio_transcript:
        # Audio is pre-transcribed via Whisper
        text_query = f"[نص صوتي]: {audio_transcript}\n\n{text_query}"

    user_content.append({"type": "text", "text": text_query})

    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": response})

    return {"messages": messages}


def load_jsonl_dataset(file_path: str) -> list[dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} samples from {file_path}")
    return data


def create_sample_dataset(output_dir: str) -> None:
    """
    Create a sample dataset demonstrating all supported formats.
    أنشئ مجموعة بيانات نموذجية توضّح جميع التنسيقات المدعومة

    Three types of samples:
    1. Text-only: standard chat messages
    2. Image + Text: image as base64 or file path reference
    3. Audio transcript + Text: pre-transcribed audio content
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create a small test image (solid color) for validation
    _create_test_image(output_path / "test_image.png")

    samples = [
        # ── Type 1: Text-only ──
        {
            "messages": [
                {"role": "system", "content": "أنت مساعد ذكي متعدد الوسائط."},
                {"role": "user", "content": "ما هي فوائد التمارين الرياضية؟"},
                {"role": "assistant", "content": "التمارين الرياضية لها فوائد عديدة تشمل تحسين صحة القلب والأوعية الدموية، وتقوية العضلات والعظام، وتحسين المزاج من خلال إفراز الإندورفين، وتعزيز جودة النوم، والمساعدة في الحفاظ على وزن صحي."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "أنت مساعد طبي متخصص."},
                {"role": "user", "content": "ما هي أعراض نقص فيتامين D؟"},
                {"role": "assistant", "content": "أعراض نقص فيتامين D تشمل: التعب والإرهاق المزمن، آلام العظام والمفاصل، ضعف العضلات، تأخر التئام الجروح، تساقط الشعر، الاكتئاب وتقلب المزاج، وضعف المناعة. يُنصح بفحص مستوى فيتامين D بانتظام خصوصاً في المناطق ذات التعرض المحدود للشمس."}
            ]
        },
        # ── Type 2: Image + Text (file path reference) ──
        {
            "messages": [
                {"role": "system", "content": "أنت طبيب ذكي يحلل الصور الطبية ويقدم تحليلاً مبدئياً."},
                {"role": "user", "content": [
                    {"type": "image", "image_path": "test_image.png"},
                    {"type": "text", "text": "صف ما تراه في هذه الصورة."}
                ]},
                {"role": "assistant", "content": "أرى صورة اختبار بسيطة. في الاستخدام الفعلي، سأقدم تحليلاً مفصلاً للصورة الطبية المقدمة، مع ملاحظة أي علامات غير طبيعية وتقديم تشخيص تفريقي مبدئي."}
            ]
        },
        # ── Type 3: Audio transcript + Text ──
        {
            "messages": [
                {"role": "system", "content": "أنت مساعد طبي يفهم النصوص والتسجيلات الصوتية."},
                {"role": "user", "content": "[نص صوتي]: المريض يشكو من ألم في الصدر منذ ثلاثة أيام مع ضيق تنفس عند المشي\n\nما التشخيص المبدئي المحتمل؟"},
                {"role": "assistant", "content": "بناءً على شكوى المريض بألم في الصدر لمدة ثلاثة أيام مع ضيق تنفس عند المجهود، يجب أولاً استبعاد الحالات الطارئة. التشخيص التفريقي يشمل: متلازمة الشريان التاجي الحادة، الانصمام الرئوي، التهاب التامور، أو أسباب غير قلبية مثل الارتجاع المعدي المريئي أو الألم العضلي الهيكلي. يُنصح بإجراء تخطيط قلب كهربائي وفحوصات إنزيمات القلب وتصوير الصدر بالأشعة السينية."}
            ]
        },
        # ── Type 4: Complex multimodal (image + detailed text) ──
        {
            "messages": [
                {"role": "system", "content": "أنت أخصائي تحليل بيانات بصرية. تحلل الصور والرسوم البيانية بدقة."},
                {"role": "user", "content": [
                    {"type": "image", "image_path": "test_image.png"},
                    {"type": "text", "text": "هل هذه الصورة تظهر أي نمط غير طبيعي؟ قدم تحليلاً مفصلاً."}
                ]},
                {"role": "assistant", "content": "الصورة المقدمة هي صورة اختبار ذات لون موحد. لا تظهر أنماط غير طبيعية لأنها صورة تجريبية. في السيناريو الفعلي، سأحلل الأنماط البصرية والألوان والهياكل في الصورة وأقدم تقريراً مفصلاً عن أي ملاحظات."}
            ]
        },
    ]

    output_file = output_path / "train.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"Sample dataset created at: {output_file}")
    logger.info(f"  Total samples: {len(samples)}")
    logger.info(f"  Text-only: 2, Image+Text: 2, Audio+Text: 1")
    logger.info(f"  Test image: {output_path / 'test_image.png'}")
    logger.info("")
    logger.info("To add your own data:")
    logger.info("  1. Add JSONL lines to data/processed/train.jsonl")
    logger.info("  2. For images: use 'image_path' (relative to data/processed/) or 'image' (base64)")
    logger.info("  3. For audio: pre-transcribe with Whisper and include as text")
    logger.info("  4. Format: {\"messages\": [{\"role\":..., \"content\":...}, ...]}")


def _create_test_image(path: Path) -> None:
    """Create a simple test image for dataset validation."""
    try:
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(64, 128, 200))
        img.save(str(path))
        logger.info(f"Test image created: {path}")
    except ImportError:
        logger.warning("PIL not available — skipping test image creation")


def prepare_dataset(
    data_dir: str,
    train_split: float = 0.9,
    seed: int = 42,
) -> DatasetDict:
    """
    Load and split dataset for training.

    Args:
        data_dir: Directory containing .jsonl files
        train_split: Fraction for training (rest is validation)
        seed: Random seed for reproducibility

    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    data_path = Path(data_dir)
    all_samples = []

    # Load all JSONL files
    for jsonl_file in sorted(data_path.glob("*.jsonl")):
        samples = load_jsonl_dataset(str(jsonl_file))
        all_samples.extend(samples)

    if not all_samples:
        logger.error(f"No data found in {data_dir}. Run create_sample_dataset first.")
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    logger.info(f"Total samples loaded: {len(all_samples)}")

    # Convert to HuggingFace Dataset
    # Flatten messages to text for training
    processed = []
    for sample in all_samples:
        messages = sample.get("messages", [])
        processed.append({"messages": json.dumps(messages, ensure_ascii=False)})

    dataset = Dataset.from_list(processed)

    # Split
    split = dataset.train_test_split(
        test_size=1.0 - train_split,
        seed=seed,
    )

    dataset_dict = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    logger.info(f"Dataset split — Train: {len(dataset_dict['train'])}, Val: {len(dataset_dict['validation'])}")
    return dataset_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare multimodal dataset")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--create-sample", action="store_true", help="Create sample dataset")
    args = parser.parse_args()

    from src.utils.config import load_config
    cfg = load_config(args.config)

    if args.create_sample:
        create_sample_dataset(cfg.training.dataset_path)
    else:
        ds = prepare_dataset(cfg.training.dataset_path, cfg.training.train_split, cfg.training.seed)
        print(f"Dataset ready: {ds}")
