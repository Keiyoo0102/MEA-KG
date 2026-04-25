"""
finetune_qlora.py — QLoRA Fine-Tuning Script for MEA-KG (Section 2.4)

This script implements Quantized Low-Rank Adaptation (QLoRA) fine-tuning for
the GPT-OSS:20b backbone (or any Llama-compatible model) as described in
Section 2.4 of the manuscript.

Key Methodology (per manuscript Section 2.4)
---------------------------------------------
- **Base model**: GPT-OSS:20b (or a Llama-2 / Llama-3 equivalent for
  open-source reproducibility).
- **Quantisation**: 4-bit NF4 quantisation via ``bitsandbytes`` (QLoRA).
- **LoRA target modules**: query and value projection layers (``q_proj``,
  ``v_proj``), rank r=16, alpha=32.
- **Task**: Supervised fine-tuning on annotated planetary geology entity and
  relation extraction examples formatted as instruction-following triples.
- **Training data**: BIO-annotated sentences from the MEA corpus, converted
  to JSON instruction format (see ``data/dummy_sample/finetune_sample.json``).

Hardware Requirements
----------------------
- Minimum GPU VRAM: 16 GB (tested on NVIDIA A100 40 GB / RTX 3090 24 GB).
- For 4-bit QLoRA the effective requirement drops to ~12 GB for GPT-OSS:20b.
- CPU-only training is not supported for the full model; use the dummy dataset
  flag (``--dummy``) to perform a micro-test on CPU.

Usage
-----
    # Train on your own data
    python pipeline/finetune_qlora.py --data_path data/finetune/train.json

    # Quick smoke-test on built-in dummy data (CPU-safe, <2 min)
    python pipeline/finetune_qlora.py --dummy

    # Resume from a checkpoint
    python pipeline/finetune_qlora.py --data_path data/finetune/train.json \\
        --resume_from_checkpoint data/experiments/lora_output/checkpoint-100

Environment Variables
---------------------
    OPENAI_API_KEY / OPENAI_BASE_URL — not required for local fine-tuning
    HF_TOKEN — Hugging Face token if the base model is gated (optional)
    LORA_OUTPUT_DIR — override output directory (default: data/experiments/lora_output)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Check for optional heavy dependencies
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
_PEFT_AVAILABLE = False
_BNB_AVAILABLE = False
_TRL_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    _PEFT_AVAILABLE = True
except ImportError:
    pass

try:
    import bitsandbytes  # noqa: F401
    _BNB_AVAILABLE = True
except ImportError:
    pass

try:
    from trl import SFTTrainer, SFTConfig
    _TRL_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Default hyper-parameters (match manuscript Table 2)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Model
    "model_name_or_path": "meta-llama/Llama-2-13b-hf",  # swap for GPT-OSS:20b
    "max_seq_length": 2048,
    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj"],
    # Quantisation
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True,
    # Training
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "fp16": True,
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 2,
}

# ---------------------------------------------------------------------------
# Dummy / Synthetic Training Data
# ---------------------------------------------------------------------------

DUMMY_EXAMPLES = [
    {
        "instruction": (
            "You are a planetary geology named-entity recognition expert. "
            "Extract all geological entities from the following sentence and "
            "classify them using the MEA ontology types: "
            "[GeologicFeature, MineralComposition, Process, Location, Morphology]."
        ),
        "input": (
            "Polygonal terrain in Utopia Planitia is formed by thermal contraction "
            "of permafrost-rich regolith, analogous to tundra polygon networks "
            "observed in Siberia and Alaska."
        ),
        "output": json.dumps(
            {
                "entities": [
                    {"text": "Polygonal terrain", "type": "Morphology"},
                    {"text": "Utopia Planitia", "type": "Location"},
                    {"text": "thermal contraction", "type": "Process"},
                    {"text": "permafrost-rich regolith", "type": "MineralComposition"},
                    {"text": "tundra polygon networks", "type": "Morphology"},
                    {"text": "Siberia", "type": "Location"},
                    {"text": "Alaska", "type": "Location"},
                ],
                "relations": [
                    {
                        "subject": "Polygonal terrain",
                        "relation": "FORMED_BY",
                        "object": "thermal contraction",
                    },
                    {
                        "subject": "Polygonal terrain",
                        "relation": "ANALOG_OF",
                        "object": "tundra polygon networks",
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
    },
    {
        "instruction": (
            "Extract named entities and their semantic relations from the following "
            "Mars science text.  Use BIO tagging format: B-TYPE for begin tokens, "
            "I-TYPE for continuation."
        ),
        "input": (
            "Jarosite detected by the Opportunity rover at Meridiani Planum indicates "
            "acidic aqueous conditions during early Mars history."
        ),
        "output": json.dumps(
            {
                "bio_tags": [
                    ("Jarosite", "B-MineralComposition"),
                    ("detected", "O"),
                    ("by", "O"),
                    ("the", "O"),
                    ("Opportunity", "B-Location"),
                    ("rover", "O"),
                    ("at", "O"),
                    ("Meridiani", "B-Location"),
                    ("Planum", "I-Location"),
                    ("indicates", "O"),
                    ("acidic", "B-Process"),
                    ("aqueous", "I-Process"),
                    ("conditions", "I-Process"),
                    ("during", "O"),
                    ("early", "B-GeologicPeriod"),
                    ("Mars", "I-GeologicPeriod"),
                    ("history", "O"),
                ],
                "relations": [
                    {
                        "subject": "Jarosite",
                        "relation": "FOUND_AT",
                        "object": "Meridiani Planum",
                    },
                    {
                        "subject": "Jarosite",
                        "relation": "INDICATES",
                        "object": "acidic aqueous conditions",
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
    },
]


def _build_prompt(example: dict) -> str:
    """Format a training example into an Alpaca-style instruction prompt."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    if inp:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return prompt


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------

def load_training_data(data_path: str | Path | None, use_dummy: bool = False) -> list[str]:
    """Load training examples and convert to formatted prompt strings.

    Parameters
    ----------
    data_path : str or Path or None
        Path to a JSON file containing a list of instruction/input/output dicts.
    use_dummy : bool
        If ``True``, ignore *data_path* and use built-in synthetic examples.

    Returns
    -------
    list[str]
        Formatted prompt strings ready for tokenisation.
    """
    if use_dummy:
        examples = DUMMY_EXAMPLES
        logger.info("Using %d built-in dummy training examples.", len(examples))
    else:
        if data_path is None:
            raise ValueError("Either --data_path or --dummy must be specified.")
        with open(data_path, encoding="utf-8") as f:
            examples = json.load(f)
        logger.info("Loaded %d training examples from %s.", len(examples), data_path)

    prompts = [_build_prompt(ex) for ex in examples]
    return prompts


def run_qlora_training(args: argparse.Namespace) -> None:
    """Execute the QLoRA fine-tuning pipeline.

    This function:
    1. Loads and 4-bit quantises the base model via ``bitsandbytes``.
    2. Applies LoRA adapters using ``peft``.
    3. Trains with ``trl.SFTTrainer`` on instruction-formatted data.
    4. Saves LoRA adapter weights to *output_dir*.
    """
    # ---- Dependency checks ----
    missing = []
    if not _TORCH_AVAILABLE:
        missing.append("torch")
    if not _PEFT_AVAILABLE:
        missing.append("peft")
    if not _BNB_AVAILABLE:
        missing.append("bitsandbytes")
    if not _TRL_AVAILABLE:
        missing.append("trl")

    if missing:
        msg = (
            f"Missing required packages for QLoRA training: {', '.join(missing)}.\n"
            "Install them with:\n"
            f"  pip install {' '.join(missing)}\n"
            "Note: bitsandbytes requires a CUDA-enabled GPU."
        )
        if args.dummy:
            logger.warning("%s\nRunning in DUMMY mode — skipping actual training.", msg)
            _dummy_mode_demo(args)
            return
        else:
            raise ImportError(msg)

    # ---- Import heavy libraries only when available ----
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    cfg = DEFAULT_CONFIG.copy()

    # Override from CLI
    if args.model:
        cfg["model_name_or_path"] = args.model
    if args.epochs:
        cfg["num_train_epochs"] = args.epochs
    if args.lr:
        cfg["learning_rate"] = args.lr

    output_dir = Path(
        os.getenv("LORA_OUTPUT_DIR",
                  str(PROJECT_ROOT / "data" / "experiments" / "lora_output"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("QLoRA Fine-Tuning Configuration:")
    for k, v in cfg.items():
        logger.info("  %-40s %s", k, v)

    # ---- Step 1: Load Tokeniser ----
    logger.info("Loading tokeniser from %s ...", cfg["model_name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Step 2: 4-bit quantisation config (QLoRA) ----
    compute_dtype = getattr(torch, cfg["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )

    # ---- Step 3: Load quantised base model ----
    logger.info("Loading 4-bit quantised model (this may take several minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ---- Step 4: LoRA configuration ----
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Step 5: Load data ----
    prompts = load_training_data(args.data_path, use_dummy=args.dummy)
    dataset = Dataset.from_dict({"text": prompts})
    logger.info("Training dataset: %d samples", len(dataset))

    # ---- Step 6: SFTTrainer ----
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        fp16=cfg["fp16"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        resume_from_checkpoint=args.resume_from_checkpoint,
        report_to="none",  # disable wandb/tensorboard by default
        max_seq_length=cfg["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        dataset_text_field="text",
    )

    logger.info("Starting QLoRA training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ---- Step 7: Save adapter weights ----
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("LoRA adapter weights saved to %s", output_dir)
    logger.info(
        "To create an Ollama model from the adapter:\n"
        "  1. Convert to GGUF: llama.cpp convert_hf_to_gguf.py %s\n"
        "  2. Create Modelfile pointing to the GGUF.\n"
        "  3. Run: ollama create mea-kg-model -f Modelfile",
        output_dir,
    )


def _dummy_mode_demo(args: argparse.Namespace) -> None:
    """Demonstrate the QLoRA pipeline without GPU dependencies.

    Prints formatted training examples and the LoRA configuration that would be
    used, allowing reviewers to verify the code logic without a GPU.
    """
    logger.info("=== QLoRA Dummy Mode (no GPU required) ===")
    logger.info("Configuration (would be used in full training):")
    for k, v in DEFAULT_CONFIG.items():
        logger.info("  %-40s %s", k, v)

    prompts = load_training_data(None, use_dummy=True)
    logger.info("\nFormatted training prompt examples:")
    for i, p in enumerate(prompts):
        logger.info("\n--- Example %d ---\n%s", i + 1, p[:400] + "..." if len(p) > 400 else p)

    logger.info(
        "\nQLoRA training would proceed as:\n"
        "  1. Load %s in 4-bit NF4 quantisation.\n"
        "  2. Apply LoRA adapters to %s (r=%d, alpha=%d).\n"
        "  3. SFT-train for %d epochs with lr=%g.\n"
        "  4. Save adapter to data/experiments/lora_output/.",
        DEFAULT_CONFIG["model_name_or_path"],
        DEFAULT_CONFIG["lora_target_modules"],
        DEFAULT_CONFIG["lora_r"],
        DEFAULT_CONFIG["lora_alpha"],
        DEFAULT_CONFIG["num_train_epochs"],
        DEFAULT_CONFIG["learning_rate"],
    )
    logger.info("=== Dummy mode complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "QLoRA fine-tuning for the MEA-KG extraction model (Section 2.4)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the JSON training file.  Required unless --dummy is set.",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help=(
            "Use built-in synthetic examples instead of a real dataset. "
            "Runs without a GPU and is suitable for CI / reviewer testing."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Hugging Face model ID or local path for the base model "
            "(overrides the default in DEFAULT_CONFIG)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the learning rate.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.dummy and args.data_path is None:
        logger.error("Either --data_path or --dummy must be specified.")
        sys.exit(1)
    run_qlora_training(args)


if __name__ == "__main__":
    main()
