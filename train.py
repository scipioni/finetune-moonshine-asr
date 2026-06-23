"""
Main training script for Moonshine fine-tuning with curriculum learning.

Based on Moonshine paper (arXiv:2410.15608v2) recommendations:
- Training instances in [4, 30] seconds
- Bimodal distribution (naturally emerges)
- <0.5% of data should be <1s (avoids repetitions, >100% WER)

Examples:
    # Phase 1 with Common Voice French
    python train.py --config moonshine_ft/configs/common_voice_fr.yaml --phase 1

    # Phase 2 (resume from Phase 1)
    python train.py --config moonshine_ft/configs/common_voice_fr.yaml --phase 2 \\
                   --resume ./results-moonshine-fr-phase1

    # Phase 3 (resume from Phase 2)
    python train.py --config moonshine_ft/configs/common_voice_fr.yaml --phase 3 \\
                   --resume ./results-moonshine-fr-phase2

    # Train without curriculum learning
    python train.py --config moonshine_ft/configs/common_voice_fr.yaml --no-curriculum
"""

import moonshine_ft.compat  # noqa: F401 — must be first import
import argparse
import yaml
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import evaluate
from transformers import (
    AutoProcessor,
    MoonshineForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from moonshine_ft.data_loader import MoonshineDataLoader
from moonshine_ft.curriculum import CurriculumScheduler


def patch_layer_norms_for_rocm(model: nn.Module) -> None:
    """
    Replace nn.LayerNorm with an element-wise fallback on AMD ROCm.

    ROCm's NativeLayerNormBackward0 kernel produces NaN for certain activation
    patterns in Moonshine's encoder/decoder.  Decomposing layer norm into basic
    ops sidesteps the fused kernel and gives numerically stable gradients.
    Only applied when running on ROCm; NVIDIA/CPU paths keep the fast kernel.
    """
    if not (hasattr(torch.version, 'hip') and torch.version.hip is not None):
        return

    class _StableLN(nn.Module):
        def __init__(self, ln: nn.LayerNorm):
            super().__init__()
            self.register_parameter('weight', ln.weight)
            self.register_parameter('bias', ln.bias)
            self.eps = ln.eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(-1, keepdim=True)
            diff = x - mean
            var = (diff * diff).mean(-1, keepdim=True)
            x_hat = diff * torch.rsqrt(var + self.eps)
            if self.weight is not None:
                x_hat = x_hat * self.weight
            if self.bias is not None:
                x_hat = x_hat + self.bias
            return x_hat

    replaced = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.LayerNorm):
                setattr(parent, child_name, _StableLN(child))
                replaced += 1
    if replaced:
        print(f"[ROCm] Replaced {replaced} LayerNorm(s) with element-wise fallback")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune Moonshine ASR with Curriculum Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --config moonshine_ft/configs/common_voice_fr.yaml --phase 1
  python train.py --config moonshine_ft/configs/common_voice_fr.yaml --phase 2 --resume ./results-moonshine-fr-phase1
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--phase',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='Curriculum learning phase (1-3)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Checkpoint directory to resume from'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    parser.add_argument(
        '--no-curriculum',
        action='store_true',
        help='Disable curriculum learning (train on full data)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Override max training steps'
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push model to HuggingFace Hub after training'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode: use only 500 samples for quick validation'
    )

    return parser.parse_args()


# ============================================
# Data Collator for Moonshine
# ============================================
class DataCollatorMoonshineSeq2SeqWithPadding:
    """
    Data collator for Moonshine that handles:
    - Padding audio inputs
    - Padding labels with -100 (ignored in loss)
    - Creating decoder_input_ids (shifted right with BOS token)

    Based on original implementation but optimized for readability.
    """

    def __init__(self, processor, decoder_start_token_id, pad_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        # Extract input values and labels
        input_values = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        # Pad audio inputs
        batch = self.processor.feature_extractor.pad(
            input_values,
            return_tensors="pt",
            return_attention_mask=True
        )

        # Pad labels using PyTorch's efficient pad_sequence
        label_tensors = [torch.tensor(labels, dtype=torch.long) for labels in label_features]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            label_tensors,
            batch_first=True,
            padding_value=-100  # Ignored in loss calculation
        )

        # Create decoder_input_ids: [BOS, t1, t2, ..., tN]
        # Labels are: [t1, t2, ..., tN, EOS]
        # So decoder_input_ids = [BOS] + labels[:-1]
        decoder_input_tensors = [
            torch.tensor(
                [self.decoder_start_token_id] + labels[:-1],
                dtype=torch.long
            )
            for labels in label_features
        ]

        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            decoder_input_tensors,
            batch_first=True,
            padding_value=self.pad_token_id
        )

        batch["decoder_input_ids"] = decoder_input_ids
        batch["labels"] = labels_padded

        return batch


# ============================================
# Custom Trainer for Moonshine
# ============================================
class MoonshineSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom trainer that properly handles Moonshine's generate() method.

    Moonshine uses variable-length sequences, so we need to:
    1. Calculate max_new_tokens based on audio duration
    2. Use phase-specific generation parameters
    """

    def __init__(self, *args, generation_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_config = generation_config or {}

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step for Moonshine."""
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Calculate generation parameters based on audio length
        # Paper: Training instances in [4, 30] seconds
        if 'input_values' in inputs:
            audio_length = inputs['input_values'].shape[-1]
            audio_duration = audio_length / 16000  # 16kHz sampling rate
            # Roughly 6 tokens per second (conservative estimate)
            max_new_tokens = max(5, min(int(audio_duration * 6), 50))
        else:
            max_new_tokens = 50

        labels = inputs.get("labels", None)

        # If only computing loss, don't generate
        if prediction_loss_only:
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, "loss") else None
            return (loss, None, None)

        # Generate predictions
        with torch.no_grad():
            # Calculate loss first (if labels available)
            if has_labels:
                outputs = model(**inputs)
                loss = outputs.loss
            else:
                loss = None

            # Generate transcriptions with phase-specific parameters.
            # max_length=None suppresses the "both max_new_tokens and max_length set" warning
            # that fires when the model's default generation_config has max_length defined.
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "max_length": None,
                **self.generation_config
            }

            # Moonshine generate expects input_values (not input_features)
            generated_tokens = model.generate(
                input_values=inputs["input_values"],
                attention_mask=inputs.get("attention_mask", None),
                **generation_kwargs
            )

        if labels is not None:
            labels = labels.detach()

        return (loss, generated_tokens, labels)


def main():
    args = parse_args()

    # ============================================
    # Load Configuration
    # ============================================
    print("\n" + "="*80)
    print("MOONSHINE FINE-TUNING WITH CURRICULUM LEARNING")
    print("="*80)
    print(f"Based on Moonshine paper (arXiv:2410.15608v2)")
    print(f"  - Training instances: [4, 30] seconds")
    print(f"  - Bimodal distribution (Phase 3)")
    print(f"  - <0.5% data <1s (avoids repetitions)")
    print("="*80)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nConfiguration loaded from: {args.config}")

    # Override config with command-line arguments
    if args.no_curriculum:
        config['curriculum']['enabled'] = False
        print("[WARNING] Curriculum learning DISABLED")

    # ============================================
    # Setup Curriculum Learning
    # ============================================
    curriculum = CurriculumScheduler.from_config(config)

    # Print curriculum summary
    curriculum.print_summary()

    if config['curriculum']['enabled']:
        phase_idx = args.phase - 1  # Convert to 0-indexed
        phase = curriculum.get_phase(phase_idx)
        print(f"\n{'='*80}")
        print(f"PHASE {args.phase}: {phase.name}")
        print(f"{'='*80}")
        print(f"Description: {phase.description}")
        print(f"Duration range: [{phase.min_duration}s, {phase.max_duration}s]")
        print(f"Max words: {phase.max_words or 'unlimited'}")
        print(f"Max steps: {phase.max_steps:,}")
        print(f"Learning rate: {phase.learning_rate}")
        print(f"Target WER: <{phase.target_wer}%")
    else:
        phase = curriculum.get_phase(0)  # Use first phase config
        print(f"\nCurriculum learning DISABLED - training on full dataset")
        print(f"Duration range: [{phase.min_duration}s, {phase.max_duration}s]")

    # ============================================
    # Model Configuration
    # ============================================
    model_name = config['model']['name']
    model_path = model_name if args.resume is None else args.resume

    if args.resume:
        print(f"\n[OK] Resuming from checkpoint: {args.resume}")
    else:
        print(f"\n[OK] Starting from pre-trained model: {model_name}")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif config['curriculum']['enabled']:
        output_dir = f"{config['training']['output_dir']}-phase{args.phase}"
    else:
        output_dir = config['training']['output_dir']

    print(f"[OK] Output directory: {output_dir}")

    # ============================================
    # Load Processor
    # ============================================
    print("\n" + "="*60)
    print("LOADING PROCESSOR")
    print("="*60)
    processor = AutoProcessor.from_pretrained(model_name)

    # Register special tokens
    special_tokens = {
        'bos_token': '<s>',
        'eos_token': '</s>',
        'pad_token': '</s>',
    }
    processor.tokenizer.add_special_tokens(special_tokens)

    print(f"[OK] Tokenizer loaded")
    print(f"  BOS token ID: {processor.tokenizer.bos_token_id}")
    print(f"  EOS token ID: {processor.tokenizer.eos_token_id}")
    print(f"  PAD token ID: {processor.tokenizer.pad_token_id}")

    # ============================================
    # Load Dataset
    # ============================================
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)

    data_loader = MoonshineDataLoader.from_config(config)
    dataset_dict = data_loader.load_dataset()

    # Test mode: use only 500 samples (enough to get samples in various duration ranges)
    if args.test_mode:
        print("\n[WARNING] TEST MODE: Using only 500 samples per split")
        dataset_dict['train'] = dataset_dict['train'].select(range(min(500, len(dataset_dict['train']))))
        dataset_dict['test'] = dataset_dict['test'].select(range(min(500, len(dataset_dict['test']))))

    # Filter by global duration constraints (remove very short/long)
    print(f"\n{'='*60}")
    print("APPLYING GLOBAL DURATION FILTERS")
    print(f"{'='*60}")

    min_dur = config['audio'].get('min_duration', 1.0)
    max_dur = config['audio'].get('max_duration', 30.0)

    print(f"Global constraints: [{min_dur}s, {max_dur}s]")

    dataset_dict['train'] = data_loader.filter_by_duration(
        dataset_dict['train'],
        max_duration=max_dur,
        min_duration=min_dur
    )
    dataset_dict['test'] = data_loader.filter_by_duration(
        dataset_dict['test'],
        max_duration=max_dur,
        min_duration=min_dur
    )

    # Apply curriculum filtering if enabled
    if config['curriculum']['enabled']:
        print(f"\n{'='*60}")
        print(f"APPLYING CURRICULUM PHASE {args.phase} FILTER")
        print(f"{'='*60}")

        # Detect which duration column exists (different datasets use different names)
        duration_col = 'duration' if 'duration' in dataset_dict['train'].column_names else 'audio_duration'

        dataset_dict['train'] = curriculum.filter_dataset(
            dataset_dict['train'],
            phase,
            duration_column=duration_col,
            text_column='sentence'
        )
        dataset_dict['test'] = curriculum.filter_dataset(
            dataset_dict['test'],
            phase,
            duration_column=duration_col,
            text_column='sentence'
        )

    # Check if we have enough data
    if len(dataset_dict['train']) < 10:
        print(f"\n[WARNING] WARNING: Only {len(dataset_dict['train'])} training samples!")
        print("This is too few for meaningful training. Check your duration filters.")
        if not args.test_mode:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        else:
            print("[WARNING] Test mode: Skipping confirmation prompt. Using larger sample in test mode.")
            # In test mode with insufficient data, use more samples
            if len(dataset_dict['train']) == 0:
                print("[WARNING] No training samples found. Exiting.")
                return

    # Prepare datasets (feature extraction + tokenization)
    print(f"\n{'='*60}")
    print("PREPROCESSING DATASETS")
    print(f"{'='*60}")

    dataset_dict['train'] = data_loader.prepare_dataset(
        dataset_dict['train'],
        processor
    )
    dataset_dict['test'] = data_loader.prepare_dataset(
        dataset_dict['test'],
        processor
    )

    # Save encoded datasets
    encoded_path = f'{output_dir}_encoded'
    os.makedirs(encoded_path, exist_ok=True)
    dataset_dict.save_to_disk(encoded_path)
    print(f"\n[OK] Saved encoded datasets to: {encoded_path}")

    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(dataset_dict['train']):,} samples")
    print(f"  Test:  {len(dataset_dict['test']):,} samples")

    # ============================================
    # Load Model
    # ============================================
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    print(f"Loading from: {model_path}")

    model = MoonshineForConditionalGeneration.from_pretrained(model_path)

    # Configure model tokens
    model.config.pad_token_id = 2
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.config.decoder_start_token_id = 1
    model.config.use_cache = False  # Required for gradient checkpointing

    # Configure generation (phase-specific)
    if hasattr(model, 'generation_config'):
        gen_config = curriculum.get_generation_config(phase)
        for key, value in gen_config.items():
            setattr(model.generation_config, key, value)

        print(f"\n[OK] Generation config:")
        print(f"  Repetition penalty: {model.generation_config.repetition_penalty}")
        print(f"  Num beams: {model.generation_config.num_beams}")
        print(f"  No repeat ngram size: {model.generation_config.no_repeat_ngram_size}")

    # Patch LayerNorm for ROCm stability before any other model modification
    patch_layer_norms_for_rocm(model)

    # torch.compile fuses element-wise ops (including the ROCm LayerNorm fallback)
    # back into efficient kernels and reduces memory round-trips. Inductor backend
    # works on both ROCm and NVIDIA. Adds ~60s one-time compilation cost.
    if config.get('training', {}).get('torch_compile', False):
        print("\n[OK] Compiling model with torch.compile (Inductor)...")
        model = torch.compile(model)
        print("[OK] Compilation scheduled (JIT — first step will be slower)")

    # Freeze encoder if specified
    if config['model'].get('freeze_encoder', False):
        print("\n[WARNING] Freezing encoder weights (decoder-only training)")
        for param in model.encoder.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = model.num_parameters()
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    else:
        print(f"\n[OK] Model loaded: {model.num_parameters():,} parameters")

    # ============================================
    # Data Collator
    # ============================================
    data_collator = DataCollatorMoonshineSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        pad_token_id=model.config.pad_token_id,
    )

    # ============================================
    # Evaluation Metrics
    # ============================================
    wer_metric = evaluate.load('wer')

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad_token_id for decoding
        label_ids = np.where(label_ids == -100, model.config.pad_token_id, label_ids)
        pred_ids = np.where(pred_ids < 0, model.config.pad_token_id, pred_ids)

        # Decode
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Calculate WER (handle empty strings)
        pred_empty = np.array([p.strip() == "" for p in pred_str])
        label_empty = np.array([l.strip() == "" for l in label_str])

        wer_scores = np.ones(len(pred_str))
        wer_scores[pred_empty & label_empty] = 0  # Both empty = perfect

        non_silence = ~label_empty
        if np.any(non_silence):
            non_silence_wer = wer_metric.compute(
                predictions=np.array(pred_str)[non_silence].tolist(),
                references=np.array(label_str)[non_silence].tolist()
            )
            wer_scores[non_silence] = non_silence_wer

        avg_wer = np.mean(wer_scores)

        # Log examples
        print("\n" + "="*80)
        print(f"EVALUATION EXAMPLES (Phase {args.phase if config['curriculum']['enabled'] else 'Full'}):")
        print("="*80)
        for i in range(min(5, len(pred_str))):
            print(f"\nExample {i+1}:")
            print(f"  Prediction: '{pred_str[i]}'")
            print(f"  Reference:  '{label_str[i]}'")
        print("="*80)

        return {"wer": 100 * avg_wer}

    # ============================================
    # Training Arguments
    # ============================================
    train_config = config['training']

    # Override with phase-specific and CLI args
    max_steps = args.max_steps or phase.max_steps
    warmup_steps = train_config.get('warmup_steps', phase.warmup_steps)
    if args.test_mode:
        test_max = max(50, (500 // train_config['per_device_train_batch_size']) * 2)
        max_steps = min(max_steps, test_max)
        warmup_steps = min(warmup_steps, max(10, max_steps // 10))
        print(f"[WARNING] TEST MODE: Capping max_steps to {max_steps}, warmup_steps to {warmup_steps}")
    learning_rate = phase.learning_rate

    # schedule_free_* optimizers handle warmup internally; an external warmup scheduler
    # double-warms the LR and corrupts the optimizer's weight_sum accounting.
    optim_name = train_config.get('optim', 'adamw_torch')
    if optim_name.startswith('schedule_free'):
        effective_scheduler = 'constant'
    else:
        effective_scheduler = train_config.get('lr_scheduler_type', 'linear')

    # logging_dir was deprecated in transformers v5.2; use the env var instead.
    if 'logging_dir' in train_config:
        os.environ.setdefault('TENSORBOARD_LOGGING_DIR', train_config['logging_dir'])

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,

        # Batch sizes
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],

        # Optimization
        optim=optim_name,
        learning_rate=learning_rate,
        lr_scheduler_type=effective_scheduler,
        warmup_steps=warmup_steps,
        max_grad_norm=train_config['max_grad_norm'],
        max_steps=max_steps,
        label_smoothing_factor=phase.label_smoothing,

        # Memory optimization
        gradient_checkpointing=train_config['gradient_checkpointing'],
        fp16=train_config['fp16'],
        fp16_full_eval=train_config.get('fp16_full_eval', False),
        bf16=train_config.get('bf16', False),
        bf16_full_eval=train_config.get('bf16_full_eval', False),

        # Evaluation
        eval_strategy=train_config['eval_strategy'],
        eval_steps=train_config['eval_steps'],
        save_steps=train_config['save_steps'],
        logging_steps=train_config['logging_steps'],
        predict_with_generate=train_config['predict_with_generate'],

        # Model selection
        load_best_model_at_end=train_config['load_best_model_at_end'],
        metric_for_best_model=train_config['metric_for_best_model'],
        greater_is_better=train_config['greater_is_better'],

        # Logging
        report_to=train_config['report_to'],

        # Hub
        push_to_hub=args.push_to_hub or train_config.get('push_to_hub', False),
        hub_model_id=train_config.get('hub_model_id'),

        # Run name
        run_name=f"moonshine_phase{args.phase}" if config['curriculum']['enabled'] else "moonshine_full",
    )

    # ============================================
    # Initialize Trainer
    # ============================================
    generation_config = curriculum.get_generation_config(phase)

    trainer = MoonshineSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        generation_config=generation_config,
    )

    # Save processor
    processor.save_pretrained(output_dir)

    # ============================================
    # Start Training
    # ============================================
    print("\n" + "="*80)
    print(f"STARTING TRAINING")
    print("="*80)
    print(f"Training samples: {len(dataset_dict['train']):,}")
    print(f"Evaluation samples: {len(dataset_dict['test']):,}")
    print(f"Max steps: {max_steps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Effective batch size: {train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}")
    if config['curriculum']['enabled']:
        print(f"Target WER: <{phase.target_wer}%")
    print("="*80 + "\n")

    trainer.train()

    # ============================================
    # Save Final Model
    # ============================================
    final_path = os.path.join(output_dir, 'final')
    trainer.save_model(final_path)
    print(f"\n[OK] Saved final model to: {final_path}")

    # ============================================
    # Final Evaluation
    # ============================================
    print("\nRunning final evaluation...")

    # Convert model to FP32 before final evaluation to avoid dtype mismatch
    if training_args.fp16:
        print("Converting model to FP32 for final evaluation...")
        model = model.float()
        trainer.model = model

    try:
        results = trainer.evaluate()
    except RuntimeError as e:
        if "should be the same" in str(e):
            print(f"\n⚠️  Final evaluation skipped due to dtype mismatch (this is a known issue with FP16 training)")
            print(f"Your model was saved successfully to: {training_args.output_dir}/final")
            print(f"\nYou can evaluate it separately with:")
            print(f"  python scripts/evaluate.py --model {training_args.output_dir}/final --dataset {config['dataset']['name']} --split test")
            results = None
        else:
            raise

    print("\n" + "="*60)
    if config['curriculum']['enabled']:
        print(f"PHASE {args.phase} TRAINING COMPLETE")
    else:
        print("TRAINING COMPLETE")
    print("="*60)

    if results is not None:
        for key, value in results.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        if config['curriculum']['enabled']:
            actual_wer = results.get('eval_wer', 100)
            if actual_wer < phase.target_wer:
                print(f"\n[OK] SUCCESS! WER ({actual_wer:.1f}%) < Target ({phase.target_wer}%)")
                if args.phase < len(curriculum.phases):
                    print(f"\n➜ Ready for Phase {args.phase + 1}!")
                    print(f"   Run: python train.py --config {args.config} "
                          f"--phase {args.phase + 1} --resume {output_dir}")
            else:
                print(f"\n[WARNING] WER ({actual_wer:.1f}%) > Target ({phase.target_wer}%)")
                print("   Consider training longer or adjusting parameters.")
    else:
        print("\nNote: Final evaluation was skipped. Run evaluate.py separately for metrics.")

    print("="*60)


if __name__ == '__main__':
    main()
