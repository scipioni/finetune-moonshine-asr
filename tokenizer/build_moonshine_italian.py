#!/usr/bin/env python3
"""
build_moonshine_italian.py
==========================
Script end-to-end per creare un tokenizer BPE italiano e
avviare il fine-tuning di Moonshine ASR sulla lingua italiana.

Esecuzione:
    pip install -r requirements_moonshine_it.txt
    python build_moonshine_italian.py

Opzioni principali (modifica la sezione CONFIG più in basso):
    - VOCAB_SIZE       : dimensione del vocabolario BPE (default 8192)
    - MAX_TRAIN_HOURS  : ore di audio da usare per il training (default: tutto)
    - TRAIN_STEPS      : steps di fine-tuning (default 20000)
    - USE_NEW_TOKENIZER: True = tokenizer italiano nuovo (consigliato per produzione)
                         False = riusa tokenizer inglese (più veloce da addestrare)
"""

import os
import sys
import json
import struct
import shutil
import argparse
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# CONFIG — modifica qui prima di eseguire
# ─────────────────────────────────────────────────────────────────
CONFIG = {
    # Tokenizer
    "VOCAB_SIZE": 8192,
    "USE_NEW_TOKENIZER": True,       # False = riusa tokenizer inglese (meno steps)

    # Dataset (Common Voice italiano su HuggingFace)
    "HF_DATASET": "mozilla-foundation/common_voice_17_0",
    "LANGUAGE_CODE": "it",
    "TRAIN_SPLIT": "train",
    "TEST_SPLIT": "validation",
    "MAX_TRAIN_SAMPLES": None,       # None = tutto il dataset; int = limite campioni

    # Modello base Moonshine
    "BASE_MODEL": "UsefulSensors/moonshine-tiny",

    # Training
    "OUTPUT_DIR": "./moonshine-tiny-it",
    "TRAIN_STEPS": 20000,
    "BATCH_SIZE": 16,
    "GRAD_ACCUM": 2,                 # batch effettivo = BATCH_SIZE * GRAD_ACCUM
    "LEARNING_RATE": 5e-5,
    "WARMUP_STEPS": 1000,
    "EVAL_STEPS": 1000,
    "SAVE_STEPS": 2000,
    "FP16": True,                    # False su CPU o GPU senza fp16
    "MAX_AUDIO_DURATION": 20.0,      # secondi
    "MIN_AUDIO_DURATION": 1.0,

    # Moonshine convert_tokenizer.py (da clonare la repo)
    "MOONSHINE_REPO": "https://github.com/moonshine-ai/moonshine.git",
    "MOONSHINE_REPO_DIR": "./moonshine_repo",
}
# ─────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════
# STEP 0 — Controlla dipendenze
# ══════════════════════════════════════════════════════════════════

REQUIREMENTS = [
    "torch",
    "transformers>=4.40",
    "datasets",
    "tokenizers",
    "evaluate",
    "jiwer",                 # metrica WER
    "soundfile",
    "librosa",
    "schedulefree",          # optimizer senza lr schedule
    "accelerate",
    "sentencepiece",         # per convert_tokenizer.py (opzionale)
    "torchaudio",
]

def check_imports():
    missing = []
    import importlib
    for pkg in ["torch", "transformers", "datasets", "tokenizers",
                "evaluate", "soundfile", "librosa", "accelerate"]:
        try:
            importlib.import_module(pkg.split(">=")[0].replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[!] Pacchetti mancanti: {missing}")
        print("    Esegui: pip install " + " ".join(missing))
        sys.exit(1)
    print("[✓] Dipendenze OK")


# ══════════════════════════════════════════════════════════════════
# STEP 1 — Scarica dataset italiano
# ══════════════════════════════════════════════════════════════════

def load_italian_dataset():
    from datasets import load_dataset, Audio

    print("\n[1/5] Caricamento dataset italiano...")
    print(f"      Dataset : {CONFIG['HF_DATASET']}")
    print(f"      Lingua  : {CONFIG['LANGUAGE_CODE']}")

    ds_train = load_dataset(
        CONFIG["HF_DATASET"],
        CONFIG["LANGUAGE_CODE"],
        split=CONFIG["TRAIN_SPLIT"],
        trust_remote_code=True,
    )
    ds_test = load_dataset(
        CONFIG["HF_DATASET"],
        CONFIG["LANGUAGE_CODE"],
        split=CONFIG["TEST_SPLIT"],
        trust_remote_code=True,
    )

    # Cast audio a 16kHz
    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000))
    ds_test  = ds_test.cast_column("audio",  Audio(sampling_rate=16000))

    # Limite campioni
    if CONFIG["MAX_TRAIN_SAMPLES"]:
        ds_train = ds_train.select(range(min(CONFIG["MAX_TRAIN_SAMPLES"], len(ds_train))))
        ds_test  = ds_test.select(range(min(CONFIG["MAX_TRAIN_SAMPLES"] // 10, len(ds_test))))

    print(f"      Train: {len(ds_train):,} campioni  |  Test: {len(ds_test):,} campioni")
    return ds_train, ds_test


# ══════════════════════════════════════════════════════════════════
# STEP 2 — Addestra tokenizer BPE italiano
# ══════════════════════════════════════════════════════════════════

def train_italian_tokenizer(ds_train):
    """
    Addestra un tokenizer BPE byte-level sul testo italiano delle trascrizioni.
    Restituisce il percorso di tokenizer_it.json.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormSeq

    out_json = Path("tokenizer_it.json")

    if out_json.exists():
        print(f"\n[2/5] tokenizer_it.json già presente, skip addestramento.")
        return str(out_json)

    print(f"\n[2/5] Addestramento tokenizer BPE italiano (vocab={CONFIG['VOCAB_SIZE']})...")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder       = ByteLevelDecoder()

    # NON facciamo lowercase: l'italiano ha maiuscole nei nomi propri
    # e vogliamo che il modello le gestisca correttamente.

    trainer = BpeTrainer(
        vocab_size=CONFIG["VOCAB_SIZE"],
        min_frequency=2,
        special_tokens=["<s>", "</s>", "[UNK]"],
        # <s>=0 (BOS), </s>=1 (EOS/PAD), [UNK]=2
    )

    def text_iterator():
        batch_size = 1000
        texts = []
        for item in ds_train:
            sentence = item.get("sentence", item.get("text", ""))
            if sentence:
                texts.append(sentence.strip())
            if len(texts) >= batch_size:
                yield from texts
                texts = []
        if texts:
            yield from texts

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(str(out_json))

    vocab_size_actual = tokenizer.get_vocab_size()
    print(f"      Vocab finale: {vocab_size_actual} token  →  {out_json}")
    return str(out_json)


# ══════════════════════════════════════════════════════════════════
# STEP 3 — Converti tokenizer.json → tokenizer.bin
# ══════════════════════════════════════════════════════════════════

def write_bin_tokenizer(tokens: list, output_path: str) -> None:
    """
    Scrive il formato BinTokenizer usato da Moonshine (C++ core).
    Replica esatta di scripts/convert_tokenizer.py della repo ufficiale.

    Formato per ogni token:
        len < 128  → 1 byte (len) + N bytes (stringa)
        len >= 128 → 2 bytes (len encoded) + N bytes
        len == 0   → 1 byte 0x00
    """
    with open(output_path, "wb") as f:
        for token_bytes in tokens:
            length = len(token_bytes)
            if length == 0:
                f.write(b"\x00")
            elif length < 128:
                f.write(bytes([length]))
                f.write(token_bytes)
            else:
                first_byte  = (length % 128) + 128
                second_byte = length // 128
                f.write(bytes([first_byte, second_byte]))
                f.write(token_bytes)


def convert_tokenizer_json_to_bin(json_path: str, bin_path: str) -> int:
    """Converte tokenizer.json HuggingFace → tokenizer.bin Moonshine."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data.get("model", {}).get("vocab", {})
    if not vocab:
        raise ValueError("Vocabolario non trovato in tokenizer.json")

    vocab_size = max(vocab.values()) + 1
    tokens = [b""] * vocab_size

    for token_str, token_id in vocab.items():
        tokens[token_id] = token_str.encode("utf-8")

    # Gestione added_tokens (special tokens)
    for added in data.get("added_tokens", []):
        tid     = added.get("id")
        content = added.get("content", "")
        if tid is not None:
            if tid >= len(tokens):
                tokens.extend([b""] * (tid - len(tokens) + 1))
            tokens[tid] = content.encode("utf-8")

    write_bin_tokenizer(tokens, bin_path)
    return len(tokens)


def convert_tokenizer(json_path: str) -> str:
    bin_path = json_path.replace(".json", ".bin")

    if Path(bin_path).exists():
        print(f"\n[3/5] {bin_path} già presente, skip conversione.")
        return bin_path

    print(f"\n[3/5] Conversione tokenizer.json → tokenizer.bin ...")
    n = convert_tokenizer_json_to_bin(json_path, bin_path)
    size = Path(bin_path).stat().st_size
    print(f"      Scritti {n} token  |  {size:,} bytes  →  {bin_path}")
    return bin_path


# ══════════════════════════════════════════════════════════════════
# STEP 4 — Prepara modello + processor per l'italiano
# ══════════════════════════════════════════════════════════════════

def build_processor_and_model(tokenizer_json_path: str):
    import torch
    from transformers import (
        AutoProcessor,
        AutoFeatureExtractor,
        MoonshineForConditionalGeneration,
        PreTrainedTokenizerFast,
    )
    from tokenizers import Tokenizer as HFTokenizer

    print(f"\n[4/5] Caricamento modello base: {CONFIG['BASE_MODEL']} ...")

    # Feature extractor (elaborazione audio) — invariato
    feature_extractor = AutoFeatureExtractor.from_pretrained(CONFIG["BASE_MODEL"])

    if CONFIG["USE_NEW_TOKENIZER"]:
        print("      Modalità: tokenizer ITALIANO (nuovo BPE)")
        hf_tok = HFTokenizer.from_file(tokenizer_json_path)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=hf_tok,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="[UNK]",
            pad_token="</s>",
        )
    else:
        print("      Modalità: tokenizer INGLESE riusato (più veloce)")
        processor_en = AutoProcessor.from_pretrained(CONFIG["BASE_MODEL"])
        tokenizer = processor_en.tokenizer
        tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "</s>",
        })

    # Costruisci processor combinato
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(CONFIG["BASE_MODEL"])
    processor.tokenizer = tokenizer

    # Modello
    model = MoonshineForConditionalGeneration.from_pretrained(CONFIG["BASE_MODEL"])

    # Se tokenizer nuovo: ridimensiona embedding
    if CONFIG["USE_NEW_TOKENIZER"]:
        old_vocab = model.config.vocab_size
        new_vocab = len(tokenizer)
        model.resize_token_embeddings(new_vocab)
        print(f"      Embedding ridimensionato: {old_vocab} → {new_vocab} token")

    model.config.pad_token_id              = tokenizer.pad_token_id
    model.config.bos_token_id              = tokenizer.bos_token_id
    model.config.eos_token_id              = tokenizer.eos_token_id
    model.config.decoder_start_token_id   = tokenizer.bos_token_id
    model.config.use_cache                 = False

    print(f"      Parametri totali: {model.num_parameters():,}")
    return processor, model


# ══════════════════════════════════════════════════════════════════
# STEP 5 — Fine-tuning
# ══════════════════════════════════════════════════════════════════

class DataCollatorMoonshine:
    """Collator che gestisce padding audio + labels per Moonshine."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        import torch

        input_values  = [{"input_values": f["input_values"]} for f in features]
        label_features = [f["labels"] for f in features]

        batch = self.processor.feature_extractor.pad(
            input_values, return_tensors="pt", return_attention_mask=True
        )

        label_tensors = [torch.tensor(lb, dtype=torch.long) for lb in label_features]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            label_tensors, batch_first=True, padding_value=-100
        )

        bos_id = self.processor.tokenizer.bos_token_id
        pad_id = self.processor.tokenizer.pad_token_id
        decoder_inputs = [
            torch.tensor([bos_id] + lb[:-1], dtype=torch.long)
            for lb in label_features
        ]
        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            decoder_inputs, batch_first=True, padding_value=pad_id
        )

        batch["decoder_input_ids"] = decoder_input_ids
        batch["labels"]            = labels_padded
        return batch


def preprocess_dataset(ds, processor):
    """Estrae feature audio e tokenizza il testo."""
    import numpy as np

    def process(batch):
        audio = batch["audio"]
        sr    = audio["sampling_rate"]
        arr   = np.array(audio["array"], dtype=np.float32)

        # Filtro per durata
        duration = len(arr) / sr
        if duration < CONFIG["MIN_AUDIO_DURATION"] or duration > CONFIG["MAX_AUDIO_DURATION"]:
            return {"input_values": None, "labels": None}

        inputs = processor.feature_extractor(
            arr, sampling_rate=sr, return_tensors="np"
        )

        sentence = batch.get("sentence", batch.get("text", ""))
        with processor.tokenizer.as_target_tokenizer():
            labels = processor.tokenizer(sentence.strip()).input_ids

        return {
            "input_values": inputs.input_values[0],
            "labels":       labels,
        }

    ds = ds.map(
        process,
        remove_columns=ds.column_names,
        num_proc=1,
        desc="Preprocessing",
    )
    ds = ds.filter(lambda x: x["input_values"] is not None)
    return ds


def run_finetuning(processor, model, ds_train, ds_test):
    import torch
    import numpy as np
    import evaluate
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    print(f"\n[5/5] Avvio fine-tuning ...")
    print(f"      Output dir : {CONFIG['OUTPUT_DIR']}")
    print(f"      Steps      : {CONFIG['TRAIN_STEPS']:,}")
    print(f"      Batch eff. : {CONFIG['BATCH_SIZE'] * CONFIG['GRAD_ACCUM']}")

    # Preprocessing
    print("      Preprocessing train set...")
    ds_train_enc = preprocess_dataset(ds_train, processor)
    print("      Preprocessing test set...")
    ds_test_enc  = preprocess_dataset(ds_test,  processor)

    print(f"      Campioni dopo filtro durata: train={len(ds_train_enc):,}  test={len(ds_test_enc):,}")

    data_collator = DataCollatorMoonshine(processor)
    wer_metric    = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids

        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
        pred_ids  = np.where(pred_ids < 0,      processor.tokenizer.pad_token_id, pred_ids)

        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        non_empty = [i for i, l in enumerate(label_str) if l.strip()]
        if non_empty:
            wer = wer_metric.compute(
                predictions=[pred_str[i]  for i in non_empty],
                references= [label_str[i] for i in non_empty],
            )
        else:
            wer = 1.0

        # Log 3 esempi
        for i in range(min(3, len(pred_str))):
            print(f"  [pred]  {pred_str[i]}")
            print(f"  [ref ]  {label_str[i]}")
        return {"wer": round(100 * wer, 2)}

    # Prova optimizer schedule-free se disponibile
    try:
        import schedulefree
        optim = "schedulefree_adamw"
        lr_scheduler = "constant"
        print("      Optimizer: ScheduleFree AdamW")
    except ImportError:
        optim = "adamw_torch"
        lr_scheduler = "linear"
        print("      Optimizer: AdamW standard")

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = CONFIG["OUTPUT_DIR"],
        per_device_train_batch_size = CONFIG["BATCH_SIZE"],
        per_device_eval_batch_size  = CONFIG["BATCH_SIZE"],
        gradient_accumulation_steps = CONFIG["GRAD_ACCUM"],
        learning_rate               = CONFIG["LEARNING_RATE"],
        warmup_steps                = CONFIG["WARMUP_STEPS"],
        max_steps                   = CONFIG["TRAIN_STEPS"],
        optim                       = optim,
        lr_scheduler_type           = lr_scheduler,
        fp16                        = CONFIG["FP16"],
        fp16_full_eval              = CONFIG["FP16"],
        gradient_checkpointing      = True,
        eval_strategy               = "steps",
        eval_steps                  = CONFIG["EVAL_STEPS"],
        save_steps                  = CONFIG["SAVE_STEPS"],
        save_total_limit            = 3,
        predict_with_generate       = True,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        logging_steps               = 100,
        report_to                   = ["tensorboard"],
        run_name                    = "moonshine-tiny-it",
        group_by_length             = True,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = ds_train_enc,
        eval_dataset    = ds_test_enc,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
        processing_class= processor,
    )

    # Salva processor prima del training
    Path(CONFIG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(CONFIG["OUTPUT_DIR"])

    trainer.train()

    # Salva modello finale
    final_path = Path(CONFIG["OUTPUT_DIR"]) / "final"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"\n[✓] Modello salvato in: {final_path}")


# ══════════════════════════════════════════════════════════════════
# STEP 6 (bonus) — Copia tokenizer.bin nella cartella del modello
# ══════════════════════════════════════════════════════════════════

def copy_bin_tokenizer(bin_path: str):
    """
    Copia tokenizer.bin nella cartella finale del modello,
    così può essere usato direttamente con il runtime C++ di Moonshine.
    """
    dst = Path(CONFIG["OUTPUT_DIR"]) / "final" / "tokenizer.bin"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bin_path, dst)
    print(f"[✓] tokenizer.bin copiato in: {dst}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build Moonshine Italian ASR end-to-end")
    parser.add_argument("--only-tokenizer", action="store_true",
                        help="Esegui solo addestramento tokenizer + conversione .bin, senza fine-tuning")
    parser.add_argument("--no-new-tokenizer", action="store_true",
                        help="Riusa tokenizer inglese (più veloce, meno accurato)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override TRAIN_STEPS")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limita il numero di campioni del dataset")
    args = parser.parse_args()

    if args.no_new_tokenizer:
        CONFIG["USE_NEW_TOKENIZER"] = False
    if args.steps:
        CONFIG["TRAIN_STEPS"] = args.steps
    if args.max_samples:
        CONFIG["MAX_TRAIN_SAMPLES"] = args.max_samples

    print("=" * 60)
    print("  Moonshine Italian ASR — build end-to-end")
    print("=" * 60)
    print(f"  Tokenizer nuovo  : {CONFIG['USE_NEW_TOKENIZER']}")
    print(f"  Vocab size       : {CONFIG['VOCAB_SIZE']}")
    print(f"  Steps training   : {CONFIG['TRAIN_STEPS']:,}")
    print(f"  Output dir       : {CONFIG['OUTPUT_DIR']}")
    print("=" * 60)

    check_imports()

    # 1. Dataset
    ds_train, ds_test = load_italian_dataset()

    # 2. Tokenizer BPE italiano
    if CONFIG["USE_NEW_TOKENIZER"]:
        tok_json = train_italian_tokenizer(ds_train)
    else:
        # Scarica tokenizer.json dal modello inglese e salvalo
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(CONFIG["BASE_MODEL"])
        tok_json = "tokenizer_it.json"
        proc.tokenizer.save_pretrained(".")
        # HF salva come tokenizer.json nella cartella corrente
        if Path("tokenizer.json").exists() and not Path(tok_json).exists():
            Path("tokenizer.json").rename(tok_json)

    # 3. Converti → .bin per il runtime C++ di Moonshine
    tok_bin = convert_tokenizer(tok_json)

    if args.only_tokenizer:
        print("\n[✓] Solo tokenizer richiesto. Fine.")
        print(f"    JSON : {tok_json}")
        print(f"    BIN  : {tok_bin}")
        return

    # 4. Modello + processor
    processor, model = build_processor_and_model(tok_json)

    # 5. Fine-tuning
    run_finetuning(processor, model, ds_train, ds_test)

    # 6. Copia .bin nella cartella finale
    copy_bin_tokenizer(tok_bin)

    print("\n" + "=" * 60)
    print("  ✓  COMPLETATO")
    print("=" * 60)
    print(f"\n  Modello HuggingFace : {CONFIG['OUTPUT_DIR']}/final/")
    print(f"  Tokenizer .bin      : {CONFIG['OUTPUT_DIR']}/final/tokenizer.bin")
    print(f"\n  Per usarlo con python:")
    print(f"    from transformers import pipeline")
    print(f"    asr = pipeline('automatic-speech-recognition',")
    print(f"                   model='{CONFIG['OUTPUT_DIR']}/final')")
    print(f"    print(asr('audio.wav'))")
    print()


if __name__ == "__main__":
    main()
