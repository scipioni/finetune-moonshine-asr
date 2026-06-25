import onnxruntime as ort
import numpy as np
import soundfile as sf
import os
import json

# =========================
# PATH MODELLO
# =========================
MODEL_DIR = "moonshine-it/ort"  # <-- cambia qui se serve

ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_model.ort")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder_model_merged.ort")

AUDIO_PATH = "test-completo-1m.wav"

# =========================
# LOAD AUDIO
# =========================
audio, sr = sf.read(AUDIO_PATH)

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

audio = audio.astype(np.float32)

# =========================
# LOAD MODELS
# =========================
encoder = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
decoder = ort.InferenceSession(DECODER_PATH, providers=["CPUExecutionProvider"])

print("✅ Modelli caricati")

# =========================
# ENCODER
# =========================
enc_inputs = encoder.get_inputs()
enc_input_name = enc_inputs[0].name

audio = audio[np.newaxis, :]

attention_mask = np.ones_like(audio, dtype=np.int64)

enc_out = encoder.run(
    None,
    {
        "input_values": audio,
        "attention_mask": attention_mask
    }
)

encoder_hidden = enc_out[0]

print("✅ Encoder OK")

# =========================
# DECODER LOOP (GREEDY BASIC)
# =========================

# ATTENZIONE: token speciali dipendono dal tokenizer_extended
# questi sono placeholder comuni
BOS_TOKEN = 1
EOS_TOKEN = 2

tokens = [BOS_TOKEN]

max_len = 200

for i in range(max_len):

    dec_inputs = decoder.get_inputs()
    input_name = dec_inputs[0].name

    ort_inputs = {
        input_name: np.array([tokens], dtype=np.int64),
    }

    outputs = decoder.run(None, ort_inputs)

    logits = outputs[0]
    next_token = int(np.argmax(logits[0, -1]))

    tokens.append(next_token)

    if next_token == EOS_TOKEN:
        break

print("✅ Tokens:", tokens)

# =========================
# OUTPUT RAW
# =========================
print("\n--- OUTPUT ---")
print(tokens)

# =========================
# SALVATAGGIO
# =========================
with open("output.json", "w") as f:
    json.dump({"tokens": tokens}, f)
