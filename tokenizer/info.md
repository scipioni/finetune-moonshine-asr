# 🚀 Installazione ed avvio

```bash
pip install -r requirements_moonshine_it.txt
python build_moonshine_italian.py
```

---

# ⚙️ Opzioni utili da riga di comando

## 🧪 Solo tokenizer (senza training)
Utile per test iniziali:

```bash
python build_moonshine_italian.py --only-tokenizer
```

---

## ♻️ Riutilizza tokenizer inglese
Meno step di training richiesti (≈ 8k invece di 20k):

```bash
python build_moonshine_italian.py --no-new-tokenizer --steps 8000
```

---

## ⚡ Test rapido con pochi dati
Per verificare velocemente il funzionamento:

```bash
python build_moonshine_italian.py --max-samples 500 --steps 200
```
