import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import torch
from transformers import DistilBertTokenizer, DistilBertModel

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME  = "distilbert-base-uncased"
OUTPUT_PATH = "mcp_services/document_processor/model/distilbert.onnx"

print(f"Exporting {MODEL_NAME} to ONNX...")
print(f"Output: {OUTPUT_PATH}")
print()

# ── Load model and tokenizer ──────────────────────────────────────────────────
print("Step 1: Loading DistilBERT from HuggingFace (~250MB download)...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model     = DistilBertModel.from_pretrained(MODEL_NAME)
model.eval()
print("  Model loaded")

# ── Create dummy input for ONNX export ───────────────────────────────────────
print("Step 2: Creating dummy input...")
sample_text = "Income 1200000 salaried employment income proof document"
inputs      = tokenizer(
    sample_text,
    return_tensors="pt",
    max_length=128,
    padding="max_length",
    truncation=True
)
dummy_input_ids      = inputs["input_ids"]
dummy_attention_mask = inputs["attention_mask"]
print("  Dummy input created")

# ── Export to ONNX ────────────────────────────────────────────────────────────
print("Step 3: Exporting to ONNX...")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    OUTPUT_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids":       {0: "batch_size", 1: "sequence_length"},
        "attention_mask":  {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=14,
    do_constant_folding=True,
)
print(f"  Exported to {OUTPUT_PATH}")

# ── Verify the export ─────────────────────────────────────────────────────────
print("Step 4: Verifying ONNX file...")
import onnx
onnx_model = onnx.load(OUTPUT_PATH)
onnx.checker.check_model(onnx_model)
file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"  Verification passed")
print(f"  File size: {file_size_mb:.1f} MB")

# ── Quick inference test ──────────────────────────────────────────────────────
print("Step 5: Testing inference with ONNX Runtime...")
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(OUTPUT_PATH)
outputs = session.run(
    None,
    {
        "input_ids":      dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy(),
    }
)
print(f"  Inference OK — output shape: {outputs[0].shape}")

print()
print("=" * 60)
print("✅ ONNX export complete")
print(f"   File: {OUTPUT_PATH}")
print(f"   Size: {file_size_mb:.1f} MB")
print()
print("This file is the AIRS Model Scanning target.")
print("In CI/CD, AIRS scans this file before every deployment.")
print("If tampered — build fails.")
print("=" * 60)
