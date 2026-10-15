"""4-bit weight-only quantisation (ONNX MatMulNBits) of a fine-tuned text
detector, for smaller browser downloads: about 40% smaller than the
dynamic-int8 file, which lets the site ship a bigger model in the same budget.
Accuracy is checked afterwards with the browser's own runtime
(eval/neural-text.js, ORT Web WASM) on the in-house set.

  python eval/train/text/quant4.py <hf model dir> <out dir> [block size, default 32]
  → <out>/model_q4.onnx (+ model.onnx fp32, tokenizer files)
"""
import os
import shutil
import sys

import onnx
import torch
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

src, out = sys.argv[1], sys.argv[2]
block = int(sys.argv[3]) if len(sys.argv) > 3 else 32
os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForSequenceClassification.from_pretrained(src, attn_implementation="eager").eval()
enc = tok(["An example sentence to trace the graph."], return_tensors="pt")
fp32 = f"{out}/model.onnx"
# Same export as eval/train/text/train_raid.py (inputs, names, opset).
torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), fp32,
                  input_names=["input_ids", "attention_mask"], output_names=["logits"],
                  dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}},
                  opset_version=17, dynamo=False)
q = MatMulNBitsQuantizer(onnx.load(fp32), block_size=block, is_symmetric=True)
q.process()
q.model.save_model_to_file(f"{out}/model_q4.onnx", use_external_data_format=False)
tok.save_pretrained(out)
for f in ("tokenizer.json", "tokenizer_config.json"):
    if not os.path.exists(f"{out}/{f}"):
        shutil.copy(os.path.join(src, f), out)
print({f: os.path.getsize(f"{out}/{f}") for f in ("model.onnx", "model_q4.onnx")})
