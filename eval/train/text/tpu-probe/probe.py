"""Kaggle TPU probe: which frameworks exist, and how fast Ettin-150M trains
on a v5e-8 with torch_xla (vs ~44 rows/s on 2xT4)."""
import importlib, subprocess, sys, time
# The TPU VM image ships JAX; torch_xla may need installing to match its torch.
try:
    import torch_xla  # noqa: F401
except Exception:
    import torch
    v = torch.__version__.split("+")[0]
    print("installing torch_xla for torch", v, flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"torch_xla[tpu]=={v}",
                        "-f", "https://storage.googleapis.com/libtpu-releases/index.html"], capture_output=True, text=True)
    print("pip rc", r.returncode, r.stderr[-600:], flush=True)
for m in ("torch", "torch_xla", "jax", "flax", "transformers"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"), flush=True)
    except Exception as e:
        print(m, "MISSING", repr(e)[:120], flush=True)
try:
    import jax
    print("jax devices", jax.devices(), flush=True)
except Exception as e:
    print("jax devices failed", repr(e)[:200], flush=True)
try:
    from transformers import FlaxAutoModelForSequenceClassification
    FlaxAutoModelForSequenceClassification.from_pretrained("jhu-clsp/ettin-encoder-150m", num_labels=2, from_pt=True)
    print("flax ettin: OK", flush=True)
except Exception as e:
    print("flax ettin: NO", repr(e)[:200], flush=True)
try:
    import torch, torch_xla.core.xla_model as xm
    import torch_xla
    dev = torch_xla.device()
    print("xla devices", xm.xla_device_hw(dev), torch_xla.runtime.global_runtime_device_count(), flush=True)
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("jhu-clsp/ettin-encoder-150m", num_labels=2, attn_implementation="eager").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    B, L = 32, 512
    x = torch.randint(5, 50000, (B, L)).to(dev); m = torch.ones(B, L, dtype=torch.long).to(dev); y = torch.randint(0, 2, (B,)).to(dev)
    for i in range(23):
        if i == 3:
            xm.wait_device_ops(); t0 = time.time()
        with torch.autocast("xla", dtype=torch.bfloat16):
            loss = model(input_ids=x, attention_mask=m, labels=y).loss
        loss.backward(); opt.step(); opt.zero_grad(); xm.mark_step()
    xm.wait_device_ops()
    print(f"single-core: {20 * B / (time.time() - t0):.1f} rows/s at L={L}", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
