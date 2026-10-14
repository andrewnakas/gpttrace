

def onnx_eval():
    """CPU: score what the browser would run — the base int8 classifier, the
    fine-tuned model quantised two ways, and their average — on the same
    held-out images. Appended to image_train.py (its loaders and views)."""
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import ViTForImageClassification
    from sklearn.metrics import roc_auc_score
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count()
    base = hf_hub_download(REPO, "onnx/model_int8.onnx", revision=REV)
    cfg = glob.glob("/kaggle/input/**/model/config.json", recursive=True)[0]
    ft = ViTForImageClassification.from_pretrained(os.path.dirname(cfg)).eval()
    torch.onnx.export(ft, (torch.randn(1, 3, CROP, CROP),), "/tmp/ft.onnx", input_names=["pixel_values"], output_names=["logits"],
                      dynamic_axes={"pixel_values": {0: "b"}, "logits": {0: "b"}}, opset_version=17, dynamo=False)
    quantize_dynamic("/tmp/ft.onnx", "/tmp/ft_all.onnx", weight_type=QuantType.QInt8)
    quantize_dynamic("/tmp/ft.onnx", "/tmp/ft_mm.onnx", weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    models = {k: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]) for k, p in
              {"base": base, "ft_all": "/tmp/ft_all.onnx", "ft_mm": "/tmp/ft_mm.onnx"}.items()}
    hai, hreal = heldout()
    items = [(d, "dragon+megalith", g, "ai") for d, g in hai] + [(d, "dragon+megalith", "megalith", "human") for d, _ in hreal]
    items += internal_bench()
    print("items", len(items), flush=True)
    rows = []
    for i, (data, bench, gen, lab) in enumerate(items):
        try:
            views = torch.stack(eval_views(data)).numpy()
        except Exception:
            continue
        r = {"bench": bench, "gen": gen, "label": lab}
        for k, s in models.items():
            lg = np.array([s.run(None, {"pixel_values": views[j:j + 1]})[0].ravel()[0] for j in range(len(views))])
            r[k] = float(lg[0]); r[k + "_crop"] = float(lg[1:].max()) if len(lg) > 1 else np.nan
        rows.append(r)
        if i % 500 == 0:
            print(i, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(f"{OUT}/onnx_scores.parquet")
    res = {}
    for b, g in df.groupby("bench"):
        y = (g.label == "ai").astype(int)
        mx = lambda k: np.maximum(g[k], g[k + "_crop"].fillna(-99))
        res[b] = {k: round(float(roc_auc_score(y, v)), 4) for k, v in {
            "base": g.base, "base+crop": mx("base"), "ft_all": g.ft_all, "ft_mm": g.ft_mm,
            "avg(base,ft_mm)+crop": 0.5 * mx("base") + 0.5 * mx("ft_mm"), "avg(base,ft_all)+crop": 0.5 * mx("base") + 0.5 * mx("ft_all")}.items()}
        print(b, res[b], flush=True)
    os.makedirs(f"{OUT}/onnx", exist_ok=True)
    shutil.copy("/tmp/ft_mm.onnx", f"{OUT}/onnx/model_int8_matmul.onnx")
    json.dump(res, open(f"{OUT}/onnx_eval.json", "w"), indent=1)


import glob
import shutil
onnx_eval()
