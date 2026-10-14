

def requant():
    """Find an int8 export that keeps the fp32 EER. Dynamic quantisation of
    the convolutional feature encoder can wreck wav2vec-style models, so try
    MatMul-only variants too. CPU only; appended to audio_train.py."""
    from transformers import AutoModelForAudioClassification
    from onnxruntime.quantization import QuantType, quantize_dynamic
    cfg = glob.glob("/kaggle/input/**/model/config.json", recursive=True)[0]
    model = AutoModelForAudioClassification.from_pretrained(os.path.dirname(cfg)).eval()
    torch.set_num_threads(os.cpu_count())
    os.makedirs(f"{OUT}/onnx", exist_ok=True)
    torch.onnx.export(model, (torch.randn(1, WIN),), "/tmp/fp32.onnx", input_names=["input_values"], output_names=["logits"],
                      dynamic_axes={"input_values": {0: "b", 1: "n"}, "logits": {0: "b"}}, opset_version=17, dynamo=False)
    variants = {
        "s8-all": dict(weight_type=QuantType.QInt8),
        "s8-matmul": dict(weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"]),
        "u8-matmul": dict(weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul"]),
        "s8pc-matmul": dict(weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"]),
    }
    rows = sample(itw(), EVAL_N) + sample(asv21df(), EVAL_N // 2)
    y = np.array([r[1] for r in rows])
    res = {}
    _, ref = evaluate(model, rows)
    ref = np.array(ref)
    ok = ~np.isnan(ref)
    res["fp32"] = {"eer": round(eer(y[ok], ref[ok]), 4)}
    print("fp32", res["fp32"], flush=True)
    best, best_eer = None, 9
    for name, kw in variants.items():
        path = f"/tmp/{name}.onnx"
        try:
            quantize_dynamic("/tmp/fp32.onnx", path, **kw)
            _, s = evaluate(OnnxModel(path), rows)
            s = np.array(s)
            e = eer(y[ok], s[ok])
            res[name] = {"eer": round(e, 4), "bytes": os.path.getsize(path), "mean_abs_vs_fp32": round(float(np.nanmean(np.abs(s - ref))), 4)}
            print(name, res[name], flush=True)
            if e < best_eer:
                best, best_eer = name, e
        except Exception as ex:
            res[name] = {"error": repr(ex)[:200]}
            print(name, "FAILED", repr(ex)[:200], flush=True)
        json.dump(res, open(f"{OUT}/requant.json", "w"), indent=1)
    import shutil
    shutil.copy(f"/tmp/{best}.onnx", f"{OUT}/onnx/model_quantized.onnx")
    res["chosen"] = best
    json.dump(res, open(f"{OUT}/requant.json", "w"), indent=1)


requant()
