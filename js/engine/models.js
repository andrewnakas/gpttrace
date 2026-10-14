/* Neural models GPTTrace can download. Everything runs on the visitor's
 * device; these files are only ever downloaded (GET), never sent anything.
 * Small models are served from our own origin; models over Cloudflare's
 * 25 MiB per-file limit come from R2 (models.gpttrace.com) with Hugging Face
 * as the fallback. */
export const MODELS = {
  image: {
    id: "community-forensics-vit-s-int8",
    name: "Community Forensics ViT-S (int8)",
    paper: "https://arxiv.org/abs/2411.04125",
    source: "https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT",
    revision: "ac6ee457bea904a373065754107451793b56db00",
    license: "MIT",
    trainedOn: "2.7M images from 4,803 generators",
    urls: ["/models/cf-vit-s-ac6ee45/model_int8.onnx",
      "https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/resolve/ac6ee457bea904a373065754107451793b56db00/onnx/model_int8.onnx"],
    bytes: 23028026, sha256: "968f113f1107d58b…",
    input: { shortestEdge: 440, crop: 384, mean: [0.48145466, 0.4578275, 0.40821073], std: [0.26862954, 0.26130258, 0.27577711] }
  },
  // Our fine-tune of the image model above (eval/train/image, Kaggle
  // gpttrace-image-cfvit-ft) on commercially usable data: DRAGON (CC BY-SA),
  // open-image-preferences (Apache-2.0), nano-banana (MIT) and megalith CC0
  // photos. int8 on MatMul only. The worker averages its logits with the
  // base model's (gpttrace-image-onnxeval: AIGIBench AUC 0.933 → 0.992).
  image2: {
    id: "gpttrace-cfvit-ft-v1-int8",
    name: "GPTTrace fine-tune of Community Forensics ViT-S (int8)",
    source: "https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT",
    license: "MIT (base model); fine-tuned on DRAGON (CC BY-SA 4.0), open-image-preferences (Apache-2.0), nano-banana (MIT), megalith (CC0)",
    urls: ["/models/cf-vit-ft-v1/model_int8.onnx"],
    bytes: 23860732, sha256: "9eeafb1e166965d0…"
  },
  // Optional "deep scan" for text. 144 MB — above Cloudflare's 25 MiB
  // per-file limit, so it is served from our own origin in 7 parts
  // (model_quantized.onnx.part0…6, made with `split -b 24000000`) that the
  // worker joins and caches as one file. Trained by
  // eval/train/text (Kaggle gpttrace-raid-ettin150-v3); RAID leaderboard
  // submission liamdugan/raid#220 is the same architecture (v2 weights).
  text: {
    id: "gpttrace-ettin150-raid-mage-v3-int8",
    name: "GPTTrace text detector (Ettin-150M, int8)",
    source: "https://huggingface.co/jhu-clsp/ettin-encoder-150m",
    license: "MIT (base model); fine-tuned on RAID (MIT) and MAGE (Apache-2.0)",
    trainedOn: "RAID train (11 generators, 12 adversarial attacks) and MAGE train (27 LLMs, 10 human domains)",
    base: ["/models/ettin150-v3/"],
    files: { model: "model_quantized.onnx", tokenizer: "tokenizer.json", config: "tokenizer_config.json" },
    parts: 7,
    localCache: "gpttrace/ettin150-raid-v3/onnx",
    bytes: 151386730, sha256: "8470f5d01a3afa90…",
    maxTokens: 512
  },
  // Optional neural audio-deepfake check. 117 MB, self-hosted in 6 parts
  // like the text model. Trained by eval/train/audio (Kaggle
  // gpttrace-audio-wavlm2); int8 on MatMul only (gpttrace-audio-requant) —
  // quantising the convolutional encoder too broke it.
  audio: {
    id: "gpttrace-wavlm-audio-v3-int8",
    name: "GPTTrace audio deepfake detector (WavLM-base-plus, int8)",
    source: "https://huggingface.co/microsoft/wavlm-base-plus",
    license: "MIT (base model); fine-tuned on ASVspoof 2019 LA (ODC-By), WaveFake (CC BY-SA 4.0), LJSpeech (public domain)",
    trainedOn: "ASVspoof 2019 LA train, WaveFake and LJSpeech, with MP3/Opus/AAC re-encoding",
    base: ["/models/wavlm-audio-v3/"],
    files: { model: "model_quantized.onnx" },
    parts: 6,
    localCache: "gpttrace/wavlm-audio-v3/onnx",
    bytes: 122445679
  }
};

export const mb = (b) => (b / 1048576).toFixed(0) + " MB";
