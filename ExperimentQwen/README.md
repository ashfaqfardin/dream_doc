# ExperimentQwen E1

Mask-free Qwen baseline:

1. Convert every sketch into a photorealistic isolated object.
2. Generate an empty base room from a blank canvas.
3. Add the generated objects sequentially using scene + reference multi-image inputs.

```bash
python ExperimentQwen/e1_baseline.py \
  --sketch_dir KontextPipeline/sketch \
  --out_dir results/qwen_e1_baseline
```

Or use the shared experiment runner:

```bash
python ExperimentQwen/run_all.py
python ExperimentQwen/run_all.py --only e1
python ExperimentQwen/run_all.py --list
```

The default model is `Qwen/Qwen-Image-Edit-2509` with the bf16 LightX2V
8-step Lightning adapter. An A100 80 GB is recommended. The first run downloads
the base model and LoRA; later runs use the Hugging Face cache.
