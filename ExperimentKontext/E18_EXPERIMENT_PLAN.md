# E18 — FLUX.1-Kontext Architecture Lab

E18 is a mechanistic study, not an insertion pipeline. It needs only a scene,
reference, and edit prompt—no target mask.

| RQ | Question | Experiment |
|---|---|---|
| 01 | What is the topology and parameter allocation? | module/config inventory |
| 02 | How are text, target and context tokens packed? | runtime shape trace |
| 03 | Which layers route context to target tokens? | layer/source attention |
| 04 | When is each source used during denoising? | timestep/source attention |
| 05 | Are heads source-specialized? | head selectivity |
| 06 | Is routing concentrated or diffuse? | normalized attention entropy |
| 07 | Does reference content/structure matter? | correct/blank/shuffled reference |
| 08 | Does context order matter? | scene-ref versus ref-scene |
| 09 | Does duplication bias the model? | scene-scene/ref-ref controls |
| 10 | What does temporal 3D RoPE contribute? | IDs (1,2), (1,1), (2,1) |
| 11 | Is each source causally necessary? | block text/scene/reference attention |
| 12 | Where should we modify the model? | layer-band causal attention ablation |

Run `inventory,trace` first, then counterfactual and causal groups. Use at least
five seeds before making paper claims. Attention is correlational; RQ11–12 are
the causal tests.

```bash
python ExperimentKontext/e18_kontext_architecture_lab.py \
 --scene scene.png --reference bicycle.png --object_name bicycle \
 --rqs inventory,trace,controls,rope,causal --out_dir results/e18
```

Outputs include raw CSV/JSON, token layouts, layer/timestep/head plots, matched
output grids, causal effect sizes, and a reproducibility manifest.
