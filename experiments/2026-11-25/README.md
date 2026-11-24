# Research checkpoint — 25 November 2026

Start with [progress-report.txt](progress-report.txt). This directory contains original experiment outputs and runnable copies of the session scripts. Copies have repository-relative source/model paths; algorithm settings and recorded outputs are unchanged. These experiments were assisted by an AI coding agent; the researcher should review the scripts, methods and claims before presenting them.

## Findings and limits

- **COCO pilot:** eight fixed images, first caption per image, plus cyclic next-image captions as comparisons. Both original models score the source caption above the cyclic comparison on 8/8 and select it top-1 among eight on 7/8. This is not benchmark retrieval performance, and other-source captions are not guaranteed semantically false.
- **OpenVision feasibility:** original Tiny/16-224 weights strictly load; image/text forward and backward work. Explicit final attention preserves scores within 2.98e-8 on the tested cases. This does not verify native JAX parity.
- **Negative candidate result:** a multi-query, three-head spatial-weight extension inspired by Grad-ECLIP performs worse than random on mean deletion/insertion AUC on this subset. It is not a validated Grad-ECLIP adaptation.
- **Gradient baseline:** final-token gradient-times-activation outperforms random means in these checks under mean and blur replacement. This is a local-feature baseline, not a new method or general faithfulness result.
- Final Transformer class-token output gradient is zero under patch average pooling; earlier class-token influence is not excluded.

| OpenVision method / replacement | Deletion AUC ↓ | Insertion AUC ↑ |
| --- | ---: | ---: |
| Candidate / mean | 0.227218 | 0.220368 |
| Token gradient × activation / mean | 0.204239 | 0.241985 |
| Random pooled / mean | 0.225499 | 0.228198 |
| Candidate / blur | 0.227023 | 0.221088 |
| Token gradient × activation / blur | 0.204396 | 0.239217 |
| Random pooled / blur | 0.220905 | 0.224259 |

Raw cosine AUCs depend on the selected image-caption pairs and replacement scheme; they are not directly comparable with the paper's ImageNet accuracy AUCs. Eight convenience-selected images and three random orders are not enough for statistical or generalization claims. No model training was performed.

## Evidence

| Directory/file | Contents |
| --- | --- |
| `coco-pilot/` | Eight-image manifest with IDs, source URLs, captions and SHA256 hashes; 8×8 original-model score matrices; 16 CLIP single-head Grad-ECLIP raw maps; retrieval mismatch; comparison figure |
| `openvision-probe-results.json` | Pinned checkpoint/config, strict loading and gradient checks |
| `model-config/open_clip_config.json` | Small copy of the upstream pinned config for inspection; loading script retrieves the same revision |
| `openvision-attention-results/` | Score-preserving QKV instrumentation and experimental softmax-value maps |
| `openvision-adapter-pilot/` | Exact candidate formula, protocols, raw curves/maps, negative result |
| `openvision-gradient-baselines/` | Three gradient baselines, paired mean/blur curves and pooling geometry checks |
| `results-2026-11-25*/` | Earlier one-image CLIP perturbation protocols and curves |

Generated figures include COCO-derived imagery. Original image rights remain with their respective owners; source image URLs and attribution lookup IDs are recorded in `coco-pilot/manifest.json`. Original COCO images and full annotations are not committed. Model weights, upstream clones, tokenizer downloads, dependencies and caches are not committed.

## Reproduce

Tested in Windows CPU Python 3.14.6, PyTorch 2.15.0, torchvision 0.29.2, NumPy 2.5.5, OpenCLIP 4.2.2. The original CLIP-only perturbation run used the earlier baseline environment recorded in its JSON. Use a separate environment if preserving an existing setup matters.

From the repository root, first follow the top-level setup instructions (`.\setup_windows.ps1`) to install OpenAI CLIP, acquire the pinned upstream sources, weights and baseline dependencies. Then install the probe packages into that environment:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r .\experiments\2026-11-25\requirements-probe.txt
& .\.venv\Scripts\python.exe .\experiments\2026-11-25\openvision_probe.py
& .\.venv\Scripts\python.exe .\experiments\2026-11-25\coco_subset_pilot.py
& .\.venv\Scripts\python.exe .\experiments\2026-11-25\coco_pilot_figure.py
& .\.venv\Scripts\python.exe .\experiments\2026-11-25\openvision_attention_probe.py
& .\.venv\Scripts\python.exe .\experiments\2026-11-25\openvision_pooling_adapter.py
& .\.venv\Scripts\python.exe .\experiments\2026-11-25\openvision_gradient_baselines.py
& .\.venv\Scripts\python.exe .\experiments\2026-11-25\openvision_pooling_geometry.py
```

`openvision_probe.py` is executed by several later scripts, so they also repeat its loading/sanity checks. Reruns replace the corresponding generated results. The Tiny checkpoint and tokenizer are fetched from revision `7779317ba50238043f5d3ee0751fd95d7d13aa7e`. The runner defaults to this repository's `external/` and `models/`; `RMS_PILOT_ROOT` can point to another existing baseline checkout. `probe-deps/`, if present alongside the scripts, is an optional local dependency folder from the original session.

`RMS_OPENVISION_CACHE` optionally points to an existing directory containing the downloaded Tiny checkpoint/tokenizer files. The default is an ignored `openvision-tiny/` folder alongside these scripts. Artifact consistency can be checked with `python experiments/2026-11-25/validate_artifacts.py`; this does not rerun model inference.

The COCO runner downloads Salesforce's Karpathy-val annotation and the first eight images. COCO HTTP image URLs were used because HTTPS failed certificate hostname validation in the recorded session. File hashes record identity, not secure authentication of that HTTP source. Network access is required for first-time downloads. Verify downloads before reuse if source content changes.

To repeat the earlier one-image perturbation sweeps, first generate or retain `results/2026-11-03/*_heatmap_raw.npy`, then run:

```powershell
& .\.venv\Scripts\python.exe .\experiments\2026-11-25\faithfulness_pilot.py
& .\.venv\Scripts\python.exe .\experiments\2026-11-25\faithfulness_pilot.py --replacement blur --tie-seeds 3
```

## Sources

- [Grad-ECLIP paper v3](https://arxiv.org/html/2502.18816v3) and [official implementation](https://github.com/Cyang-Zhao/Grad-Eclip), commit `e370e6cb194faf2020f5d1ed268f9d57e91a38e6`.
- [Original OpenVision checkpoint and config](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch16-224/tree/7779317ba50238043f5d3ee0751fd95d7d13aa7e); [official OpenVision source](https://github.com/UCSC-VLAA/OpenVision), inspected/cloned commit `c3f7d96d89c0bde6cb9ae9dc707e68a753cb7d89`. Runtime uses OpenCLIP 4.2.2 rather than the bundled fork.
- [Salesforce Karpathy-val annotations](https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json) and [COCO](https://cocodataset.org/).
