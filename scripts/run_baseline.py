from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import __version__ as torchvision_version

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external" / "Grad-Eclip"
OUT = ROOT / "results" / "2026-11-03"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)

notebook = json.loads((REPO / "grad_eclip_image.ipynb").read_text(encoding="utf-8"))
cell4 = "".join(notebook["cells"][4]["source"])
cell9 = "".join(notebook["cells"][9]["source"])
# The official notebook forces float16/CUDA. Keep its method, but use model dtype for CPU.
cell4 = cell4.replace("x = x.half()", "x = x.to(clipmodel.visual.conv1.weight.dtype)")
exec(cell4, globals())
exec(cell9, globals())

device = "cpu"
clipmodel, preprocess = clip.load("ViT-B/16", device=device, download_root=str(MODELS))
clipmodel.eval()
clip_inres = clipmodel.visual.input_resolution
clip_ksize = clipmodel.visual.conv1.kernel_size

image_path = REPO / "dog_and_car.png"
image = Image.open(image_path).convert("RGB")
prompts = ["dog", "car"]
image_tensor = preprocess(image).unsqueeze(0)
text_tokens = clip.tokenize(prompts)
with torch.no_grad():
    text_embedding = F.normalize(clipmodel.encode_text(text_tokens), dim=-1)
    original_embedding = F.normalize(clipmodel.encode_image(image_tensor), dim=-1)
    original_scores = (original_embedding @ text_embedding.T)[0].tolist()

outputs, last_feat, vs, qs, ks, attns, attn_outputs, map_size = clip_encode_dense(image_tensor, n=1)
modified_embedding = F.normalize(outputs[:, 0], dim=-1)
modified_scores_tensor = (modified_embedding @ text_embedding.T)[0]
modified_scores = modified_scores_tensor.detach().tolist()

# Control: preserve the original number of attention heads through the same notebook code path.
multi_cell4 = cell4.replace("def clip_encode_dense(x,n):", "def clip_encode_dense_multi(x,n):").replace(
    "attention_layer(q, k, v, 1)", "attention_layer(q, k, v, vision_heads)"
)
exec(multi_cell4, globals())
with torch.no_grad():
    multi_outputs, _, _, _, _, _, _, _ = clip_encode_dense_multi(image_tensor, n=1)
    multi_embedding = F.normalize(multi_outputs[:, 0], dim=-1)
    multi_scores = (multi_embedding @ text_embedding.T)[0].tolist()
display_image = ImageOps.fit(image, (int(clip_inres), int(clip_inres)), method=Image.Resampling.BICUBIC)
base = np.asarray(display_image).astype(np.float32)
results = []
for prompt, original, modified, multi, scalar in zip(prompts, original_scores, modified_scores, multi_scores, modified_scores_tensor):
    heatmap = grad_eclip(scalar, qs, ks, vs, attn_outputs, map_size).detach().cpu().float().numpy()
    np.save(OUT / f"{prompt}_heatmap_raw.npy", heatmap)
    raw_min, raw_max = float(heatmap.min()), float(heatmap.max())
    scaled = (heatmap - raw_min) / (raw_max - raw_min) if raw_max > raw_min else np.zeros_like(heatmap)
    large = Image.fromarray(np.uint8(np.clip(scaled * 255, 0, 255)), mode="L").resize(
        display_image.size, resample=Image.Resampling.BILINEAR
    )
    alpha = 0.6 * (np.asarray(large).astype(np.float32) / 255.0)[..., None]
    red = np.zeros_like(base)
    red[..., 0] = 255
    overlay = np.uint8(np.clip(base * (1 - alpha) + red * alpha, 0, 255))
    output_name = f"{prompt}_heatmap.png"
    Image.fromarray(overlay).save(OUT / output_name)
    results.append({
        "prompt": prompt,
        "original_cosine": float(original),
        "single_head_cosine": float(modified),
        "multi_head_control_cosine": float(multi),
        "multi_head_vs_original_difference": float(multi - original),
        "score_difference": float(modified - original),
        "heatmap_min": raw_min,
        "heatmap_max": raw_max,
        "heatmap_nonzero_patches": int(np.count_nonzero(heatmap)),
        "heatmap_patches": int(heatmap.size),
        "heatmap_file": output_name,
        "heatmap_raw_file": f"{prompt}_heatmap_raw.npy",
    })

record = {
    "source_repo": "https://github.com/Cyang-Zhao/Grad-Eclip",
    "source_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
    "source_notebook": "grad_eclip_image.ipynb",
    "clip_commit": subprocess.check_output(["git", "-C", str(ROOT / "external" / "CLIP"), "rev-parse", "HEAD"], text=True).strip(),
    "method_code": "official notebook cells 4 and 9; x.half() changed to model dtype for CPU",
    "model": "OpenAI CLIP ViT-B/16",
    "device": device,
    "image": "dog_and_car.png",
    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
    "torch": torch.__version__,
    "torchvision": torchvision_version,
    "numpy": np.__version__,
    "map_size": list(map_size),
    "results": results,
}
(OUT / "results.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
