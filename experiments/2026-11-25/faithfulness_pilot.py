import json, time, hashlib, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from torchvision.transforms.functional import gaussian_blur

parser = argparse.ArgumentParser()
parser.add_argument('--replacement', choices=['mean','blur'], default='mean')
parser.add_argument('--tie-seeds', type=int, default=0)
args = parser.parse_args()

root = Path(__file__).resolve().parents[2]
out = Path(__file__).parent / ('results-2026-11-25' if args.replacement == 'mean' and not args.tie_seeds else f'results-2026-11-25-{args.replacement}-ties{args.tie_seeds}')
out.mkdir(exist_ok=True)
torch.set_num_threads(4)
started = time.time()
model, preprocess = clip.load('ViT-B/16', device='cpu', download_root=str(root/'models'))
model.eval()
x = preprocess(Image.open(root/'external/Grad-Eclip/dog_and_car.png').convert('RGB')).unsqueeze(0)
baseline = torch.zeros_like(x) if args.replacement == 'mean' else gaussian_blur(x, [51,51], [10.,10.])
fractions = np.linspace(0, 1, 21)
counts = np.rint(fractions*196).astype(int)
record = {'protocol': {'score': 'original multi-head CLIP normalized cosine', 'map': 'saved single-head Grad-ECLIP last block', 'replacement': args.replacement, 'blur_kernel_sigma': [51,10] if args.replacement=='blur' else None, 'tie_seeds':list(range(args.tie_seeds)), 'patch_grid': [14,14], 'patch_pixels':16, 'counts': counts.tolist(), 'random_seeds':list(range(10)), 'auc': 'trapezoid over actual removed/inserted patch fraction; raw cosine'}, 'results': []}
(out/'protocol.json').write_text(json.dumps(record['protocol'],indent=2))
with torch.no_grad():
    text = F.normalize(model.encode_text(clip.tokenize(['dog','car'])),dim=-1)
    for pi,prompt in enumerate(['dog','car']):
        path = root/'results/2026-11-03'/f'{prompt}_heatmap_raw.npy'
        heat = np.load(path).reshape(-1)
        orders = {'salient_first':np.argsort(-heat,kind='stable'), 'least_first':np.argsort(heat,kind='stable')}
        orders.update({f'random_{seed}':np.random.default_rng(seed).permutation(196) for seed in range(10)})
        for seed in range(args.tie_seeds):
            perm = np.random.default_rng(seed).permutation(196)
            orders[f'salient_ties_{seed}'] = perm[np.argsort(-heat[perm],kind='stable')]
        item = {'prompt':prompt,'map_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'curves':{}}
        for name,order in orders.items():
            masks=[]
            for count in counts:
                mask=torch.ones(196)
                mask[order[:count].copy()]=0
                masks.append(mask.reshape(14,14).repeat_interleave(16,0).repeat_interleave(16,1))
            masks=torch.stack(masks)[:,None]
            values={}
            for mode,images in [('deletion',x*masks+baseline*(1-masks)),('insertion',x*(1-masks)+baseline*masks)]:
                scores=[]
                for batch in images.split(4):
                    scores.extend((F.normalize(model.encode_image(batch),dim=-1)@text[pi]).tolist())
                values[mode]={'scores':scores,'auc':float(np.trapezoid(scores,counts/196))}
            item['curves'][name]=values
            (out/f'{prompt}_partial.json').write_text(json.dumps(item,indent=2))
        record['results'].append(item)
        print(prompt, json.dumps({name:{mode:v['auc'] for mode,v in curves.items()} for name,curves in item['curves'].items()}),flush=True)
record['elapsed_seconds']=time.time()-started
record['torch']=torch.__version__
(out/'results.json').write_text(json.dumps(record,indent=2))
print('DONE',record['elapsed_seconds'],flush=True)
