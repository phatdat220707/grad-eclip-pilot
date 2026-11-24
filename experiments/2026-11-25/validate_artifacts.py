"""Validate archived numeric artifacts without downloading data or running models."""
import json, math
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, list):
        for item in value: finite(item)
    elif isinstance(value, dict):
        for item in value.values(): finite(item)

json_files=list(root.rglob('*.json'))
json_files=[p for p in json_files if not any(part in {'probe-deps','probe-cache','openvision-tiny'} for part in p.relative_to(root).parts)]
for path in json_files: finite(json.loads(path.read_text(encoding='utf-8')))
arrays=list(root.rglob('*.npy'))
for path in arrays: assert np.isfinite(np.load(path,allow_pickle=False)).all(), str(path)
coco=json.loads((root/'coco-pilot/results.json').read_text())
manifest=json.loads((root/'coco-pilot/manifest.json').read_text())
assert len(manifest['images'])==8 and len(coco['maps'])==16
assert len({r['image_id'] for r in manifest['images']})==8
for item in coco['maps']: assert (root/'coco-pilot'/item['file']).is_file()
for name in ['clip','openvision']:
    a=np.asarray(coco[name+'_cosine_matrix'])
    assert a.shape==(8,8)
    summary=coco[name+'_summary']
    assert abs(float(np.diag(a).mean())-summary['matched_cosine_mean'])<1e-9
    assert int(sum(a.argmax(1)==np.arange(8)))==summary['image_to_text_top1_in_8']
for folder in ['openvision-adapter-pilot','openvision-gradient-baselines']:
    data=json.loads((root/folder/'results.json').read_text())
    assert len(data['records'])==8
    for name,summary in data['summary'].items():
        for mode in ['deletion','insertion']:
            observed=np.mean([r['methods'][name][mode]['auc'] for r in data['records']])
            assert abs(float(observed)-summary[mode])<1e-9
    for record in data['records']:
        for curves in record['methods'].values():
            assert abs(curves['deletion']['scores'][0]-record['score'])<1e-6
            assert abs(curves['insertion']['scores'][-1]-record['score'])<1e-6
print(f'PASS: {len(json_files)} JSON files, {len(arrays)} finite arrays; COCO counts, summaries and endpoints consistent.')
