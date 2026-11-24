"""Check gradients at final tokens for actual average-pooling model."""
exec(compile((__import__('pathlib').Path(__file__).parent/'openvision_probe.py').read_text(),'openvision_probe.py','exec'))
manifest=json.loads((ROOT/'coco-pilot/manifest.json').read_text())
captions=[r['captions'][0] for r in manifest['images']]
with torch.no_grad():tf=F.normalize(model.encode_text(tokenizer(captions)),dim=-1)
cache={}
def capture_tokens(module,inputs,output):cache['tokens']=output
h=model.visual.transformer.register_forward_hook(capture_tokens)
records=[]
for i,row in enumerate(manifest['images']):
    xi=transform(Image.open(ROOT/'coco-pilot'/row['file']).convert('RGB')).unsqueeze(0)
    score=(F.normalize(model.encode_image(xi),dim=-1)@tf[i])[0]
    t=cache['tokens'];g=torch.autograd.grad(score,t)[0]
    records.append({'image_id':row['image_id'],'class_token_gradient_l1':float(g[:,0].abs().sum()),'max_patch_gradient_difference':float((g[:,1:]-g[:,1:2]).abs().max()),'patch_gradient_l1':float(g[:,1:].abs().sum())})
h.remove()
(ROOT/'openvision-gradient-baselines/pooling_geometry.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
