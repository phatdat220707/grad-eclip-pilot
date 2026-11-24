"""Fixed gradient baselines, paired mean/blur perturbation evaluation on existing COCO subset."""
exec(compile((__import__('pathlib').Path(__file__).parent/'openvision_probe.py').read_text(),'openvision_probe.py','exec'))
import numpy as np,time
from torchvision.transforms.functional import gaussian_blur
out=ROOT/'openvision-gradient-baselines';out.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'coco-pilot/manifest.json').read_text())
protocol={'selection':manifest['selection'],'token_grad_activation':'ReLU(sum_c grad(score, final_transformer_token[j,c])*token[j,c]), j>0; includes residual and MLP output before pooling','input_grad_activation':'ReLU(sum over channels,pixels of input * grad(score,input) within each 16x16 patch); input normalized by model mean/std','input_abs_gradient':'sum absolute input gradient over channels and pixels within each patch','candidate':'previous saved multi-query spatial weighting; no formula changes','replacements':{'mean':'zero normalized image','blur':'torchvision Gaussian kernel51 sigma10 applied to normalized image; reflected padding'},'counts':np.rint(np.linspace(0,196,11)).astype(int).tolist(),'random_seeds':[0,1,2],'target':'original OpenVision source-caption normalized cosine; raw trapezoidal AUC','purpose':'paired diagnostic, no tuning/benchmark claim'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
captions=[r['captions'][0] for r in manifest['images']]
with torch.no_grad():tf=F.normalize(model.encode_text(tokenizer(captions)),dim=-1)
cache={}
def hook(module,inputs,output):cache['tokens']=output
handle=model.visual.transformer.register_forward_hook(hook)
records=[];start=time.time()
for i,row in enumerate(manifest['images']):
    xi=transform(Image.open(ROOT/'coco-pilot'/row['file']).convert('RGB')).unsqueeze(0).requires_grad_(True)
    score=(F.normalize(model.encode_image(xi),dim=-1)@tf[i])[0]
    tokens=cache['tokens']
    gt,gx=torch.autograd.grad(score,[tokens,xi])
    token_heat=(gt*tokens).sum(-1)[0,1:].relu().reshape(14,14)
    def patches(t):return t.reshape(14,16,14,16).sum((1,3))
    maps={'token_grad_activation':token_heat,'input_grad_activation':patches((gx*xi).sum(1)[0]).relu(),'input_abs_gradient':patches(gx.abs().sum(1)[0])}
    item={'image_id':row['image_id'],'score':float(score.detach()),'gradient_finite':bool(torch.isfinite(gt).all() and torch.isfinite(gx).all()),'methods':{},'map_stats':{}}
    orders={}
    for name,heat in maps.items():
        arr=heat.detach().numpy();assert np.isfinite(arr).all()
        np.save(out/f"{row['image_id']}_{name}.npy",arr)
        item['map_stats'][name]={'nonzero':int(np.count_nonzero(arr)),'max':float(arr.max())}
        orders[name]=np.argsort(-arr.reshape(-1),kind='stable')
    candidate=np.load(ROOT/'openvision-adapter-pilot'/f"{row['image_id']}_pooling_candidate.npy")
    orders['pooling_candidate']=np.argsort(-candidate.reshape(-1),kind='stable')
    orders.update({f'random_{s}':np.random.default_rng(s).permutation(196) for s in range(3)})
    with torch.no_grad():
        for replacement,base in [('mean',torch.zeros_like(xi)),('blur',gaussian_blur(xi,[51,51],[10.,10.]))]:
            for name,order in orders.items():
                masks=[]
                for count in protocol['counts']:
                    mask=torch.ones(196);mask[order[:count].copy()]=0
                    masks.append(mask.reshape(14,14).repeat_interleave(16,0).repeat_interleave(16,1))
                masks=torch.stack(masks)[:,None];values={}
                for mode,imgs in [('deletion',xi*masks+base*(1-masks)),('insertion',xi*(1-masks)+base*masks)]:
                    vals=[]
                    for batch in imgs.split(4):vals.extend((F.normalize(model.encode_image(batch),dim=-1)@tf[i]).tolist())
                    assert abs(vals[0 if mode=='deletion' else -1]-float(score))<1e-6
                    values[mode]={'scores':vals,'auc':float(np.trapezoid(vals,np.array(protocol['counts'])/196))}
                item['methods'][replacement+'_'+name]=values
    records.append(item)
    (out/'partial.json').write_text(json.dumps(records,indent=2))
    print('BASELINES',i,row['image_id'],flush=True)
handle.remove()
summary={name:{mode:float(np.mean([r['methods'][name][mode]['auc'] for r in records])) for mode in ['deletion','insertion']} for name in records[0]['methods']}
result={'protocol':protocol,'records':records,'summary':summary,'elapsed_seconds':time.time()-start,'limitations':'8 fixed convenience images; only source captions; 3 random seeds; raw scores and model-specific perturbations; descriptive diagnostic only'}
(out/'results.json').write_text(json.dumps(result,indent=2))
print('SUMMARY',json.dumps(summary,indent=2),flush=True)
