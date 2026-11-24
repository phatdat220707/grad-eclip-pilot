"""Experimental multi-query/multi-head Grad-ECLIP-inspired adapter and COCO pilot.
Not an official Grad-ECLIP implementation; see saved protocol for differences.
"""
exec(compile((__import__('pathlib').Path(__file__).parent/'openvision_probe.py').read_text(),'openvision_probe.py','exec'))
import numpy as np, types, time
out=ROOT/'openvision-adapter-pilot'; out.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'coco-pilot/manifest.json').read_text())
protocol={'candidate':'pooling-aware multi-query multi-head Grad-ECLIP-inspired variant','formula':'H_j=ReLU(sum_h,i,c grad(score,z_hic)*lambda_hij*v_hjc), j patch tokens; lambda=minmax cosine(q_hi,k_hj) over patch keys j per query/head','differences_from_official':'all receiving queries, original 3 heads; channel gradient through downstream pooling/MLP; ReLU after query/head aggregation; epsilon clamp for constant QK rows; not validated as equivalent to paper','controls':['ordinary softmax contribution','lambda=1 ablation','random seeds 0,1,2'],'evaluation':'original OpenVision cosine; 11 rounded patch counts spanning 0..196; zero after model normalization (ImageNet mean RGB); trapezoidal raw cosine AUC','selection':manifest['selection'],'finite_difference':'scale last-block pre-output-projection attention context by 1+-0.001; compare central score difference to sum(grad_z*z)'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
captions=[r['captions'][0] for r in manifest['images']]
with torch.no_grad(): text_features=F.normalize(model.encode_text(tokenizer(captions)),dim=-1)
attn=model.visual.transformer.resblocks[-1].attn
original_forward=attn.forward
cache={}; scale=1.0
def explicit(self,query,k_x=None,v_x=None,attn_mask=None):
    assert attn_mask is None and k_x is query and v_x is query
    assert self.logit_scale is None and self.head_scale is None
    assert isinstance(self.ln_q,torch.nn.Identity) and isinstance(self.ln_k,torch.nn.Identity) and isinstance(self.ln_inner,torch.nn.Identity)
    b,n,d=query.shape; h=self.num_heads
    q,k,v=[t.reshape(b,n,h,d//h).transpose(1,2) for t in F.linear(query,self.in_proj_weight,self.in_proj_bias).chunk(3,-1)]
    a=((q@k.transpose(-1,-2))*self.scale).softmax(-1)
    z=a@v
    cache.update(q=q,k=k,v=v,a=a,z=z)
    return self.out_drop(F.linear((z*scale).transpose(1,2).reshape(b,n,d),self.out_proj.weight,self.out_proj.bias))
records=[]; counts=np.rint(np.linspace(0,196,11)).astype(int)
start=time.time()
for index,row in enumerate(manifest['images']):
    xi=transform(Image.open(ROOT/'coco-pilot'/row['file']).convert('RGB')).unsqueeze(0).requires_grad_(True)
    with torch.no_grad(): original_score=(F.normalize(model.encode_image(xi),dim=-1)@text_features[index])[0]
    attn.forward=types.MethodType(explicit,attn)
    score=(F.normalize(model.encode_image(xi),dim=-1)@text_features[index])[0]
    difference=float(abs(score.detach()-original_score)); assert difference<1e-6
    q,k,v,a,z=[cache[t] for t in ['q','k','v','a','z']]
    grad=torch.autograd.grad(score,z,retain_graph=True)[0]
    cos=F.normalize(q,dim=-1)@F.normalize(k[:,:,1:],dim=-1).transpose(-1,-2)
    low=cos.amin(-1,keepdim=True); high=cos.amax(-1,keepdim=True)
    lam=(cos-low)/(high-low).clamp_min(1e-12)
    maps={
      'pooling_candidate':torch.einsum('bhic,bhij,bhjc->bj',grad,lam,v[:,:,1:])[0].relu(),
      'softmax_control':torch.einsum('bhic,bhij,bhjc->bj',grad,a[:,:,:,1:],v[:,:,1:])[0].relu(),
      'no_spatial':torch.einsum('bhic,bhjc->bj',grad,v[:,:,1:])[0].relu()}
    predicted=float((grad*z).sum().detach())
    fd_scores=[]
    with torch.no_grad():
        for scale in [1.001,.999]:
            fd_scores.append(float((F.normalize(model.encode_image(xi),dim=-1)@text_features[index])[0]))
    scale=1.; attn.forward=original_forward
    fd=(fd_scores[0]-fd_scores[1])/.002
    item={'image_id':row['image_id'],'caption':captions[index],'score':float(original_score),'forward_difference':difference,'gradient_directional':predicted,'finite_difference':fd,'absolute_derivative_error':abs(predicted-fd),'methods':{}}
    orders={}
    for name,heat in maps.items():
        array=heat.detach().numpy(); assert np.isfinite(array).all()
        np.save(out/f"{row['image_id']}_{name}.npy",array.reshape(14,14))
        orders[name]=np.argsort(-array,kind='stable')
    orders.update({f'random_{seed}':np.random.default_rng(seed).permutation(196) for seed in range(3)})
    with torch.no_grad():
        for name,order in orders.items():
            masks=[]
            for count in counts:
                mask=torch.ones(196);mask[order[:count].copy()]=0
                masks.append(mask.reshape(14,14).repeat_interleave(16,0).repeat_interleave(16,1))
            masks=torch.stack(masks)[:,None]
            values={}
            for mode,imgs in [('deletion',xi.detach()*masks),('insertion',xi.detach()*(1-masks))]:
                scores_list=[]
                for batch in imgs.split(4):scores_list.extend((F.normalize(model.encode_image(batch),dim=-1)@text_features[index]).tolist())
                values[mode]={'scores':scores_list,'auc':float(np.trapezoid(scores_list,counts/196))}
            item['methods'][name]=values
    records.append(item)
    (out/'results_partial.json').write_text(json.dumps(records,indent=2))
    print('ADAPTER IMAGE',index,row['image_id'],'parity',difference,'derivative_error',abs(predicted-fd),flush=True)
summary={name:{mode:float(np.mean([r['methods'][name][mode]['auc'] for r in records])) for mode in ['deletion','insertion']} for name in orders}
result={'protocol':protocol,'records':records,'summary':summary,'elapsed_seconds':time.time()-start,'limitations':'candidate implementation only; eight nonrandom images; one perturbation baseline; no significance or superiority claim; native JAX parity not established'}
(out/'results.json').write_text(json.dumps(result,indent=2))
print('ADAPTER SUMMARY',json.dumps(summary,indent=2),flush=True)
