"""Score-preserving final-block attention decomposition; experimental attribution, not Grad-ECLIP reproduction."""
exec(compile((__import__('pathlib').Path(__file__).parent/'openvision_probe.py').read_text(), 'openvision_probe.py','exec'))
import types, numpy as np
out=ROOT/'openvision-attention-results'
out.mkdir(exist_ok=True)
attn=model.visual.transformer.resblocks[-1].attn
original_forward=attn.forward
cache={}
def explicit(self, query, k_x=None, v_x=None, attn_mask=None):
    key=k_x if k_x is not None else query
    value=v_x if v_x is not None else query
    assert attn_mask is None
    assert query is key and key is value
    b,n,d=query.shape
    q,k,v=F.linear(query,self.in_proj_weight,self.in_proj_bias).chunk(3,dim=-1)
    h=self.num_heads
    q,k,v=[t.reshape(b,n,h,d//h).transpose(1,2) for t in (q,k,v)]
    a=((q@k.transpose(-2,-1))*(d//h)**-0.5).softmax(dim=-1)
    z=a@v
    output=F.linear(z.transpose(1,2).reshape(b,n,d),self.out_proj.weight,self.out_proj.bias)
    cache.update(q=q,k=k,v=v,a=a,z=z)
    return output
attn.forward=types.MethodType(explicit,attn)
embedding=F.normalize(model.encode_image(x),dim=-1)
modified=(embedding@text.T)[0]
diff=float((modified.detach()-scores.detach()).abs().max())
assert diff<1e-6, diff
items=[]
for prompt,score in zip(['dog','car'],modified):
    g=torch.autograd.grad(score,cache['z'],retain_graph=True)[0]
    # Contributions grouped by source token j: sum over receiving query i,
    # head h and channel c of d(score)/d(z_ihc) * A_hij * V_hjc.
    signed=torch.einsum('bhic,bhij,bhjc->bj',g,cache['a'],cache['v'])[0]
    heat=signed[1:].relu().reshape(14,14).detach().numpy()
    np.save(out/f'{prompt}_experimental_map.npy',heat)
    np.save(out/f'{prompt}_signed_tokens.npy',signed.detach().numpy())
    scale=heat/(heat.max()+1e-12)
    from PIL import ImageOps
    base=Image.open(image_path).convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
    alpha=np.asarray(Image.fromarray((scale*255).astype('uint8')).resize((224,224),Image.Resampling.BILINEAR))/255.*.6
    rgb=np.asarray(base).astype(float)
    rgb=rgb*(1-alpha[...,None])+np.array([255,0,0])*alpha[...,None]
    Image.fromarray(rgb.astype('uint8')).save(out/f'{prompt}_experimental_overlay.png')
    items.append({'prompt':prompt,'score':float(score.detach()),'gradient_finite':bool(torch.isfinite(g).all()),'nonzero_patches':int(np.count_nonzero(heat)),'map_max':float(heat.max())})
attn.forward=original_forward
record={'method':'experimental gradient-weighted attention/value source-token decomposition; NOT official Grad-ECLIP','formula':'ReLU(sum_h,i,c grad_z[h,i,c] * attention[h,i,j] * value[h,j,c]); discard source class token j=0','attention':'original 3 heads; final block only; gradients through full downstream average pooling/MLP/LayerNorm','score_difference_max':diff,'qkv_shape':list(cache['q'].shape),'results':items,'limitations':'local first-order intermediate attribution; not a complete input attribution or demonstrated faithfulness; no loosened spatial weighting or official class-token formula'}
(out/'results.json').write_text(json.dumps(record,indent=2))
print('ATTENTION PROBE',json.dumps(record,indent=2))
