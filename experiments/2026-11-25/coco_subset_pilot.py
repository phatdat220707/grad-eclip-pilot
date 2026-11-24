"""Deterministic 8-image COCO Karpathy-val pilot; no benchmark claims."""
exec(compile((__import__('pathlib').Path(__file__).parent/'openvision_probe.py').read_text(),'openvision_probe.py','exec'))
import numpy as np, time
import clip
ov_model,ov_transform,ov_tokenizer=model,transform,tokenizer
out=ROOT/'coco-pilot'
out.mkdir(exist_ok=True)
source='https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json'
annotation=out/'coco_karpathy_val.json'
if not annotation.exists(): urllib.request.urlretrieve(source,annotation)
rows=json.loads(annotation.read_text())[:8]
manifest={'annotation_source':source,'annotation_sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),'selection':'first 8 entries in downloaded Karpathy validation annotation order; first caption per entry; no result-based selection','negative_rule':'caption from next selected image cyclically; source mismatch, not guaranteed semantically negative','images':[]}
for row in rows:
    path=out/Path(row['image']).name
    url='http://images.cocodataset.org/'+row['image']
    if not path.exists(): urllib.request.urlretrieve(url,path)
    Image.open(path).verify()
    manifest['images'].append({'image_id':int(path.stem.split('_')[-1]),'file':path.name,'source_url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'captions':row['caption']})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
captions=[r['caption'][0] for r in rows]
with torch.no_grad():
    ov_text=F.normalize(ov_model.encode_text(ov_tokenizer(captions)),dim=-1)
    ov_images=torch.cat([ov_transform(Image.open(out/r['file']).convert('RGB')).unsqueeze(0) for r in manifest['images']])
    ov_img=F.normalize(ov_model.encode_image(ov_images),dim=-1)
    ov_scores=(ov_img@ov_text.T).tolist()
clipmodel,preprocess=clip.load('ViT-B/16',device='cpu',download_root=str(PILOT_ROOT/'models'))
clipmodel.eval()
clip_inres=clipmodel.visual.input_resolution
clip_ksize=clipmodel.visual.conv1.kernel_size
notebook=json.loads((PILOT_ROOT/'external/Grad-Eclip/grad_eclip_image.ipynb').read_text(encoding='utf-8'))
cell4=''.join(notebook['cells'][4]['source']).replace('x = x.half()','x = x.to(clipmodel.visual.conv1.weight.dtype)')
exec(cell4,globals())
exec(''.join(notebook['cells'][9]['source']),globals())
with torch.no_grad():
    ct=F.normalize(clipmodel.encode_text(clip.tokenize(captions)),dim=-1)
record={'manifest':'manifest.json','openvision_cosine_matrix':ov_scores,'captions':captions,'clip_cosine_matrix':[],'maps':[],'limitations':'8-image convenience subset; source-mismatched captions may be semantically compatible; maps are official CLIP single-head Grad-ECLIP, not OpenVision adaptation'}
for i,row in enumerate(manifest['images']):
    xi=preprocess(Image.open(out/row['file']).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        original=(F.normalize(clipmodel.encode_image(xi),dim=-1)@ct.T)[0]
    record['clip_cosine_matrix'].append(original.tolist())
    outputs,_,vs,qs,ks,_,attn_outputs,map_size=clip_encode_dense(xi,n=1)
    modified=(F.normalize(outputs[:,0],dim=-1)@ct.T)[0]
    for kind,j in [('matched',i),('source_mismatched',(i+1)%len(rows))]:
        heat=grad_eclip(modified[j],qs,ks,vs,attn_outputs,map_size).detach().cpu().numpy()
        name=f"{row['image_id']}_{kind}.npy"
        np.save(out/name,heat)
        record['maps'].append({'image_id':row['image_id'],'caption_index':j,'condition':kind,'original_cosine':float(original[j]),'single_head_cosine':float(modified[j].detach()),'nonzero':int(np.count_nonzero(heat)),'file':name})
    (out/'results_partial.json').write_text(json.dumps(record,indent=2))
    print('COMPLETED IMAGE',i,row['image_id'],flush=True)
for name,matrix in [('clip',record['clip_cosine_matrix']),('openvision',ov_scores)]:
    a=np.array(matrix)
    record[name+'_summary']={'matched_cosine_mean':float(np.diag(a).mean()),'cyclic_mismatch_cosine_mean':float(np.mean([a[i,(i+1)%len(rows)] for i in range(len(rows))])),'matched_beats_cyclic_count':int(sum(a[i,i]>a[i,(i+1)%len(rows)] for i in range(len(rows)))),'image_to_text_top1_in_8':int(sum(a.argmax(1)==np.arange(len(rows))))}
(out/'results.json').write_text(json.dumps(record,indent=2))
print('COCO DONE',json.dumps({k:v for k,v in record.items() if k.endswith('_summary')},indent=2))
