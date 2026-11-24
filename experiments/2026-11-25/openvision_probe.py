import sys, json, hashlib, urllib.request, os
from pathlib import Path
ROOT=Path(__file__).parent
PILOT_ROOT=Path(os.environ.get('RMS_PILOT_ROOT', str(ROOT.parents[1])))
sys.path.insert(0,str(ROOT/'probe-deps'))
os.environ['HF_HOME']=str(ROOT/'probe-cache')
import torch
torch.set_num_threads(4)
folder=Path(os.environ.get('RMS_OPENVISION_CACHE', str(ROOT/'openvision-tiny')))
folder.mkdir(exist_ok=True)
revision='7779317ba50238043f5d3ee0751fd95d7d13aa7e'
for name in ['open_clip_pytorch_model.bin','open_clip_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']:
    path=folder/name
    if not path.exists():
        urllib.request.urlretrieve(f'https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch16-224/resolve/{revision}/{name}',path)
state=torch.load(folder/'open_clip_pytorch_model.bin',map_location='cpu',weights_only=True)
print('WEIGHT KEYS',list(state)[:10],flush=True)
import open_clip
from PIL import Image
cfg=json.loads((folder/'open_clip_config.json').read_text())
model=open_clip.model.CLIP(**cfg['model_cfg'])
loaded=model.load_state_dict(state,strict=True)
model.eval()
tokenizer=open_clip.tokenizer.HFTokenizer(str(folder),context_length=80,strip_sep_token=True)
transform=open_clip.transform.image_transform(image_size=224,is_train=False,**cfg['preprocess_cfg'])
image_path=PILOT_ROOT/'external/Grad-Eclip/dog_and_car.png'
x=transform(Image.open(image_path).convert('RGB')).unsqueeze(0).requires_grad_(True)
tokens=tokenizer(['dog','car'])
captured={}
def capture(module, inputs, output):
    captured['attention_output']=output[0] if isinstance(output,tuple) else output
hook=model.visual.transformer.resblocks[-1].attn.register_forward_hook(capture)
import torch.nn.functional as F
image=F.normalize(model.encode_image(x),dim=-1)
text=F.normalize(model.encode_text(tokens),dim=-1)
scores=(image@text.T)[0]
grads=[]
attention_grads=[]
for score in scores:
    g=torch.autograd.grad(score,x,retain_graph=True)[0]
    grads.append({'finite':bool(torch.isfinite(g).all()),'l1':float(g.abs().sum())})
    ag=torch.autograd.grad(score,captured['attention_output'],retain_graph=True)[0]
    attention_grads.append({'shape':list(ag.shape),'finite':bool(torch.isfinite(ag).all()),'l1':float(ag.abs().sum())})
hook.remove()
result={'revision':revision,'weight_sha256':hashlib.sha256((folder/'open_clip_pytorch_model.bin').read_bytes()).hexdigest(),'strict_load':str(loaded),'open_clip_version':open_clip.__version__,'scores':scores.detach().tolist(),'image_shape':list(image.shape),'text_shape':list(text.shape),'input_gradients':grads,'config':cfg}
with torch.no_grad():
    unhooked=F.normalize(model.encode_image(x.detach()),dim=-1)@text.detach().T
result['attention_gradients']=attention_grads
result['hook_score_difference_max']=float((unhooked[0]-scores.detach()).abs().max())
(ROOT/'openvision-probe-results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
