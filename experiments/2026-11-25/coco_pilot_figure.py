from pathlib import Path
import json, numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
root=Path(__file__).parent/'coco-pilot'
r=json.loads((root/'results.json').read_text())
m=json.loads((root/'manifest.json').read_text())
def load_font(size):
    for name in ['arial.ttf','DejaVuSans.ttf']:
        try: return ImageFont.truetype(name,size)
        except OSError: pass
    return ImageFont.load_default(size=size)
font=load_font(16)
small=load_font(14)
canvas=Image.new('RGB',(1120,8*260+75),'white')
d=ImageDraw.Draw(canvas)
d.text((10,10),'COCO 8-image pilot: CLIP single-head Grad-ECLIP maps (separately normalized)',font=font,fill='black')
d.text((10,38),'Input / matched caption map / next-image caption map. Scores below are original multi-head CLIP.',font=small,fill='black')
for i,row in enumerate(m['images']):
    y=75+i*260
    base=ImageOps.fit(Image.open(root/row['file']).convert('RGB'),(224,224),method=Image.Resampling.BICUBIC)
    canvas.paste(base,(10,y))
    for k,condition in enumerate(['matched','source_mismatched']):
        heat=np.load(root/f"{row['image_id']}_{condition}.npy")
        scaled=(heat-heat.min())/(heat.max()-heat.min()+1e-12)
        alpha=np.asarray(Image.fromarray((scaled*255).astype('uint8')).resize((224,224),Image.Resampling.BILINEAR))/255*.6
        overlay=np.asarray(base)*(1-alpha[...,None])+np.array([255,0,0])*alpha[...,None]
        canvas.paste(Image.fromarray(overlay.astype('uint8')),(244+234*k,y))
    lines=[f"COCO ID {row['image_id']}",'Matched: '+r['captions'][i],'Other source: '+r['captions'][(i+1)%8],f"Original cosine: {r['clip_cosine_matrix'][i][i]:.4f} vs {r['clip_cosine_matrix'][i][(i+1)%8]:.4f}"]
    yy=y
    import textwrap
    for line in lines:
        for part in textwrap.wrap(line,width=48):
            d.text((722,yy),part,font=small,fill='black'); yy+=20
    d.text((10,y+230),'Input',font=small,fill='black')
    d.text((244,y+230),'Matched',font=small,fill='black')
    d.text((478,y+230),'Other-source caption',font=small,fill='black')
canvas.save(root/'comparison.png')
failures={}
for model,key in [('CLIP','clip_cosine_matrix'),('OpenVision','openvision_cosine_matrix')]:
    a=np.array(r[key]); failures[model]=[]
    for i,j in enumerate(a.argmax(1)):
        if i!=j:
            failures[model].append({'image_id':m['images'][i]['image_id'],'matched_caption':r['captions'][i],'selected_caption':r['captions'][j],'matched_score':float(a[i,i]),'selected_score':float(a[i,j])})
(root/'retrieval_failures.json').write_text(json.dumps(failures,indent=2))
print(json.dumps(failures,indent=2))
assert len(r['maps'])==16
assert all(np.isfinite(np.load(root/t['file'])).all() for t in r['maps'])
print('16 maps checked finite')
