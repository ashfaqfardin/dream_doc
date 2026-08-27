"""E18: twelve-RQ mechanistic lab for FLUX.1-Kontext (no target mask)."""
from __future__ import annotations
import argparse,csv,json,math,os,random,types
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image,ImageDraw,ImageOps

GROUPS={"inventory":[1],"trace":[2,3,4,5,6],"controls":[7,8,9],"rope":[10],"causal":[11,12]}
def indices(s,n):
 o=set()
 for x in s.split(','):
  if x=='all':o|=set(range(n))
  elif '-' in x:a,b=map(int,x.split('-'));o|=set(range(a,b+1))
  elif x:o.add(int(x))
 if any(i<0 or i>=n for i in o):raise ValueError(f"bad indices {o} for n={n}")
 return o
def dump(x,p):p.parent.mkdir(parents=True,exist_ok=True);json.dump(x,open(p,'w',encoding='utf8'),indent=2)
def grid(d,p):
 w=384; c=Image.new('RGB',(w*min(3,len(d)),(w+30)*math.ceil(len(d)/3)),'white');q=ImageDraw.Draw(c)
 for i,(k,v) in enumerate(d.items()):x=i%3*w;y=i//3*(w+30);q.text((x+5,y+8),k,fill='black');c.paste(ImageOps.fit(v,(w,w)),(x,y+30))
 c.save(p)
def dist(a,b):
 x=np.asarray(a.resize((256,256)),dtype=float)/255;y=np.asarray(b.resize((256,256)),dtype=float)/255
 return float(np.mean(abs(x-y)))
def pack(pipe,tids=(1,2)):
 from diffusers.utils.torch_utils import randn_tensor
 def f(self,image,batch_size,num_channels_latents,height,width,dtype,device,generator=None,latents=None):
  height=2*(height//(self.vae_scale_factor*2));width=2*(width//(self.vae_scale_factor*2));shape=(batch_size,num_channels_latents,height,width)
  raw=self._encode_vae_image(image.to(device=device,dtype=dtype),generator=generator); ps=[];ids=[]
  if raw.shape[0]!=2*batch_size:raise RuntimeError('E18 requires exactly two contexts')
  for i,t in enumerate(tids):
   z=raw[i*batch_size:(i+1)*batch_size];h,w=z.shape[-2:];ps.append(self._pack_latents(z,batch_size,num_channels_latents,h,w));u=self._prepare_latent_image_ids(batch_size,h//2,w//2,device,dtype);u[...,0]=t;ids.append(u)
  lid=self._prepare_latent_image_ids(batch_size,height//2,width//2,device,dtype)
  if latents is None:latents=self._pack_latents(randn_tensor(shape,generator=generator,device=device,dtype=dtype),batch_size,num_channels_latents,height,width)
  return latents,torch.cat(ps,1),lid,torch.cat(ids,0)
 pipe.prepare_latents=types.MethodType(f,pipe)
class Probe:
 def __init__(self,tr,layers,steps,labels=('scene','reference'),block=None):
  self.tr,self.layers,self.steps,self.labels,self.block=tr,layers,steps,labels,block;self.step=-1;self.layer=-1;self.layout=None;self.rows=[];self.orig=F.scaled_dot_product_attention;self.h=[]
  self.h.append(tr.register_forward_pre_hook(self.pre,with_kwargs=True));bs=list(tr.transformer_blocks)+list(getattr(tr,'single_transformer_blocks',[]))
  for i,b in enumerate(bs):self.h+=[b.register_forward_pre_hook(lambda m,a,k,j=i:setattr(self,'layer',j),with_kwargs=True),b.register_forward_hook(lambda m,a,o:setattr(self,'layer',-1))]
  F.scaled_dot_product_attention=self.sdpa
 def pre(self,m,a,k):
  self.step+=1;h=k.get('hidden_states');e=k.get('encoder_hidden_states');t=k.get('timestep');self.time=float(t.flatten()[0].cpu()) if t is not None else 0
  if h is not None:
   if h.shape[1]%3:raise RuntimeError('unexpected Kontext token packing')
   self.layout=(e.shape[1],h.shape[1]//3)
 def sdpa(self,*a,**kw):
  # PyTorch accepts both positional and keyword-only SDPA calls. Recent
  # Diffusers attention_dispatch uses the latter.
  q=a[0] if len(a)>0 else kw.get('query');k=a[1] if len(a)>1 else kw.get('key');v=a[2] if len(a)>2 else kw.get('value')
  if q is None or k is None or v is None:return self.orig(*a,**kw)
  if self.layout and self.layer in self.layers and self.step in self.steps and q.shape[-2]==self.layout[0]+3*self.layout[1]:
   nt,ni=self.layout;sl={'text':slice(0,nt),'target':slice(nt,nt+ni),self.labels[0]:slice(nt+ni,nt+2*ni),self.labels[1]:slice(nt+2*ni,nt+3*ni)}
   supplied_scale=kw.get('scale');scale=float(supplied_scale) if supplied_scale is not None else q.shape[-1]**-.5
   if self.block:
    scores=q[:,:,sl['target']].float()@k.float().transpose(-2,-1)*scale
    scores[:,:,:,sl[self.block]]=torch.finfo(scores.dtype).min;w=torch.softmax(scores,-1).to(v.dtype)
    if len(a)>=3: full=self.orig(*a,**kw)
    else:
     forwarded=dict(kw);forwarded.update(query=q,key=k,value=v);full=self.orig(**forwarded)
    full[:,:,sl['target']]=w@v;return full
   # Reduce in query chunks; never materialize the full target-by-source matrix.
   totals={src:torch.zeros(q.shape[1],device=q.device) for src in sl};ents={src:torch.zeros(q.shape[1],device=q.device) for src in sl};count=0
   for lo in range(0,ni,128):
    scores=q[:,:,sl['target'],:][:,:,lo:lo+128].float()@k.float().transpose(-2,-1)*scale;w=torch.softmax(scores,-1);count+=w.shape[2]
    for src,r in sl.items():
     p=w[:,:,:,r];totals[src]+=p.sum(-1).sum((0,2));z=p/p.sum(-1,keepdim=True).clamp_min(1e-9);ents[src]+=(-(z*z.clamp_min(1e-9).log()).sum(-1)/math.log(max(2,p.shape[-1]))).sum((0,2))
   for src in sl:
    mass=totals[src]/count;ent=ents[src]/count
    for head in range(len(mass)):self.rows.append({'step':self.step,'timestep':self.time,'layer':self.layer,'head':head,'source':src,'mass':float(mass[head]),'entropy':float(ent[head])})
  return self.orig(*a,**kw)
 def close(self):F.scaled_dot_product_attention=self.orig;[x.remove() for x in self.h]
def run(pipe,args,ctx,tids=(1,2),probe=None):
 pack(pipe,tids);g=torch.Generator(device='cuda' if args.device.startswith('cuda') else 'cpu').manual_seed(args.seed)
 return pipe(image=list(ctx),prompt=args.prompt,width=args.width,height=args.height,max_area=args.width*args.height,num_inference_steps=args.steps,guidance_scale=args.guidance,generator=g).images[0].convert('RGB')
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--reference',required=True);p.add_argument('--object_name',default='object');p.add_argument('--prompt');p.add_argument('--out_dir',default='results/e18');p.add_argument('--rqs',default='inventory,trace');p.add_argument('--model_id',default='black-forest-labs/FLUX.1-Kontext-dev');p.add_argument('--device',default='cuda');p.add_argument('--width',type=int,default=1024);p.add_argument('--height',type=int,default=1024);p.add_argument('--steps',type=int,default=28);p.add_argument('--guidance',type=float,default=2.5);p.add_argument('--seed',type=int,default=42);p.add_argument('--trace_steps',default='0,7,14,21,27');p.add_argument('--trace_layers',default='all');p.add_argument('--ablation_layers',default='0-18');a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
 groups=list(GROUPS) if a.rqs=='all' else a.rqs.split(',');a.prompt=a.prompt or f'Image one is the scene. Add the exact {a.object_name} from image two naturally.'
 from diffusers import FluxKontextPipeline;pipe=FluxKontextPipeline.from_pretrained(a.model_id,torch_dtype=torch.bfloat16).to(a.device);n=len(pipe.transformer.transformer_blocks)+len(getattr(pipe.transformer,'single_transformer_blocks',[]));ls=indices(a.trace_layers,n);ss=indices(a.trace_steps,a.steps);als=indices(a.ablation_layers,n)
 scene=ImageOps.pad(Image.open(a.scene).convert('RGB'),(a.width,a.height));ref=ImageOps.pad(Image.open(a.reference).convert('RGB'),(a.width,a.height));blank=Image.new('RGB',ref.size,tuple(np.asarray(ref).mean((0,1)).astype(int)));shuf=ref.resize((16,16)).resize(ref.size)
 dump({'args':vars(a),'groups':{x:GROUPS[x] for x in groups},'layers':n},out/'manifest.json')
 if 'inventory' in groups:
  dump({'config':dict(pipe.transformer.config),'double_blocks':len(pipe.transformer.transformer_blocks),'single_blocks':len(getattr(pipe.transformer,'single_transformer_blocks',[])),'parameters':sum(x.numel() for x in pipe.transformer.parameters())},out/'architecture.json')
 if 'trace' in groups:
  pr=Probe(pipe.transformer,ls,ss)
  try:im=run(pipe,a,(scene,ref));im.save(out/'trace_output.png')
  finally:pr.close()
  dump({'layout':pr.layout,'rows':pr.rows},out/'routing.json');save_csv=csv.DictWriter(open(out/'routing.csv','w',newline=''),fieldnames=pr.rows[0].keys());save_csv.writeheader();save_csv.writerows(pr.rows)
  for axis in ('layer','step'):
   xs=sorted({r[axis] for r in pr.rows});fig,ax=plt.subplots();
   for src in sorted({r['source'] for r in pr.rows}):ax.plot(xs,[np.mean([r['mass'] for r in pr.rows if r[axis]==x and r['source']==src]) for x in xs],label=src)
   ax.legend();ax.set(xlabel=axis,ylabel='target attention mass');fig.savefig(out/f'routing_by_{axis}.png',dpi=300);plt.close(fig)
 if 'controls' in groups:
  cond={'correct':((scene,ref),(1,2)),'blank':((scene,blank),(1,2)),'shuffled':((scene,shuf),(1,2)),'reversed':((ref,scene),(1,2)),'scene_scene':((scene,scene),(1,2)),'ref_ref':((ref,ref),(1,2))};ims={k:run(pipe,a,*v) for k,v in cond.items()};grid(ims,out/'controls.png');dump({k:dist(ims['correct'],v) for k,v in ims.items()},out/'control_effects.json')
 if 'rope' in groups:
  ims={k:run(pipe,a,(scene,ref),ids) for k,ids in {'normal':(1,2),'shared':(1,1),'swapped':(2,1)}.items()};grid(ims,out/'rope.png');dump({k:dist(ims['normal'],v) for k,v in ims.items()},out/'rope_effects.json')
 if 'causal' in groups:
  ims={'baseline':run(pipe,a,(scene,ref))}
  for src in ('text','scene','reference'):
   pr=Probe(pipe.transformer,als,set(range(a.steps)),block=src)
   try:ims['block_'+src]=run(pipe,a,(scene,ref))
   finally:pr.close()
  grid(ims,out/'causal.png');dump({k:dist(ims['baseline'],v) for k,v in ims.items()},out/'causal_effects.json')
 print('E18 complete:',out)
if __name__=='__main__':main()
