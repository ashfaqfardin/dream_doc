"""E5: object feature transplantation with Qwen-Image-Edit.

Encodes each isolated object with the same Qwen transformer, captures its
hidden features and image-stream K/V projections, aligns them to a
counterfactual placement region, and injects them while editing the scene.
The final output is Qwen's raw decode; masks are never used for preservation.
"""
from __future__ import annotations
import argparse,warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageChops,ImageFilter
from tqdm.auto import tqdm
import diffusers
import e3_prompt_suite as e3
import e4_feature_frequency_locked_insertion as e4
from e1_baseline import infer,load_pipe,make_generator,save_json
from e2_sam_collage_repaint import composite,load_segmenter,place_cutout,probe_placement

HERE=Path(__file__).resolve().parent

def soft_gate(mask,n,device,dtype):
 a=np.asarray(mask.resize((64,64),Image.Resampling.BILINEAR),np.float32)/255;g=torch.from_numpy(a).to(device,dtype).flatten()
 # Qwen concatenates generated-image tokens before conditioning-image tokens.
 # Never inject into the latter. Interpolating over the entire concatenated
 # sequence was the main cause of activation collapse / black decodes.
 if n>g.numel() and n%g.numel()==0:g=torch.cat([g,torch.zeros(n-g.numel(),device=device,dtype=dtype)])
 elif n!=g.numel():g=F.interpolate(g[None,None],size=n,mode='linear',align_corners=False).flatten()
 return g.reshape(1,n,1)

class ObjectTransplant:
 """Capture/inject object hidden states and image-stream K/V projections."""
 def __init__(self,pipe,args,gate_mask):
  blocks=list(pipe.transformer.transformer_blocks);self.blocks=blocks[max(0,len(blocks)-args.transplant_layers):];self.args=args;self.mask=gate_mask
  self.mode='off';self.capture_name='';self.bank={n:{'hidden':[],'k':[],'v':[]} for n in ('neutral','object')};self.cursor={'hidden':0,'k':0,'v':0};self.handles=[];self.used={'hidden':0,'k':0,'v':0}
 def _route(self,kind,x):
  if self.mode=='capture':self.bank[self.capture_name][kind].append(x.detach().to('cpu',dtype=torch.float16));return x
  if self.mode!='inject' or kind not in self.args.transplant_mode:return x
  i=self.cursor[kind];self.cursor[kind]+=1
  if i>=min(len(self.bank['object'][kind]),len(self.bank['neutral'][kind])):return x
  obj=self.bank['object'][kind][i].to(x.device,x.dtype);neutral=self.bank['neutral'][kind][i].to(x.device,x.dtype)
  if obj.shape!=neutral.shape or obj.ndim!=3 or x.ndim!=3:return x
  residual=obj-neutral
  if residual.shape[1]!=x.shape[1]:residual=F.interpolate(residual.transpose(1,2),size=x.shape[1],mode='linear',align_corners=False).transpose(1,2)
  if residual.shape[-1]!=x.shape[-1]:return x
  # Preserve the target activation distribution; inject a unit-RMS residual
  # instead of replacing it with an unrelated absolute object trajectory.
  residual=residual/(residual.float().square().mean(-1,keepdim=True).sqrt().to(x.dtype).clamp_min(1e-5));residual=residual*x.float().square().mean(-1,keepdim=True).sqrt().to(x.dtype)
  gate=soft_gate(self.mask,x.shape[1],x.device,x.dtype);strength=self.args.kv_strength if kind in {'k','v'} else self.args.hidden_strength;self.used[kind]+=1
  return x+residual*(gate*strength)
 def hidden_hook(self,_m,_a,out):
  vals=list(out) if isinstance(out,tuple) else [out];ids=[i for i,x in enumerate(vals) if torch.is_tensor(x) and x.ndim==3]
  if ids:vals[ids[-1]]=self._route('hidden',vals[ids[-1]])
  return tuple(vals) if isinstance(out,tuple) else vals[0]
 def proj_hook(self,kind):return lambda _m,_a,out:self._route(kind,out)
 def install(self):
  for b in self.blocks:
   self.handles.append(b.register_forward_hook(self.hidden_hook));attn=getattr(b,'attn',None)
   for kind,name in [('k','to_k'),('v','to_v')]:
    module=getattr(attn,name,None)
    if module is not None:self.handles.append(module.register_forward_hook(self.proj_hook(kind)))
 def capture(self,name):self.mode='capture';self.capture_name=name;self.bank[name]={k:[] for k in self.bank[name]}
 def inject(self):self.mode='inject';self.cursor={k:0 for k in self.cursor};self.used={k:0 for k in self.used}
 def close(self):
  for h in self.handles:h.remove()

def call(pipe,image,prompt,args,seed,output_type='pil',latents=None):
 kw=dict(image=[image],prompt=prompt,negative_prompt=args.negative_prompt,true_cfg_scale=args.true_cfg_scale,guidance_scale=1.,num_inference_steps=args.steps,width=args.width,height=args.height,generator=make_generator(args.device,seed),output_type=output_type)
 if latents is not None:kw['latents']=latents
 out=pipe(**kw);return out.images if hasattr(out,'images') else out[0]

def transplant_edit(pipe,scene,reference,collage,name,mask,args,seed):
 controller=ObjectTransplant(pipe,args,mask);controller.install()
 try:
  neutral=Image.new('RGB',reference.size,'white');capture_prompt='Preserve Image 1 exactly without adding or changing anything.'
  controller.capture('neutral');call(pipe,neutral,capture_prompt,args,seed,'latent')
  controller.capture('object');call(pipe,reference,capture_prompt,args,seed,'latent')
  controller.inject();prompt=(f'Naturally integrate the newly placed {name}. Preserve its identity and complete structure, and preserve all existing scene content. Adjust only perspective, light, contact and shadow.')
  result=call(pipe,collage,prompt,args,seed,'pil');image=(result[0] if isinstance(result,list) else result).convert('RGB')
 finally:controller.close()
 return image,{'captured':{n:{k:len(v) for k,v in bank.items()} for n,bank in controller.bank.items()},'used':controller.used,'mode':args.transplant_mode,'injection':'normalized_object_minus_neutral_residual'}

def parser():
 parent=e4.build_parser(add_help=False);p=argparse.ArgumentParser(parents=[parent],formatter_class=argparse.ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
 p.set_defaults(out_dir='results/qwen_e5_object_feature_transplant')
 p.add_argument('--transplant_mode',choices=('hidden','kv','hybrid'),default='hidden');p.add_argument('--transplant_layers',type=int,default=6);p.add_argument('--hidden_strength',type=float,default=.06);p.add_argument('--kv_strength',type=float,default=.03);return p

def main():
 args=parser().parse_args();args.transplant_mode={'hybrid':'hiddenkv'}.get(args.transplant_mode,args.transplant_mode)
 if diffusers.__version__!='0.40.0':warnings.warn(f'E5 targets diffusers 0.40.0; found {diffusers.__version__}')
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);pf=Path(args.prompts).resolve();cases=e3.select_cases(e3.load_suite(pf),args.case_ids);save_json(vars(args),out/'config.json');sam,backend=load_segmenter(args);pipe=load_pipe(args);refs=e3.generate_references(pipe,cases,args,out,pf);cuts=e3.generate_cutouts(sam,refs,args,out);summary=[]
 for case in tqdm(cases,desc='E5 transplant cases'):
  cid=int(case['id']);d=out/'cases'/f'case_{cid:03d}';sd=d/'steps';sd.mkdir(parents=True,exist_ok=True);bp=d/'base.png'
  if bp.is_file() and args.resume:current=Image.open(bp).convert('RGB')
  else:current=infer(pipe,[Image.new('RGB',(args.width,args.height),'white')],'Replace blank Image 1 with: '+case['base_prompt'],args,args.seed+cid*10000);current.save(bp)
  occupied=Image.new('L',current.size);history=[]
  for i,item in enumerate(case['objects'][:args.max_objects or None],1):
   key=e3.reference_key(item)
   if key not in cuts:continue
   box,probe=probe_placement(pipe,current,item['name'],cuts[key],occupied,args,args.seed+cid*100000+i*1000,sd/f'{i:02d}_{e3.slug(item["name"])}_placement.png');canvas,mask,placed=place_cutout(cuts[key],box,current.size,args.object_scale);collage=composite(current,canvas,mask);reference=Image.open(refs[key]['image']).convert('RGB').resize(current.size)
   current,report=transplant_edit(pipe,current,reference,collage,item['name'],mask,args,args.seed+cid*10000+i*100);current.save(sd/f'{i:02d}_{e3.slug(item["name"])}_final.png');occupied=ImageChops.lighter(occupied,mask);history.append({'name':item['name'],'box':box,'probe':probe,'transplant':report});save_json(history,d/'history.json')
  current.save(d/'FINAL.png');summary.append({'id':cid,'final':str(d/'FINAL.png')});save_json(summary,out/'summary.json')
if __name__=='__main__':main()
