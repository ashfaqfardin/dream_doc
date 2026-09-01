"""E4: mask-free recurrent feature-memory insertion for Qwen-Image-Edit.

Masks position the initial collage only. The final image is always Qwen's raw
decode. Existing content is protected by replaying accepted-scene transformer
features; the new edit region is discovered from scene-to-collage feature
displacement rather than a spatial preservation mask.
"""
from __future__ import annotations
import argparse,warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageChops
from tqdm.auto import tqdm
import diffusers
import e3_prompt_suite as e3
from e1_baseline import infer,load_pipe,make_generator,save_json
from e2_sam_collage_repaint import composite,load_segmenter,place_cutout,probe_placement

HERE=Path(__file__).resolve().parent

class RecurrentFeatureMemory:
 def __init__(self,pipe,args):
  blocks=list(pipe.transformer.transformer_blocks);start=max(0,len(blocks)-args.memory_layers)
  self.blocks=blocks[start:];self.args=args;self.mode='off';self.name='';self.bank={'scene':[],'collage':[]};self.cursor=0;self.handles=[];self.stats=[];self.active=False
 def _hook(self,_module,_inputs,output):
  values=list(output) if isinstance(output,tuple) else [output];ids=[i for i,x in enumerate(values) if torch.is_tensor(x) and x.ndim==3]
  if not ids:return output
  idx=ids[-1];x=values[idx]
  if self.mode=='capture':self.bank[self.name].append(x.detach().to('cpu',dtype=torch.float16));return output
  if self.mode!='inject':return output
  k=self.cursor;self.cursor+=1
  if k>=min(len(self.bank['scene']),len(self.bank['collage'])):return output
  scene=self.bank['scene'][k].to(x.device,x.dtype);collage=self.bank['collage'][k].to(x.device,x.dtype)
  if scene.shape!=x.shape or collage.shape!=x.shape:return output
  similarity=F.cosine_similarity(x,scene,dim=-1).unsqueeze(-1)
  preserve=torch.sigmoid((similarity-self.args.similarity_threshold)/self.args.gate_temperature)
  displacement=(collage-scene).float().square().mean(-1,keepdim=True).sqrt();normalizer=displacement.mean(1,keepdim=True).clamp_min(1e-6)
  edit=torch.sigmoid((displacement/normalizer-1)/self.args.gate_temperature).to(x.dtype)
  preserve=preserve*(1-edit);a=(preserve*self.args.scene_replay_strength).clamp(0,1);b=(edit*self.args.object_replay_strength).clamp(0,1)
  values[idx]=x*(1-a-b)+scene*a+collage*b;self.active=True;self.stats.append((float(preserve.mean()),float(edit.mean())))
  return tuple(values) if isinstance(output,tuple) else values[0]
 def install(self):self.handles=[b.register_forward_hook(self._hook) for b in self.blocks]
 def capture(self,name):self.mode='capture';self.name=name;self.bank[name]=[]
 def inject(self):self.mode='inject';self.cursor=0;self.stats=[];self.active=False
 def close(self):
  for h in self.handles:h.remove()
 def report(self):return {'active':self.active,'captured_scene':len(self.bank['scene']),'captured_collage':len(self.bank['collage']),'mean_preserve_gate':float(np.mean([x[0] for x in self.stats])) if self.stats else 0,'mean_edit_gate':float(np.mean([x[1] for x in self.stats])) if self.stats else 0}

def call(pipe,image,prompt,args,seed,output_type):
 result=pipe(image=[image],prompt=prompt,negative_prompt=args.negative_prompt,true_cfg_scale=args.true_cfg_scale,guidance_scale=1.,num_inference_steps=args.steps,width=args.width,height=args.height,generator=make_generator(args.device,seed),output_type=output_type)
 return result.images if hasattr(result,'images') else result[0]

def feature_edit(pipe,scene,collage,name,args,seed):
 memory=RecurrentFeatureMemory(pipe,args);memory.install()
 try:
  memory.capture('scene');call(pipe,scene,'Preserve this complete image exactly.',args,seed,'latent')
  memory.capture('collage');call(pipe,collage,f'Preserve this image exactly including the pasted {name}.',args,seed,'latent')
  memory.inject();prompt=(f'Harmonize the newly pasted {name} naturally without moving or removing it. Preserve every previously existing object and the complete scene. Adjust only the new object perspective, lighting, contact, boundary and shadow.')
  result=call(pipe,collage,prompt,args,seed,'pil');image=(result[0] if isinstance(result,list) else result).convert('RGB')
 finally:memory.close()
 return image,memory.report()

def verify(scene,collage,result):
 s=np.asarray(scene,np.float32);c=np.asarray(collage,np.float32);r=np.asarray(result,np.float32);intended=np.mean(abs(c-s),2);actual=np.mean(abs(r-s),2)
 region=intended>max(4,float(np.quantile(intended,.9)));inside=float(actual[region].mean()) if region.any() else 0;outside=float(actual[~region].mean()) if (~region).any() else 0
 return {'object_change':inside,'background_change':outside,'score':inside-1.5*outside,'passed':inside>=8 and outside<=12}

def args_parser():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 p.add_argument('--prompts',default=str(HERE/'e3_prompts.json'));p.add_argument('--out_dir',default='results/qwen_e4_feature_memory');p.add_argument('--case_ids',type=int,nargs='+');p.add_argument('--max_objects',type=int);p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--missing_policy',choices=('skip','error'),default='skip')
 p.add_argument('--model_id',default='Qwen/Qwen-Image-Edit-2509');p.add_argument('--lightning_repo',default='lightx2v/Qwen-Image-Lightning');p.add_argument('--lightning_weight',default='Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors');p.add_argument('--lora_scale',type=float,default=1);p.add_argument('--device',default='cuda');p.add_argument('--width',type=int,default=1024);p.add_argument('--height',type=int,default=1024);p.add_argument('--steps',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--object_seed',type=int,default=1337);p.add_argument('--true_cfg_scale',type=float,default=1);p.add_argument('--negative_prompt',default=' ')
 p.add_argument('--mask_backend',choices=('auto','sam2','difference'),default='auto');p.add_argument('--sam_model_id',default='facebook/sam2-hiera-small');p.add_argument('--sam_device',default='cpu');p.add_argument('--background_threshold',type=float,default=24);p.add_argument('--probe_steps',type=int,default=4);p.add_argument('--probe_quantile',type=float,default=.88);p.add_argument('--probe_blur',type=float,default=1.2);p.add_argument('--box_margin',type=int,default=24);p.add_argument('--occupancy_margin',type=int,default=24);p.add_argument('--default_object_height',type=float,default=.22);p.add_argument('--object_height_priors');p.add_argument('--object_scale',type=float,default=.92)
 p.add_argument('--memory_layers',type=int,default=8);p.add_argument('--scene_replay_strength',type=float,default=.30);p.add_argument('--object_replay_strength',type=float,default=.20);p.add_argument('--similarity_threshold',type=float,default=.82);p.add_argument('--gate_temperature',type=float,default=.12);p.add_argument('--retries',type=int,default=1);return p.parse_args()

def main():
 args=args_parser();
 if diffusers.__version__!='0.40.0':warnings.warn(f'E4 targets diffusers 0.40.0; found {diffusers.__version__}')
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);pf=Path(args.prompts).resolve();cases=e3.select_cases(e3.load_suite(pf),args.case_ids);save_json(vars(args),out/'config.json');segmenter,backend=load_segmenter(args);pipe=load_pipe(args);refs=e3.generate_references(pipe,cases,args,out,pf);cutouts=e3.generate_cutouts(segmenter,refs,args,out);summary=[]
 for case in tqdm(cases,desc='E4 feature-memory cases'):
  cid=int(case['id']);case_dir=out/'cases'/f'case_{cid:03d}';steps=case_dir/'steps';steps.mkdir(parents=True,exist_ok=True);base_path=case_dir/'base.png'
  if base_path.is_file() and args.resume:current=Image.open(base_path).convert('RGB')
  else:current=infer(pipe,[Image.new('RGB',(args.width,args.height),'white')],'Replace blank Image 1 with: '+case['base_prompt'],args,args.seed+cid*10000);current.save(base_path)
  occupied=Image.new('L',current.size);history=[]
  for i,item in enumerate(case['objects'][:args.max_objects or None],1):
   key=e3.reference_key(item)
   if key not in cutouts:continue
   box,probe=probe_placement(pipe,current,item['name'],cutouts[key],occupied,args,args.seed+cid*100000+i*1000,steps/f'{i:02d}_{e3.slug(item["name"])}_attention_region.png');canvas,mask,placed=place_cutout(cutouts[key],box,current.size,args.object_scale);collage=composite(current,canvas,mask);collage.save(steps/f'{i:02d}_{e3.slug(item["name"])}_collage.png');best=None
   for attempt in range(args.retries+1):
    raw,memory=feature_edit(pipe,current,collage,item['name'],args,args.seed+cid*10000+i*100+attempt);metrics=verify(current,collage,raw)
    if best is None or metrics['score']>best[1]['score']:best=(raw,metrics,memory,attempt)
    if metrics['passed']:break
   current,metrics,memory,attempt=best;current.save(steps/f'{i:02d}_{e3.slug(item["name"])}_final.png');occupied=ImageChops.lighter(occupied,mask);history.append({'name':item['name'],'box':box,'placed_box':placed,'probe':probe,'verification':metrics,'feature_memory':memory,'attempt':attempt});save_json(history,case_dir/'history.json')
  current.save(case_dir/'FINAL.png');summary.append({'id':cid,'final':str(case_dir/'FINAL.png')});save_json(summary,out/'summary.json')
if __name__=='__main__':main()
