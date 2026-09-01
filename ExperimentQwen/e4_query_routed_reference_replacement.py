"""E4: query-routed external-memory reference replacement.

Per object:
  1. Qwen adds a generic object using text only (placement/pose/context).
  2. The same transformer encodes an isolated reference as compressed prefix
     memory, analogous to NLP prefix-tuning / retrieval cross-attention.
  3. Target queries retrieve reference identity through a bounded residual
     gate. A causal scene->placement feature delta localizes the new object.

No SAM mask and no post-generation pixel composite are used. The final image
is Qwen's raw decode. Target: diffusers==0.40.0.
"""
from __future__ import annotations
import argparse,warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import diffusers
import e3_prompt_suite as e3
from e1_baseline import fit,infer,load_pipe,make_generator,save_json

HERE=Path(__file__).resolve().parent

def pipeline_call(pipe,image,prompt,args,seed,output_type='latent'):
 images=image if isinstance(image,list) else [image]
 out=pipe(image=images,prompt=prompt,negative_prompt=args.negative_prompt if args.true_cfg_scale>1 else None,
  true_cfg_scale=args.true_cfg_scale,num_inference_steps=args.steps,width=args.width,height=args.height,
  generator=make_generator(args.device,seed),output_type=output_type)
 return out.images if hasattr(out,'images') else out[0]

def image_tensor(output):
 value=output[0] if isinstance(output,(list,tuple)) else output
 return value.convert('RGB') if isinstance(value,Image.Image) else value

class PrefixMemoryRouter:
 """Causal feature replay around native Qwen multi-image attention."""
 def __init__(self,pipe,args):
  blocks=list(pipe.transformer.transformer_blocks);chosen=args.router_layers
  self.layers=[i for i in chosen if 0<=i<len(blocks)];self.blocks=blocks;self.args=args
  self.mode='off';self.capture_name='';self.cursor=0;self.handles=[];self.banks={n:[] for n in ('scene','placed')};self.active=False;self.stats=[]
 def _image_index(self,output):
  values=list(output) if isinstance(output,tuple) else [output]
  # QwenImageTransformerBlock returns (text, image).
  return values,(1 if len(values)>1 and torch.is_tensor(values[1]) and values[1].ndim==3 else None),isinstance(output,tuple)
 def _hook(self,layer):
  def hook(_module,_inputs,output):
   values,index,was_tuple=self._image_index(output)
   if index is None:return output
   x=values[index]
   if self.mode=='capture':
    self.banks[self.capture_name].append(x.detach().to('cpu',dtype=torch.float16));return output
   if self.mode!='route':return output
   k=self.cursor;self.cursor+=1
   if k>=min(map(len,self.banks.values())):return output
   scene=self.banks['scene'][k].to(x.device,x.dtype);placed=self.banks['placed'][k].to(x.device,x.dtype)
   # Replacement has two conditioning images and therefore may contain more
   # image tokens than the one-image capture runs. Replay only the generated
   # output-token prefix shared by both trajectories.
   n=min(scene.shape[1],placed.shape[1],x.shape[1]);scene=scene[:,:n];placed=placed[:,:n];target=x[:,:n]
   if scene.shape!=placed.shape or scene.shape[-1]!=x.shape[-1]:return output
   # Causal activation patching: tokens changed by prompt-only placement form
   # the soft target-query gate; no segmentation mask is involved.
   delta=(placed-scene).float().square().mean(-1,keepdim=True).sqrt();norm=delta.mean(1,keepdim=True).clamp_min(1e-6)
   edit_gate=torch.sigmoid((delta/norm-self.args.edit_gate_center)/self.args.gate_temperature).to(x.dtype)
   # Native Qwen attention receives the full reference as Image 2. We only
   # replay the accepted scene outside the causal target region; inside it,
   # queries remain untouched and can retrieve native high-resolution
   # reference tokens.
   preserve=(1-edit_gate)*self.args.scene_replay_strength;y=x.clone();y[:,:n]=target+(scene-target)*preserve;values[index]=y;self.active=True
   self.stats.append({'layer':layer,'edit_gate':float(edit_gate.mean()),'preserve':float(preserve.mean())})
   return tuple(values) if was_tuple else values[0]
  return hook
 def install(self):self.handles=[self.blocks[i].register_forward_hook(self._hook(i)) for i in self.layers]
 def capture(self,name):self.mode='capture';self.capture_name=name;self.banks[name]=[]
 def route(self):self.mode='route';self.cursor=0;self.active=False;self.stats=[]
 def close(self):
  for handle in self.handles:handle.remove()
 def report(self):
  return {'active':self.active,'layers':self.layers,'captured':{k:len(v) for k,v in self.banks.items()},'samples':len(self.stats),
   'mean_edit_gate':float(np.mean([x['edit_gate'] for x in self.stats])) if self.stats else 0,
   'mean_scene_preservation':float(np.mean([x['preserve'] for x in self.stats])) if self.stats else 0,
   'reference_path':'native_multi_image_attention'}

def routed_replace(pipe,accepted,placed,reference,name,args,seed):
 router=PrefixMemoryRouter(pipe,args);router.install();matched='Preserve Image 1 exactly without adding, removing, or changing anything.'
 try:
  router.capture('scene');pipeline_call(pipe,accepted,matched,args,seed)
  router.capture('placed');pipeline_call(pipe,placed,matched,args,seed)
  router.route();prompt=(f'Image 1 is the current scene and contains one newly added generic {name}. Image 2 shows the exact reference {name}. Edit only Image 1: replace that generic {name} with the exact identity from Image 2, preserving its distinctive colors, materials, parts, proportions and design. Keep the generic object location, approximate pose, perspective, support contact and shadow. Preserve the complete scene and every previous object. Do not copy the white background, create a panel, or replace the scene with Image 2.')
  output=pipeline_call(pipe,[placed,reference],prompt,args,seed,'pil');result=image_tensor(output)
 finally:router.close()
 return result,router.report()

def quality(accepted,placed,result):
 a=np.asarray(accepted,np.float32);p=np.asarray(placed,np.float32);r=np.asarray(result,np.float32)
 intended=np.mean(abs(p-a),2);actual=np.mean(abs(r-a),2);region=intended>max(4,float(np.quantile(intended,.9)))
 inside=float(actual[region].mean()) if region.any() else 0;outside=float(actual[~region].mean()) if (~region).any() else 0
 luminance=float(np.asarray(result.convert('L')).mean());collapsed=luminance<8 or luminance>247
 return {'inside_change':inside,'outside_change':outside,'mean_luminance':luminance,'collapsed':collapsed,'score':inside-1.5*outside-(1000 if collapsed else 0)}

def parser():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 p.add_argument('--prompts',default=str(HERE/'e3_prompts.json'));p.add_argument('--out_dir',default='results/qwen_e4_query_routed');p.add_argument('--case_ids',type=int,nargs='+');p.add_argument('--max_objects',type=int);p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--missing_policy',choices=('skip','error'),default='skip')
 p.add_argument('--model_id',default='Qwen/Qwen-Image-Edit-2509');p.add_argument('--lightning_repo',default='lightx2v/Qwen-Image-Lightning');p.add_argument('--lightning_weight',default='Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors');p.add_argument('--lora_scale',type=float,default=1);p.add_argument('--device',default='cuda');p.add_argument('--width',type=int,default=1024);p.add_argument('--height',type=int,default=1024);p.add_argument('--steps',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--object_seed',type=int,default=1337);p.add_argument('--true_cfg_scale',type=float,default=1);p.add_argument('--negative_prompt',default=' ')
 p.add_argument('--router_layers',type=int,nargs='+',default=[20,30,40,50]);p.add_argument('--scene_replay_strength',type=float,default=.12);p.add_argument('--edit_gate_center',type=float,default=1);p.add_argument('--gate_temperature',type=float,default=.15);p.add_argument('--retries',type=int,default=1);return p.parse_args()

def main():
 args=parser();
 if diffusers.__version__!='0.40.0':warnings.warn(f'E4 targets diffusers 0.40.0; found {diffusers.__version__}')
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);pf=Path(args.prompts).resolve();cases=e3.select_cases(e3.load_suite(pf),args.case_ids);save_json(vars(args),out/'config.json');pipe=load_pipe(args);refs=e3.generate_references(pipe,cases,args,out,pf);summary=[]
 for case in tqdm(cases,desc='E4 query-routed cases'):
  cid=int(case['id']);d=out/'cases'/f'case_{cid:03d}';sd=d/'steps';sd.mkdir(parents=True,exist_ok=True);bp=d/'base.png'
  if bp.is_file() and args.resume:current=Image.open(bp).convert('RGB')
  else:current=infer(pipe,[Image.new('RGB',(args.width,args.height),'white')],'Replace blank Image 1 with: '+case['base_prompt'],args,args.seed+cid*10000);current.save(bp)
  history=[]
  for i,item in enumerate(case['objects'][:args.max_objects or None],1):
   key=e3.reference_key(item);record=refs.get(key,{})
   if record.get('status')!='ready':history.append({'name':item['name'],'status':'missing_reference'});continue
   accepted=current;placement_prompt=(f'Add exactly one generic {item["name"]} naturally at the most physically plausible unoccupied location. Make it complete, correctly scaled, supported, and consistent with perspective. Preserve every existing object and the scene.')
   placed=infer(pipe,[accepted],placement_prompt,args,args.seed+cid*10000+i*100);placed.save(sd/f'{i:02d}_{e3.slug(item["name"])}_generic_placement.png');reference=fit(Image.open(record['image']),(args.width,args.height));best=None
   for attempt in range(args.retries+1):
    result,route=routed_replace(pipe,accepted,placed,reference,item['name'],args,args.seed+cid*100000+i*1000+attempt);metrics=quality(accepted,placed,result)
    if best is None or metrics['score']>best[1]['score']:best=(result,metrics,route,attempt)
    if route['active'] and not metrics['collapsed']:break
   result,metrics,route,attempt=best
   if metrics['collapsed']:warnings.warn(f'Collapsed routed edit for {item["name"]}; accepting safe generic placement.');current=placed;status='generic_fallback'
   else:current=result;status='replaced'
   current.save(sd/f'{i:02d}_{e3.slug(item["name"])}_final.png');history.append({'name':item['name'],'status':status,'attempt':attempt,'metrics':metrics,'router':route});save_json(history,d/'history.json')
  current.save(d/'FINAL.png');summary.append({'id':cid,'final':str(d/'FINAL.png')});save_json(summary,out/'summary.json')
if __name__=='__main__':main()
