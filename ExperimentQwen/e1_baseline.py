"""E1 Qwen baseline: sketches -> objects -> base room -> sequential insertion.

Uses Qwen-Image-Edit-2509 multi-image editing with the official LightX2V
8-step Lightning LoRA. This is intentionally a mask-free baseline.
"""
from __future__ import annotations
import argparse,gc,json,math,os,random,warnings
from importlib.metadata import PackageNotFoundError,version
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import List
import numpy as np
import torch
from PIL import Image,ImageOps
from tqdm.auto import tqdm

DEFAULT_ORDER=['bicycle','vase','ball','chair','lamp','plant','backpack']
LIGHTNING_REPO='lightx2v/Qwen-Image-Lightning'
LIGHTNING_WEIGHT='Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors'

@dataclass
class ObjectResult:
 name:str; sketch:str; image:str; seed:int

def save_json(value,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 with open(path,'w',encoding='utf-8') as f:json.dump(value,f,indent=2)

def fit(image,size):
 return ImageOps.pad(image.convert('RGB'),size,Image.Resampling.LANCZOS,color='white')

def make_generator(device,seed):
 return torch.Generator(device='cuda' if device.startswith('cuda') else 'cpu').manual_seed(seed)

def lightning_scheduler():
 from diffusers import FlowMatchEulerDiscreteScheduler
 config={'base_image_seq_len':256,'base_shift':math.log(3),'invert_sigmas':False,
  'max_image_seq_len':8192,'max_shift':math.log(3),'num_train_timesteps':1000,
  'shift':1.0,'shift_terminal':None,'stochastic_sampling':False,
  'time_shift_type':'exponential','use_beta_sigmas':False,'use_dynamic_shifting':True,
  'use_exponential_sigmas':False,'use_karras_sigmas':False}
 return FlowMatchEulerDiscreteScheduler.from_config(config)

def prepare_peft_lora_backend():
 """Bypass PEFT's optional TorchAO dispatcher when an obsolete copy is installed.

 The E1 base is bf16, not TorchAO-quantized. PEFT 0.17+ nevertheless probes its
 TorchAO dispatcher first; torchao 0.10 raises instead of returning False. In
 that case the correct behavior is to skip that optional dispatcher and let
 PEFT use its standard torch.nn.Linear LoRA implementation.
 """
 try:
  installed=version('torchao')
 except PackageNotFoundError:
  return
 try:
  from packaging.version import Version
  incompatible=Version(installed)<=Version('0.16.0')
 except Exception:
  incompatible=installed.startswith(('0.0','0.1'))
 if not incompatible:
  return
 try:
  from peft.tuners.lora import torchao as peft_torchao
  peft_torchao.is_torchao_available=lambda:False
  warnings.warn(f'torchao {installed} is incompatible with PEFT LoRA loading; E1 is bf16, so the optional TorchAO dispatcher was disabled.')
 except Exception as exc:
  raise RuntimeError(f'Incompatible torchao {installed}. Run `pip uninstall -y torchao` or install torchao>0.16.0.') from exc

def load_pipe(args):
 from diffusers import QwenImageEditPlusPipeline
 stages=tqdm(total=3,desc='Loading Qwen-Edit-2509',unit='stage',dynamic_ncols=True)
 pipe=QwenImageEditPlusPipeline.from_pretrained(args.model_id,scheduler=lightning_scheduler(),dtype=torch.bfloat16)
 stages.update();stages.set_description('Loading 8-step Lightning LoRA')
 prepare_peft_lora_backend()
 pipe.load_lora_weights(args.lightning_repo,weight_name=args.lightning_weight,adapter_name='lightning')
 pipe.set_adapters(['lightning'],adapter_weights=[args.lora_scale]);stages.update();stages.set_description('Moving pipeline to GPU')
 pipe.to(args.device);pipe.set_progress_bar_config(disable=False);stages.update();stages.set_description('Qwen pipeline ready');stages.close()
 return pipe

def infer(pipe,images,prompt,args,seed):
 return pipe(image=images,prompt=prompt,negative_prompt=args.negative_prompt,
  true_cfg_scale=args.true_cfg_scale,guidance_scale=1.0,num_inference_steps=args.steps,
  width=args.width,height=args.height,generator=make_generator(args.device,seed)).images[0].convert('RGB')

def discover_sketches(directory):
 root=Path(directory);found={p.stem.removeprefix('sketch_'):p for p in root.glob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}}
 ordered=[name for name in DEFAULT_ORDER if name in found]+sorted(set(found)-set(DEFAULT_ORDER))
 if not ordered:raise FileNotFoundError(f'No sketch images found in {root}')
 return [(name,found[name]) for name in ordered]

def main():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 p.add_argument('--sketch_dir',default='KontextPipeline/sketch');p.add_argument('--out_dir',default='results/qwen_e1_baseline')
 p.add_argument('--model_id',default='Qwen/Qwen-Image-Edit-2509');p.add_argument('--lightning_repo',default=LIGHTNING_REPO);p.add_argument('--lightning_weight',default=LIGHTNING_WEIGHT);p.add_argument('--lora_scale',type=float,default=1.0)
 p.add_argument('--device',default='cuda');p.add_argument('--width',type=int,default=1024);p.add_argument('--height',type=int,default=1024);p.add_argument('--steps',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--true_cfg_scale',type=float,default=1.0);p.add_argument('--negative_prompt',default=' ')
 p.add_argument('--base_prompt',default='Create a photorealistic empty modern living room with warm neutral walls, wooden floor, natural daylight, realistic perspective, and several physically plausible open areas for furniture and objects. Keep the room uncluttered and completely empty.')
 args=p.parse_args();out=Path(args.out_dir);objects_dir=out/'objects';steps_dir=out/'steps';objects_dir.mkdir(parents=True,exist_ok=True);steps_dir.mkdir(exist_ok=True)
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);save_json(vars(args),out/'config.json')
 sketches=discover_sketches(args.sketch_dir);pipe=load_pipe(args);results:List[ObjectResult]=[]

 print('\n=== PHASE 1: SKETCH -> PHOTOREALISTIC OBJECT ===')
 for index,(name,path) in enumerate(tqdm(sketches,desc='Generating objects',unit='object'),1):
  sketch=fit(Image.open(path),(args.width,args.height));sketch.save(objects_dir/f'{index:02d}_{name}_sketch.png')
  prompt=(f'Image 1 is a sketch of a {name}. Convert it into one photorealistic {name}. Preserve the sketch geometry, pose, proportions, viewpoint and every visible component. Use realistic materials, coherent colors and studio lighting. Show the complete object centered on a plain clean white background. Add no other objects, labels, frames, floor or scenery.')
  seed=args.seed+index*1000;image=infer(pipe,[sketch],prompt,args,seed);target=objects_dir/f'{index:02d}_{name}.png';image.save(target);results.append(ObjectResult(name,str(path),str(target),seed))
 save_json([asdict(x) for x in results],out/'objects.json')

 print('\n=== PHASE 2: BLANK CANVAS -> BASE ROOM ===')
 blank=Image.new('RGB',(args.width,args.height),'white')
 base_prompt=('Replace the blank Image 1 with this scene: '+args.base_prompt+' Do not leave a white border or blank canvas.')
 base=infer(pipe,[blank],base_prompt,args,args.seed+50000);base.save(out/'base.png')

 print('\n=== PHASE 3: SEQUENTIAL MULTI-IMAGE INSERTION ===')
 current=base;history=[]
 for index,item in enumerate(tqdm(results,desc='Inserting objects',unit='object'),1):
  reference=fit(Image.open(item.image),(args.width,args.height));before=steps_dir/f'{index:02d}_{item.name}_before.png';current.save(before)
  prompt=(f'Image 1 is the current room scene. Image 2 shows the exact {item.name} to insert. Edit only Image 1: add exactly one complete instance of the Image 2 {item.name} in a physically plausible unoccupied location. Preserve its distinctive identity, colors, material, structure and proportions while adapting scale, perspective, illumination, support contact and occlusion to the room. Keep it fully inside the frame. Preserve all existing objects and unrelated scene content. Do not copy the white background from Image 2 and do not create a collage or split image.')
  seed=args.seed+100000+index*1000;current=infer(pipe,[current,reference],prompt,args,seed);target=steps_dir/f'{index:02d}_{item.name}_after.png';current.save(target);history.append({'step':index,'name':item.name,'seed':seed,'before':str(before),'after':str(target),'reference':item.image})
 current.save(out/'FINAL.png');save_json(history,out/'history.json');print('Done:',out/'FINAL.png')

if __name__=='__main__':main()
