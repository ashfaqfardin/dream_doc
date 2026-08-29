"""E2: SAM cutout -> positioned collage -> one-pass Qwen harmonization.

Unlike E1, Qwen receives exactly one image per insertion. The reference object
is already positioned in that image, eliminating scene/reference role collapse.
"""
from __future__ import annotations
import argparse,json,math,os,warnings
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Dict,Tuple
import numpy as np
from PIL import Image,ImageChops,ImageDraw,ImageFilter
from tqdm.auto import tqdm
from e1_baseline import discover_sketches,fit,load_pipe,infer,make_generator,save_json

# Category-size priors constrain scale, but never prescribe position.  Values
# are fractions of image height and can be overridden from the command line.
DEFAULT_HEIGHT_PRIORS={'bicycle':.48,'vase':.16,'ball':.11,'chair':.38,
 'lamp':.46,'plant':.30,'backpack':.22}

@dataclass
class Cutout:
 name:str; rgb:Image.Image; alpha:Image.Image; source_box:Tuple[int,int,int,int]

def dilate(mask,r):return mask if r<=0 else mask.filter(ImageFilter.MaxFilter(2*r+1))
def erode(mask,r):return mask if r<=0 else mask.filter(ImageFilter.MinFilter(2*r+1))

def foreground_box(image,threshold):
 a=np.array(image.convert('RGB'),dtype=np.int16,copy=True);border=np.concatenate([a[0],a[-1],a[:,0],a[:,-1]],axis=0)
 bg=np.median(border,axis=0);distance=np.sqrt(((a-bg)**2).sum(2));mask=Image.fromarray(np.uint8(distance>threshold)*255)
 mask=mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(5));box=mask.getbbox()
 if box is None:raise RuntimeError('Could not distinguish object from reference background')
 return box

class SAM:
 def __init__(self,model_id,device):
  try:from sam2.sam2_image_predictor import SAM2ImagePredictor
  except ModuleNotFoundError as exc:raise ModuleNotFoundError("SAM2 is not installed. Run `pip install git+https://github.com/facebookresearch/sam2.git`, then restart the runtime.") from exc
  self.predictor=SAM2ImagePredictor.from_pretrained(model_id,device=device)
 def cutout(self,name,image,threshold):
  box=foreground_box(image,threshold);self.predictor.set_image(np.array(image.convert('RGB'),dtype=np.uint8,copy=True))
  masks,scores,_=self.predictor.predict(box=np.asarray(box,dtype=np.float32),multimask_output=True)
  mask=np.asarray(masks)[int(np.asarray(scores).argmax())];alpha=Image.fromarray(np.uint8(mask)*255);tight=alpha.getbbox()
  if tight is None:raise RuntimeError(f'SAM returned empty mask for {name}')
  return Cutout(name,image.crop(tight),alpha.crop(tight),tuple(map(int,tight)))

class DifferenceCutout:
 """Fallback for E1/E2 references, which are generated on a plain background."""
 def cutout(self,name,image,threshold):
  box=foreground_box(image,threshold);a=np.asarray(image.convert('RGB'),dtype=np.int16);border=np.concatenate([a[0],a[-1],a[:,0],a[:,-1]],axis=0);bg=np.median(border,axis=0)
  distance=np.sqrt(((a-bg)**2).sum(2));alpha=Image.fromarray(np.uint8(distance>threshold)*255).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.0));tight=alpha.getbbox()
  if tight is None:raise RuntimeError(f'Foreground extraction returned an empty mask for {name}')
  return Cutout(name,image.crop(tight),alpha.crop(tight),tuple(map(int,tight)))

def load_segmenter(args):
 if args.mask_backend=='difference':return DifferenceCutout(),'difference'
 try:return SAM(args.sam_model_id,args.sam_device),'sam2'
 except ModuleNotFoundError:
  if args.mask_backend=='sam2':raise
  warnings.warn('SAM2 is unavailable; using the plain-background difference cutout fallback. Install SAM2 for more precise boundaries.')
  return DifferenceCutout(),'difference'

def place_cutout(cutout,box,size,scale):
 x0,y0,x1,y1=box;tw=max(1,int((x1-x0)*scale));th=max(1,int((y1-y0)*scale));ratio=min(tw/cutout.rgb.width,th/cutout.rgb.height)
 nw=max(1,round(cutout.rgb.width*ratio));nh=max(1,round(cutout.rgb.height*ratio));rgb=cutout.rgb.resize((nw,nh),Image.Resampling.LANCZOS);alpha=cutout.alpha.resize((nw,nh),Image.Resampling.LANCZOS)
 px=round((x0+x1-nw)/2);py=y1-nh;px=max(0,min(size[0]-nw,px));py=max(0,min(size[1]-nh,py))
 canvas=Image.new('RGB',size);mask=Image.new('L',size);canvas.paste(rgb,(px,py));mask.paste(alpha,(px,py));return canvas,mask,(px,py,px+nw,py+nh)

def composite(scene,object_canvas,mask):return Image.composite(object_canvas,scene,mask)

def protect_scene(previous,qwen,object_canvas,object_mask,args):
 object_zone=dilate(object_mask,args.boundary_px);halo=dilate(object_zone,args.interaction_px)
 blend=halo.filter(ImageFilter.GaussianBlur(args.feather_px));result=Image.composite(qwen,previous,blend)
 if args.preserve_reference_core:
  core=erode(object_mask,args.core_erode_px);preserved=Image.composite(object_canvas,previous,core);result=Image.composite(preserved,result,core)
 return result,object_zone,halo

def normalized_box(values,size):
 if len(values)!=4:raise ValueError(f'Placement needs four values, got {values}')
 if all(0<=float(v)<=1 for v in values):return tuple(round(float(v)*s) for v,s in zip(values,(size[0],size[1],size[0],size[1])))
 return tuple(map(int,values))

def latent_image(result):
 """Return the generated latent tensor across Diffusers output variants."""
 value=result.images if hasattr(result,'images') else result[0]
 if isinstance(value,(list,tuple)):value=value[0]
 return value.detach().float().cpu()

def spatial_energy(delta,width,height):
 """Map Qwen generated-image latent tokens back to their spatial grid."""
 import torch
 while delta.ndim>0 and delta.shape[0]==1:delta=delta[0]
 if delta.ndim==3: # C,H,W or H,W,C
  if delta.shape[0]<delta.shape[-1]:energy=delta.square().mean(0)
  else:energy=delta.square().mean(-1)
 elif delta.ndim==2:
  # Qwen packs 2x2 VAE patches. Infer the grid from output aspect ratio.
  n=delta.shape[0];ratio=width/height;gh=max(1,round(math.sqrt(n/ratio)));gw=n//gh
  if gh*gw!=n:
   side=round(math.sqrt(n))
   if side*side!=n:raise RuntimeError(f'Cannot infer spatial grid from latent shape {tuple(delta.shape)}')
   gh=gw=side
  energy=delta.square().mean(-1).reshape(gh,gw)
 else:raise RuntimeError(f'Unsupported Qwen latent shape {tuple(delta.shape)}')
 energy=energy-energy.min();energy=energy/(energy.max().clamp_min(1e-8))
 return energy.numpy()

def largest_component(binary,energy):
 """Highest-energy 8-connected component without an extra scipy dependency."""
 h,w=binary.shape;seen=np.zeros_like(binary,bool);best=None
 for y in range(h):
  for x in range(w):
   if not binary[y,x] or seen[y,x]:continue
   stack=[(y,x)];seen[y,x]=True;points=[]
   while stack:
    yy,xx=stack.pop();points.append((yy,xx))
    for dy in (-1,0,1):
     for dx in (-1,0,1):
      ny,nx=yy+dy,xx+dx
      if 0<=ny<h and 0<=nx<w and binary[ny,nx] and not seen[ny,nx]:seen[ny,nx]=True;stack.append((ny,nx))
   score=sum(float(energy[yy,xx]) for yy,xx in points)*math.sqrt(len(points))
   if best is None or score>best[0]:best=(score,points)
 return None if best is None else best[1]

def box_from_heatmap(heat,name,cutout,size,args,occupied=None):
 smooth=np.asarray(Image.fromarray(np.uint8(heat*255)).filter(ImageFilter.GaussianBlur(args.probe_blur)),dtype=np.float32)/255
 if occupied is not None:
  blocked=np.asarray(dilate(occupied,args.occupancy_margin).resize((smooth.shape[1],smooth.shape[0]),Image.Resampling.BILINEAR),dtype=np.float32)/255
  smooth=smooth*(1-np.clip(blocked,0,1))
 threshold=float(np.quantile(smooth,args.probe_quantile));component=largest_component(smooth>=threshold,smooth)
 if not component:raise RuntimeError(f'No causal placement region found for {name}')
 ys=np.asarray([p[0] for p in component]);xs=np.asarray([p[1] for p in component]);gh,gw=smooth.shape
 weights=np.asarray([smooth[y,x] for y,x in component]);cx=float(np.average(xs+.5,weights=weights))/gw;cy=float(np.average(ys+.5,weights=weights))/gh
 prior=args.default_object_height;prior=DEFAULT_HEIGHT_PRIORS.get(name,prior)
 if args.object_height_priors:
  custom=json.load(open(args.object_height_priors,encoding='utf8'));prior=float(custom.get(name,prior))
 aspect=cutout.rgb.width/max(1,cutout.rgb.height);bh=round(prior*size[1]);bw=round(bh*aspect)
 # Heatmap gives the semantic center; support-biased objects are bottom anchored
 # slightly below that center without assuming a hand-written location.
 x0=round(cx*size[0]-bw/2);y0=round(cy*size[1]-bh/2);x0=max(args.box_margin,min(size[0]-args.box_margin-bw,x0));y0=max(args.box_margin,min(size[1]-args.box_margin-bh,y0))
 return (x0,y0,x0+bw,y0+bh),smooth

def save_heatmap(scene,heat,box,path):
 color=np.zeros((*heat.shape,3),dtype=np.uint8);color[...,0]=np.uint8(255*heat);color[...,1]=np.uint8(100*np.sqrt(heat))
 layer=Image.fromarray(color).resize(scene.size,Image.Resampling.BILINEAR);overlay=Image.blend(scene,layer,.38);draw=ImageDraw.Draw(overlay);draw.rectangle(box,outline=(0,255,80),width=5);overlay.save(path)

def probe_placement(pipe,scene,name,cutout,occupied,args,seed,path):
 """Counterfactual denoising: localize where the insertion instruction acts."""
 common=dict(image=[scene],negative_prompt=args.negative_prompt,true_cfg_scale=args.true_cfg_scale,
  guidance_scale=1.0,num_inference_steps=args.probe_steps,width=args.width,height=args.height,output_type='latent')
 add=(f'Add exactly one complete {name} at the most physically plausible unoccupied location in this scene. '
      'Respect support surfaces, perspective, scale and existing objects. Preserve the rest of the scene.')
 keep='Preserve this scene exactly. Do not add, remove, move or alter any object.'
 a=latent_image(pipe(prompt=add,generator=make_generator(args.device,seed),**common))
 b=latent_image(pipe(prompt=keep,generator=make_generator(args.device,seed),**common))
 heat=spatial_energy(a-b,args.width,args.height);box,heat=box_from_heatmap(heat,name,cutout,scene.size,args,occupied);save_heatmap(scene,heat,box,path)
 return box,{'latent_shape':list(a.shape),'heat_min':float(heat.min()),'heat_max':float(heat.max()),'box':list(box)}

def generate_setup(pipe,args,setup_dir):
 """Create E2's own references and base scene when no E1 cache is supplied."""
 objects_dir=setup_dir/'objects';objects_dir.mkdir(parents=True,exist_ok=True)
 objects=[]
 print('\n=== E2 SETUP: SKETCH -> PHOTOREALISTIC REFERENCES ===')
 for index,(name,path) in enumerate(tqdm(discover_sketches(args.sketch_dir),desc='Generating E2 references',unit='object'),1):
  sketch=fit(Image.open(path),(args.width,args.height))
  prompt=(f'Image 1 is a sketch of a {name}. Convert it into one photorealistic {name}. '
          'Preserve its geometry, pose, proportions, viewpoint and visible components. Use realistic materials. '
          'Show the complete object centered on a plain clean white background with no other objects, labels or scenery.')
  image=infer(pipe,[sketch],prompt,args,args.seed+index*1000);target=objects_dir/f'{index:02d}_{name}.png';image.save(target)
  objects.append({'name':name,'sketch':str(path),'image':str(target),'seed':args.seed+index*1000})
 save_json(objects,setup_dir/'objects.json')
 print('\n=== E2 SETUP: GENERATING BASE SCENE ===')
 blank=Image.new('RGB',(args.width,args.height),'white')
 base=infer(pipe,[blank],'Replace the blank Image 1 with this scene: '+args.base_prompt+' Do not leave a white border or blank canvas.',args,args.seed+50000)
 base.save(setup_dir/'base.png')
 return base,objects

def main():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 p.add_argument('--e1_dir',default='results/qwen_e1_baseline',help='Optional E1 cache; E2 generates its own setup if these files are absent');p.add_argument('--sketch_dir',default='KontextPipeline/sketch');p.add_argument('--placements',default=str(Path(__file__).with_name('e2_placements.json')));p.add_argument('--out_dir',default='results/qwen_e2_sam_collage')
 p.add_argument('--base_prompt',default='Create a photorealistic empty modern living room with warm neutral walls, wooden floor, natural daylight, realistic perspective, and several physically plausible open areas for furniture and objects. Keep the room uncluttered and completely empty.')
 p.add_argument('--placement_backend',choices=('denoise_delta','manual'),default='denoise_delta');p.add_argument('--probe_steps',type=int,default=4);p.add_argument('--probe_quantile',type=float,default=.88);p.add_argument('--probe_blur',type=float,default=1.2);p.add_argument('--box_margin',type=int,default=24);p.add_argument('--occupancy_margin',type=int,default=24);p.add_argument('--default_object_height',type=float,default=.25);p.add_argument('--object_height_priors',default=None,help='Optional JSON mapping object names to image-height fractions')
 p.add_argument('--model_id',default='Qwen/Qwen-Image-Edit-2509');p.add_argument('--lightning_repo',default='lightx2v/Qwen-Image-Lightning');p.add_argument('--lightning_weight',default='Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors');p.add_argument('--lora_scale',type=float,default=1)
 p.add_argument('--mask_backend',choices=('auto','sam2','difference'),default='auto');p.add_argument('--sam_model_id',default='facebook/sam2-hiera-small');p.add_argument('--sam_device',default='cpu');p.add_argument('--device',default='cuda');p.add_argument('--width',type=int,default=1024);p.add_argument('--height',type=int,default=1024);p.add_argument('--steps',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--true_cfg_scale',type=float,default=1);p.add_argument('--negative_prompt',default=' ')
 p.add_argument('--background_threshold',type=float,default=24);p.add_argument('--object_scale',type=float,default=.92);p.add_argument('--boundary_px',type=int,default=12);p.add_argument('--interaction_px',type=int,default=28);p.add_argument('--feather_px',type=float,default=5);p.add_argument('--preserve_reference_core',action='store_true');p.add_argument('--core_erode_px',type=int,default=3);args=p.parse_args()
 out=Path(args.out_dir);cutout_dir=out/'cutouts';step_dir=out/'steps';cutout_dir.mkdir(parents=True,exist_ok=True);step_dir.mkdir(exist_ok=True);save_json(vars(args),out/'config.json')
 e1=Path(args.e1_dir);base_path=e1/'base.png';objects_path=e1/'objects.json';size=(args.width,args.height)
 # Resolve segmentation first so a missing optional dependency never appears
 # after the expensive Qwen load.
 loading=tqdm(total=2,desc='Loading segmentation',unit='model');sam,active_mask_backend=load_segmenter(args);loading.update();loading.set_description('Loading Qwen');pipe=load_pipe(args);loading.update();loading.close()
 if base_path.is_file() and objects_path.is_file():
  print(f'Reusing E1 setup from {e1}');objects=json.load(open(objects_path,encoding='utf8'));current=Image.open(base_path).convert('RGB').resize(size)
 else:
  setup_dir=out/'setup';current,objects=generate_setup(pipe,args,setup_dir);current=current.resize(size)
 placements=json.load(open(args.placements,encoding='utf8')) if args.placement_backend=='manual' else {};current.save(out/'base.png')
 cutouts={}
 for item in tqdm(objects,desc=f'{active_mask_backend} reference cutouts',unit='object'):
  reference=Image.open(item['image']).convert('RGB').resize(size);cut=sam.cutout(item['name'],reference,args.background_threshold);cut.rgb.save(cutout_dir/f"{item['name']}_rgb.png");cut.alpha.save(cutout_dir/f"{item['name']}_alpha.png");cutouts[item['name']]=cut
 history=[];occupied=Image.new('L',size)
 for index,item in enumerate(tqdm(objects,desc='Collage and repaint',unit='object'),1):
  name=item['name']
  probe_info=None
  if args.placement_backend=='manual':
   if name not in placements:raise KeyError(f'No placement configured for {name}')
   box=normalized_box(placements[name],size)
  else:
   box,probe_info=probe_placement(pipe,current,name,cutouts[name],occupied,args,args.seed+500000+index*1000,step_dir/f'{index:02d}_{name}_placement_heatmap.png');placements[name]=[v/s for v,s in zip(box,(size[0],size[1],size[0],size[1]))];save_json(placements,out/'auto_placements.json')
  object_canvas,mask,placed_box=place_cutout(cutouts[name],box,size,args.object_scale)
  collage=composite(current,object_canvas,mask);collage.save(step_dir/f'{index:02d}_{name}_collage.png');mask.save(step_dir/f'{index:02d}_{name}_mask.png')
  prompt=(f'Image 1 is a room containing one newly pasted {name} already at its required location. Harmonize that pasted {name} into the room without moving, removing, duplicating or redesigning it. Preserve its identity, colors, materials, structure, pose and proportions. Correct only its boundary, local perspective, lighting, contact shadow and physical integration. Preserve every other pixel and every previously existing object. Do not create a collage, reference panel, border or split image.')
  raw=infer(pipe,[collage],prompt,args,args.seed+index*1000);raw.save(step_dir/f'{index:02d}_{name}_raw_qwen.png')
  current,zone,halo=protect_scene(current,raw,object_canvas,mask,args);zone.save(step_dir/f'{index:02d}_{name}_object_zone.png');halo.save(step_dir/f'{index:02d}_{name}_interaction_halo.png');current.save(step_dir/f'{index:02d}_{name}_final.png')
  occupied=ImageChops.lighter(occupied,mask)
  history.append({'step':index,'name':name,'mask_backend':active_mask_backend,'placement_backend':args.placement_backend,'configured_box':box,'placed_box':placed_box,'source_box':cutouts[name].source_box,'probe':probe_info})
 current.save(out/'FINAL.png');save_json(history,out/'history.json');print('Done:',out/'FINAL.png')
if __name__=='__main__':main()
