"""E19: sequential generic placement -> SAM -> isolated-reference inpaint."""
from __future__ import annotations
import argparse,os
from dataclasses import asdict
import numpy as np
from PIL import Image,ImageFilter
from e16_reference_composite_harmonizer import load_jobs,flip_reference,placement_prompt
from e15_generic_place_then_replace_pipeline import (DEFAULT_DETECTOR,DEFAULT_KONTEXT_MODEL,
 GenericDetector,SAM2BoxSegmenter,ensure_dir,generator_for,image_difference_map,
 load_planner_pipe,load_inpaint_pipe_from_planner,make_overlay,mask_area_fraction,
 mask_bbox,protect_outside_mask,save_json)

def dilate(mask,r):
 return mask.convert('L') if r<=0 else mask.convert('L').filter(ImageFilter.MaxFilter(2*r+1))

def place(planner,current,job,detector,sam,args,seed,directory):
 short,long=placement_prompt(job);errors=[]
 for attempt in range(1,args.placement_attempts+1):
  s=seed+101*attempt
  x1=planner(image=current,prompt=short,prompt_2=long,width=args.width,height=args.height,
   max_area=args.width*args.height,num_inference_steps=args.placement_steps,
   guidance_scale=args.placement_guidance_scale,generator=generator_for(args.device,s)).images[0].convert('RGB')
  x1.save(os.path.join(directory,f'01_x1_attempt_{attempt}.png'))
  found=detector.detect_all(x1,job.name,args.detection_threshold)
  if not found: errors.append(f'attempt {attempt}: no detection');continue
  detection=max(found,key=lambda d:d.score);evidence=image_difference_map(current,x1,args.diff_blur_px)
  mask=sam.segment(x1,detection.box,evidence);box=mask_bbox(mask);area=mask_area_fraction(mask)
  if box is None or not args.min_object_area<=area<=args.max_object_area:
   errors.append(f'attempt {attempt}: invalid SAM area {area:.3f}');continue
  return x1,mask,box,s,attempt,float(detection.score)
 raise RuntimeError(f"No valid x1 for {job.name}: "+'; '.join(errors))

def isolate(reference,job,detector,sam,args):
 found=detector.detect_all(reference,job.name,args.detection_threshold)
 if not found:raise RuntimeError(f"No {job.name} detected in reference")
 mask=sam.segment(reference,found[0].box,np.ones((reference.height,reference.width),np.float32));box=mask_bbox(mask)
 if box is None:raise RuntimeError(f"Empty reference SAM mask for {job.name}")
 crop=reference.crop(box);alpha=mask.crop(box);clean=Image.new('RGB',crop.size,'white');clean.paste(crop,(0,0),alpha)
 return clean,alpha,box

def parse_args():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 p.add_argument('--base_image',required=True);p.add_argument('--objects_json',required=True);p.add_argument('--out_dir',default='results/e19_sam_reference_inpaint')
 p.add_argument('--kontext_model_id',default=DEFAULT_KONTEXT_MODEL);p.add_argument('--detector_model_id',default=DEFAULT_DETECTOR);p.add_argument('--sam2_model_id',default='facebook/sam2-hiera-small')
 p.add_argument('--device',default='cuda');p.add_argument('--detector_device',default='cpu');p.add_argument('--sam2_device',default='cpu');p.add_argument('--torch_dtype',default='bfloat16',choices=['float16','bfloat16','float32']);p.add_argument('--cpu_offload',action='store_true');p.add_argument('--share_pipeline_components',action=argparse.BooleanOptionalAction,default=True)
 p.add_argument('--width',type=int,default=1024);p.add_argument('--height',type=int,default=1024);p.add_argument('--seed',type=int,default=42);p.add_argument('--placement_attempts',type=int,default=3);p.add_argument('--placement_steps',type=int,default=16);p.add_argument('--placement_guidance_scale',type=float,default=2.5)
 p.add_argument('--inpaint_steps',type=int,default=28);p.add_argument('--inpaint_guidance_scale',type=float,default=2.5);p.add_argument('--inpaint_strength',type=float,default=.88);p.add_argument('--detection_threshold',type=float,default=.2);p.add_argument('--diff_blur_px',type=float,default=3);p.add_argument('--min_object_area',type=float,default=.003);p.add_argument('--max_object_area',type=float,default=.35);p.add_argument('--mask_dilate_px',type=int,default=16);p.add_argument('--mask_feather_px',type=float,default=6);return p.parse_args()

def main():
 args=parse_args();ensure_dir(args.out_dir);save_json(vars(args),os.path.join(args.out_dir,'config_e19.json'));items=load_jobs(args.objects_json)
 current=Image.open(args.base_image).convert('RGB').resize((args.width,args.height));current.save(os.path.join(args.out_dir,'base.png'))
 detector=GenericDetector(args.detector_model_id,args.detector_device);sam=SAM2BoxSegmenter(args.sam2_model_id,args.sam2_device);planner=load_planner_pipe(args);inpainter=load_inpaint_pipe_from_planner(planner,args);summary=[]
 print('=== E19: X1 -> SAM -> ISOLATED REFERENCE INPAINT ===')
 for i,job in enumerate(items,1):
  directory=os.path.join(args.out_dir,f"step_{i:02d}_{job.name.replace(' ','_')}");ensure_dir(directory);current.save(os.path.join(directory,'00_before.png'))
  x1,mask,box,seed,attempt,score=place(planner,current,job,detector,sam,args,args.seed+i*10000,directory);x1.save(os.path.join(directory,'01_x1.png'));mask.save(os.path.join(directory,'02_x1_sam.png'));make_overlay(x1,mask).save(os.path.join(directory,'02_x1_overlay.png'))
  reference=flip_reference(Image.open(job.reference).convert('RGB'),job.reference_flip);clean,refmask,refbox=isolate(reference,job,detector,sam,args);clean.save(os.path.join(directory,'03_reference_no_background.png'));refmask.save(os.path.join(directory,'03_reference_mask.png'))
  editable=dilate(mask,args.mask_dilate_px);soft=editable.filter(ImageFilter.GaussianBlur(args.mask_feather_px));editable.save(os.path.join(directory,'04_inpaint_mask.png'))
  detail=(f'Replace the masked generic {job.name} with the exact reference {job.name}. Transfer its colors, materials, component shapes, proportions and structural details. Preserve the x1 location, scale and pose. Form one complete coherent object with realistic contact, lighting and occlusion. Do not copy the reference background or change anything outside the mask.')
  raw=inpainter(image=x1,mask_image=editable,image_reference=clean,prompt=f'Replace the masked {job.name} with the reference {job.name}.',prompt_2=detail,strength=args.inpaint_strength,width=args.width,height=args.height,max_area=args.width*args.height,num_inference_steps=args.inpaint_steps,guidance_scale=args.inpaint_guidance_scale,generator=generator_for(args.device,seed+5000)).images[0].convert('RGB')
  raw.save(os.path.join(directory,'05_raw_reference_inpaint.png'));final=protect_outside_mask(current,raw,soft);final.save(os.path.join(directory,'06_final.png'));current=final
  record={'step':i,'object':asdict(job),'x1_box':box,'x1_mask_area':mask_area_fraction(mask),'reference_box':refbox,'seed':seed,'attempt':attempt,'detection_score':score};save_json(record,os.path.join(directory,'summary.json'));summary.append(record);print(f'[{i}/{len(items)}] {job.name}: {box}')
 current.save(os.path.join(args.out_dir,'FINAL.png'));save_json(summary,os.path.join(args.out_dir,'summary_e19.json'))
if __name__=='__main__':main()
