"""E19: sequential generic placement -> SAM -> isolated-reference inpaint."""
from __future__ import annotations
import argparse,os,warnings,logging
from dataclasses import asdict
import numpy as np
from PIL import Image,ImageFilter
from tqdm.auto import tqdm
from e16_reference_composite_harmonizer import load_jobs,flip_reference,placement_prompt
from e15_generic_place_then_replace_pipeline import (DEFAULT_DETECTOR,DEFAULT_KONTEXT_MODEL,
 GenericDetector,SAM2BoxSegmenter,ensure_dir,generator_for,image_difference_map,
 load_planner_pipe,load_inpaint_pipe_from_planner,make_overlay,mask_area_fraction,
 mask_bbox,protect_outside_mask,save_json)

def dilate(mask,r):
 return mask.convert('L') if r<=0 else mask.convert('L').filter(ImageFilter.MaxFilter(2*r+1))

def writable(image):
 """Give torchvision an owned, writable NumPy backing buffer."""
 return Image.fromarray(np.array(image.convert('RGB'),dtype=np.uint8,copy=True))

def place(planner,current,job,detector,sam,args,seed,directory):
 short,long=placement_prompt(job);errors=[]
 for attempt in range(1,args.placement_attempts+1):
  s=seed+101*attempt
  x1=planner(image=current,prompt=short,prompt_2=long,width=args.width,height=args.height,
   max_area=args.width*args.height,num_inference_steps=args.placement_steps,
   guidance_scale=args.placement_guidance_scale,generator=generator_for(args.device,s)).images[0].convert('RGB')
  x1.save(os.path.join(directory,f'01_x1_attempt_{attempt}.png'))
  found=detector.detect_all(writable(x1),job.name,args.detection_threshold)
  if not found: errors.append(f'attempt {attempt}: no detection');continue
  evidence=image_difference_map(current,x1,args.diff_blur_px)
  # Select the detection that lies in the newly generated region. Confidence
  # alone repeatedly selects a previously present, easy-to-detect bicycle.
  h,w=evidence.shape
  def changed_score(d):
   x0,y0,x1b,y1b=[int(round(v)) for v in d.box];x0=max(0,min(w,x0));x1b=max(0,min(w,x1b));y0=max(0,min(h,y0));y1b=max(0,min(h,y1b))
   changed=float(evidence[y0:y1b,x0:x1b].mean()) if x1b>x0 and y1b>y0 else 0.0
   return changed+0.05*float(d.score)
  detection=max(found,key=changed_score)
  mask=sam.segment(x1,detection.box,evidence);box=mask_bbox(mask);area=mask_area_fraction(mask)
  if box is None or not args.min_object_area<=area<=args.max_object_area:
   errors.append(f'attempt {attempt}: invalid SAM area {area:.3f}');continue
  return x1,mask,box,s,attempt,float(detection.score)
 raise RuntimeError(f"No valid x1 for {job.name}: "+'; '.join(errors))

def isolate(reference,job,detector,sam,args):
 found=detector.detect_all(writable(reference),job.name,args.detection_threshold)
 if not found:raise RuntimeError(f"No {job.name} detected in reference")
 mask=sam.segment(reference,found[0].box,np.ones((reference.height,reference.width),np.float32));box=mask_bbox(mask)
 if box is None:raise RuntimeError(f"Empty reference SAM mask for {job.name}")
 crop=reference.crop(box);alpha=mask.crop(box);clean=Image.new('RGB',crop.size,'white');clean.paste(crop,(0,0),alpha)
 return clean,alpha,box

def segment_generated(image,job,detector,sam,evidence_source,args):
 found=detector.detect_all(writable(image),job.name,args.detection_threshold)
 if not found:return None
 evidence=image_difference_map(evidence_source,image,args.diff_blur_px)
 detection=max(found,key=lambda d:d.score)
 return sam.segment(image,detection.box,evidence)

def final_composite(current,raw,pre_mask,job,detector,sam,args,directory):
 """Protect the scene using the *result* silhouette, not the obsolete x1 silhouette."""
 if args.final_composite_mode=='raw':
  return raw,Image.new('L',raw.size,255)
 found=detector.detect_all(raw,job.name,args.detection_threshold)
 if not found:
  if args.post_sam_fallback=='raw':
   return raw,Image.new('L',raw.size,255)
  warnings.warn(f"Post-inpaint detector found no {job.name}; using pre-inpaint mask")
  post=pre_mask
 else:
  evidence=image_difference_map(current,raw,args.diff_blur_px)
  post=sam.segment(raw,max(found,key=lambda d:d.score).box,evidence)
 # Keep both the original editable region and every newly formed object part.
 union=np.maximum(np.asarray(pre_mask.convert('L')),np.asarray(post.convert('L')))
 union=Image.fromarray(union.astype(np.uint8))
 union=dilate(union,args.final_mask_dilate_px)
 union.save(os.path.join(directory,'08_final_post_sam_union_mask.png'))
 make_overlay(raw,union).save(os.path.join(directory,'08_final_post_sam_union_overlay.png'))
 blend=union.filter(ImageFilter.GaussianBlur(args.final_mask_feather_px))
 return protect_outside_mask(current,raw,blend),union

def parse_args():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 p.add_argument('--base_image',required=True);p.add_argument('--objects_json',required=True);p.add_argument('--out_dir',default='results/e19_sam_reference_inpaint')
 p.add_argument('--kontext_model_id',default=DEFAULT_KONTEXT_MODEL);p.add_argument('--detector_model_id',default=DEFAULT_DETECTOR);p.add_argument('--sam2_model_id',default='facebook/sam2-hiera-small')
 p.add_argument('--device',default='cuda');p.add_argument('--detector_device',default='cpu');p.add_argument('--sam2_device',default='cpu');p.add_argument('--torch_dtype',default='bfloat16',choices=['float16','bfloat16','float32']);p.add_argument('--cpu_offload',action='store_true');p.add_argument('--share_pipeline_components',action=argparse.BooleanOptionalAction,default=True)
 p.add_argument('--width',type=int,default=1024);p.add_argument('--height',type=int,default=1024);p.add_argument('--seed',type=int,default=42);p.add_argument('--placement_attempts',type=int,default=2);p.add_argument('--placement_steps',type=int,default=12);p.add_argument('--placement_guidance_scale',type=float,default=2.5)
 p.add_argument('--inpaint_steps',type=int,default=18);p.add_argument('--inpaint_guidance_scale',type=float,default=2.5);p.add_argument('--inpaint_strength',type=float,default=.88);p.add_argument('--second_inpaint_steps',type=int,default=12);p.add_argument('--second_inpaint_strength',type=float,default=.62);p.add_argument('--detection_threshold',type=float,default=.2);p.add_argument('--diff_blur_px',type=float,default=3);p.add_argument('--min_object_area',type=float,default=.003);p.add_argument('--max_object_area',type=float,default=.35);p.add_argument('--initial_mask_dilate_px',type=int,default=12);p.add_argument('--completion_mask_dilate_px',type=int,default=14);p.add_argument('--mask_feather_px',type=float,default=6)
 p.add_argument('--final_composite_mode',choices=['post_sam','raw'],default='post_sam',help='Protect the scene using the completed object silhouette')
 p.add_argument('--post_sam_fallback',choices=['raw','pre_mask'],default='raw');p.add_argument('--final_mask_dilate_px',type=int,default=12);p.add_argument('--final_mask_feather_px',type=float,default=4);return p.parse_args()

def main():
 warnings.filterwarnings('ignore',message='The given NumPy array is not writable.*',category=UserWarning,module='torchvision.*')
 warnings.filterwarnings('ignore',message='There are modules in .* that should be kept in float32.*',category=UserWarning)
 class EmptyKeepFP32Filter(logging.Filter):
  def filter(self,record):
   message=record.getMessage()
   return not ('should be kept in float32: []' in message and 'Casting directly with `to()`' in message)
 fp32_filter=EmptyKeepFP32Filter()
 logging.getLogger('diffusers').addFilter(fp32_filter)
 logging.getLogger('diffusers.pipelines.pipeline_utils').addFilter(fp32_filter)
 for handler in logging.getLogger().handlers:handler.addFilter(fp32_filter)
 args=parse_args();ensure_dir(args.out_dir);save_json(vars(args),os.path.join(args.out_dir,'config_e19.json'));items=load_jobs(args.objects_json)
 base=Image.open(args.base_image).convert('RGB').resize((args.width,args.height));current=base.copy();base.save(os.path.join(args.out_dir,'base.png'))
 # GroundingDINO and SAM2 use lazy constructors. Force their real loads here;
 # otherwise they unexpectedly download during the first object's mask stage.
 loading=tqdm(total=4,desc='Preparing GroundingDINO',unit='stage',dynamic_ncols=True)
 detector=GenericDetector(args.detector_model_id,args.detector_device);detector._load();loading.update()
 loading.set_description('Preparing SAM2');sam=SAM2BoxSegmenter(args.sam2_model_id,args.sam2_device);sam._load();loading.update()
 loading.set_description('Loading Kontext');planner=load_planner_pipe(args);loading.update()
 loading.set_description('Building inpaint pipeline');inpainter=load_inpaint_pipe_from_planner(planner,args);loading.update()
 loading.set_description('E19 models ready');loading.close();summary=[]
 print('=== E19: X1 -> SAM -> ISOLATED REFERENCE INPAINT ===')
 for i,job in enumerate(tqdm(items,desc='E19 objects',unit='object'),1):
  stages=tqdm(total=6,desc=f'{job.name}: placement',unit='stage',leave=False,dynamic_ncols=True)
  directory=os.path.join(args.out_dir,f"step_{i:02d}_{job.name.replace(' ','_')}");ensure_dir(directory);current.save(os.path.join(directory,'00_before.png'))
  # Each placement/mask probe is base + this object only. Previous edits are
  # deliberately excluded so SAM cannot select an earlier object.
  x1,mask,box,seed,attempt,score=place(planner,base,job,detector,sam,args,args.seed+i*10000,directory);x1.save(os.path.join(directory,'01_x1.png'));mask.save(os.path.join(directory,'02_x1_sam.png'));make_overlay(x1,mask).save(os.path.join(directory,'02_x1_overlay.png'));stages.update()
  stages.set_description(f'{job.name}: reference SAM')
  reference=flip_reference(Image.open(job.reference).convert('RGB'),job.reference_flip);clean,refmask,refbox=isolate(reference,job,detector,sam,args);clean.save(os.path.join(directory,'03_reference_no_background.png'));refmask.save(os.path.join(directory,'03_reference_mask.png'))
  stages.update();stages.set_description(f'{job.name}: first inpaint')
  editable=dilate(mask,args.initial_mask_dilate_px);editable.save(os.path.join(directory,'04_first_inpaint_mask.png'));make_overlay(x1,editable).save(os.path.join(directory,'04_first_inpaint_overlay.png'))
  detail=(f'Replace the masked generic {job.name} with the exact reference {job.name}. Transfer its colors, materials, component shapes, proportions and structural details. Preserve the x1 location, scale and pose. Form one complete coherent object with realistic contact, lighting and occlusion. Do not copy the reference background or change anything outside the mask.')
  raw=inpainter(image=x1,mask_image=editable,image_reference=clean,prompt=f'Replace the masked {job.name} with the reference {job.name}.',prompt_2=detail,strength=args.inpaint_strength,width=args.width,height=args.height,max_area=args.width*args.height,num_inference_steps=args.inpaint_steps,guidance_scale=args.inpaint_guidance_scale,generator=generator_for(args.device,seed+5000)).images[0].convert('RGB')
  raw.save(os.path.join(directory,'05_first_reference_inpaint.png'))
  stages.update();stages.set_description(f'{job.name}: refinement SAM')
  post_mask=segment_generated(raw,job,detector,sam,x1,args)
  if post_mask is None:
   warnings.warn(f'Could not segment first-pass {job.name}; retaining the first mask')
   post_mask=editable
  union=np.maximum(np.asarray(editable),np.asarray(post_mask)).astype(np.uint8)
  completion=dilate(Image.fromarray(union),args.completion_mask_dilate_px)
  post_mask.save(os.path.join(directory,'06_first_inpaint_sam.png'));completion.save(os.path.join(directory,'06_completion_inpaint_mask.png'));make_overlay(raw,completion).save(os.path.join(directory,'06_completion_overlay.png'))
  stages.update();stages.set_description(f'{job.name}: completion inpaint')
  completion_prompt=(f'Complete and repair the masked reference {job.name}. Restore every missing or disconnected structural part, preserve the exact reference colors and design, and make one fully formed physically coherent object. Keep its established pose, scale and location. Change nothing outside the mask.')
  completed=inpainter(image=raw,mask_image=completion,image_reference=clean,prompt=f'Complete the reference {job.name}.',prompt_2=completion_prompt,strength=args.second_inpaint_strength,width=args.width,height=args.height,max_area=args.width*args.height,num_inference_steps=args.second_inpaint_steps,guidance_scale=args.inpaint_guidance_scale,generator=generator_for(args.device,seed+7000)).images[0].convert('RGB')
  completed.save(os.path.join(directory,'07_second_completion_inpaint.png'))
  stages.update();stages.set_description(f'{job.name}: final SAM/composite')
  final,final_mask=final_composite(current,completed,completion,job,detector,sam,args,directory)
  final.save(os.path.join(directory,'09_final.png'));current=final;stages.update();stages.close()
  record={'step':i,'object':asdict(job),'x1_box':box,'x1_mask_area':mask_area_fraction(mask),'reference_box':refbox,'seed':seed,'attempt':attempt,'detection_score':score};save_json(record,os.path.join(directory,'summary.json'));summary.append(record);print(f'[{i}/{len(items)}] {job.name}: {box}')
 current.save(os.path.join(args.out_dir,'FINAL.png'));save_json(summary,os.path.join(args.out_dir,'summary_e19.json'))
if __name__=='__main__':main()
