"""E4 Lab: 3D analysis of placement and reference-identity trajectories.

This is a diagnostic, not an editing method. It compares four matched Qwen
trajectories: base preservation, prompt-only generic placement, isolated
reference encoding, and native multi-image reference replacement.

Outputs include shared randomized-PCA features, spatial identity heatmaps,
pre-RoPE query/reference-key affinity, optional x/y/depth scene plots, and
machine-readable metrics. Target: diffusers==0.40.0.
"""
from __future__ import annotations
import argparse,gc,json,math,warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from tqdm.auto import tqdm
import diffusers
import e3_prompt_suite as e3
from e1_baseline import fit,infer,load_pipe,make_generator,save_json

HERE=Path(__file__).resolve().parent

def call(pipe,images,prompt,args,seed,output_type='pil'):
 images=images if isinstance(images,list) else [images]
 result=pipe(image=images,prompt=prompt,negative_prompt=args.negative_prompt if args.true_cfg_scale>1 else None,
  true_cfg_scale=args.true_cfg_scale,num_inference_steps=args.steps,width=args.width,height=args.height,
  generator=make_generator(args.device,seed),output_type=output_type)
 return result.images if hasattr(result,'images') else result[0]

def first_image(value):
 value=value[0] if isinstance(value,(list,tuple)) else value
 return value.convert('RGB') if isinstance(value,Image.Image) else value

def grid_shape(args):return args.height//16,args.width//16

def sample_indices(total,count,seed):
 if count>=total:return np.arange(total,dtype=np.int64)
 return np.sort(np.random.default_rng(seed).choice(total,count,replace=False)).astype(np.int64)

class ActivationRecorder:
 def __init__(self,pipe,args):
  self.args=args;self.layers=set(args.layers);self.blocks=list(pipe.transformer.transformer_blocks);gh,gw=grid_shape(args);self.grid=(gh,gw);self.output_tokens=gh*gw
  self.indices=sample_indices(self.output_tokens,args.tokens_per_snapshot,args.seed);self.segment='output';self.trajectory='';self.records=[];self.handles=[];self.calls={}
 def _hook(self,layer):
  def hook(_module,_inputs,output):
   values=list(output) if isinstance(output,tuple) else [output];x=values[1] if len(values)>1 and torch.is_tensor(values[1]) and values[1].ndim==3 else None
   if x is None:return output
   invocation=self.calls.get(layer,0);self.calls[layer]=invocation+1;start=0 if self.segment=='output' else self.output_tokens
   if x.shape[1]<start+self.output_tokens:return output
   ids=torch.as_tensor(self.indices+start,device=x.device);features=x[0].index_select(0,ids).detach().to('cpu',dtype=torch.float16).numpy()
   self.records.append({'trajectory':self.trajectory,'layer':layer,'step':invocation,'segment':self.segment,'indices':self.indices.copy(),'features':features})
   return output
  return hook
 def install(self):self.handles=[self.blocks[i].register_forward_hook(self._hook(i)) for i in sorted(self.layers) if i<len(self.blocks)]
 def begin(self,name,segment):self.trajectory=name;self.segment=segment;self.calls={}
 def close(self):
  for h in self.handles:h.remove()

class QKAffinityRecorder:
 """Pre-RoPE affinity from replacement output queries to Image-2 keys."""
 def __init__(self,pipe,args):
  self.args=args;self.layers=set(args.layers);self.blocks=list(pipe.transformer.transformer_blocks);self.output_tokens=np.prod(grid_shape(args));self.pending={};self.calls={};self.records=[];self.handles=[];self.enabled=False
 def q_hook(self,layer):
  def hook(_m,_a,out):
   if self.enabled:self.pending[layer]=out.detach()
   return out
  return hook
 def k_hook(self,layer):
  def hook(_m,_a,out):
   if not self.enabled or layer not in self.pending:return out
   q=self.pending.pop(layer);k=out;step=self.calls.get(layer,0);self.calls[layer]=step+1;n=int(self.output_tokens)
   if q.ndim!=3 or k.ndim!=3 or q.shape[1]<n or k.shape[1]<3*n:return out
   qidx=torch.linspace(0,n-1,min(self.args.affinity_queries,n),device=q.device).long();kidx=torch.linspace(2*n,3*n-1,min(self.args.affinity_keys,n),device=k.device).long()
   qs=F.normalize(q[0].index_select(0,qidx).float(),dim=-1);ks=F.normalize(k[0].index_select(0,kidx).float(),dim=-1);logits=qs@ks.T/math.sqrt(qs.shape[-1]);weights=logits.softmax(-1)
   self.records.append({'layer':layer,'step':step,'mean_max_affinity':float(weights.max(-1).values.mean()),'entropy':float((-(weights*weights.clamp_min(1e-8).log()).sum(-1)).mean()),'query_indices':qidx.cpu().numpy(),'max_affinity':weights.max(-1).values.cpu().numpy()})
   return out
  return hook
 def install(self):
  for layer in sorted(self.layers):
   if layer>=len(self.blocks):continue
   attn=getattr(self.blocks[layer],'attn',None)
   if attn is not None:self.handles.extend([attn.to_q.register_forward_hook(self.q_hook(layer)),attn.to_k.register_forward_hook(self.k_hook(layer))])
 def begin(self):self.enabled=True;self.pending={};self.calls={}
 def end(self):self.enabled=False;self.pending={}
 def close(self):
  for h in self.handles:h.remove()

def randomized_pca(records,projection_dim,seed):
 features=np.concatenate([r['features'].astype(np.float32) for r in records]);d=features.shape[1];rng=np.random.default_rng(seed);projection=rng.normal(0,1/math.sqrt(projection_dim),(d,min(d,projection_dim))).astype(np.float32)
 reduced=features@projection;mean=reduced.mean(0,keepdims=True);centered=reduced-mean;_,_,vt=np.linalg.svd(centered,full_matrices=False);xyz=centered@vt[:3].T
 cursor=0
 for record in records:n=len(record['features']);record['xyz']=xyz[cursor:cursor+n];cursor+=n
 return {'feature_dim':d,'projection_dim':projection.shape[1],'explained_variance':np.var(xyz,axis=0).tolist()}

def reference_centroids(records):
 return {(r['layer'],r['step']):r['features'].astype(np.float32).mean(0) for r in records if r['trajectory']=='reference'}

def identity_metrics(records):
 centers=reference_centroids(records);result=[]
 for r in records:
  if r['trajectory'] not in {'generic','replacement'}:continue
  center=centers.get((r['layer'],r['step']))
  if center is None:continue
  x=r['features'].astype(np.float32);similarity=(x@center)/(np.linalg.norm(x,axis=1)*np.linalg.norm(center)+1e-8);r['identity_similarity']=similarity
  result.append({'trajectory':r['trajectory'],'layer':r['layer'],'step':r['step'],'mean':float(similarity.mean()),'max':float(similarity.max()),'p95':float(np.quantile(similarity,.95))})
 return result

def plotly_module():
 try:import plotly.graph_objects as go;return go
 except ImportError as exc:raise RuntimeError('E4 Lab requires Plotly: `pip install plotly`.') from exc

def write_feature_plot(records,path):
 go=plotly_module();fig=go.Figure();colors={'base':'#4575b4','generic':'#d73027','reference':'#1a9850','replacement':'#984ea3'}
 for trajectory in colors:
  subset=[r for r in records if r['trajectory']==trajectory]
  if not subset:continue
  xyz=np.concatenate([r['xyz'] for r in subset]);labels=np.concatenate([[f'{trajectory} L{r["layer"]} T{r["step"]} token {i}' for i in r['indices']] for r in subset])
  fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',name=trajectory,marker={'size':2,'opacity':.45,'color':colors[trajectory]},text=labels,hoverinfo='text'))
 fig.update_layout(title='Shared 3D feature space',scene={'xaxis_title':'PC1','yaxis_title':'PC2','zaxis_title':'PC3'});fig.write_html(path,include_plotlyjs='cdn')

def write_trajectory_plot(records,path):
 go=plotly_module();fig=go.Figure()
 for trajectory in ('base','generic','reference','replacement'):
  subset=sorted([r for r in records if r['trajectory']==trajectory],key=lambda x:(x['step'],x['layer']));
  if not subset:continue
  xyz=np.asarray([r['xyz'].mean(0) for r in subset]);text=[f'{trajectory} step={r["step"]} layer={r["layer"]}' for r in subset];fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='lines+markers',name=trajectory,text=text,hoverinfo='text'))
 fig.update_layout(title='Feature-centroid trajectories over layer/time');fig.write_html(path,include_plotlyjs='cdn')

def write_affinity_surface(records,path):
 go=plotly_module();layers=sorted({r['layer'] for r in records});steps=sorted({r['step'] for r in records});z=np.full((len(steps),len(layers)),np.nan)
 for r in records:z[steps.index(r['step']),layers.index(r['layer'])]=r['mean_max_affinity']
 fig=go.Figure(data=[go.Surface(x=layers,y=steps,z=z)]);fig.update_layout(title='Pre-RoPE output-query → reference-key affinity',scene={'xaxis_title':'Layer','yaxis_title':'Step','zaxis_title':'Mean max affinity'});fig.write_html(path,include_plotlyjs='cdn')

def save_identity_heatmaps(records,image,out,args):
 gh,gw=grid_shape(args);out.mkdir(parents=True,exist_ok=True)
 for r in records:
  if 'identity_similarity' not in r:continue
  grid=np.full(gh*gw,np.nan,np.float32);grid[r['indices']]=r['identity_similarity'];valid=np.isfinite(grid);fill=float(np.nanmedian(grid)) if valid.any() else 0;grid=np.nan_to_num(grid,nan=fill).reshape(gh,gw);lo,hi=np.quantile(grid,[.05,.95]);norm=np.clip((grid-lo)/(hi-lo+1e-8),0,1)
  color=np.zeros((gh,gw,3),np.uint8);color[...,0]=np.uint8(norm*255);color[...,2]=np.uint8((1-norm)*255);heat=Image.fromarray(color).resize(image.size,Image.Resampling.BILINEAR);Image.blend(image,heat,.45).save(out/f'{r["trajectory"]}_L{r["layer"]:02d}_T{r["step"]:02d}.png')

def depth_map(image,args,estimator=None):
 if args.skip_depth:return np.tile(np.linspace(0,1,image.height)[:,None],(1,image.width))
 try:
  if estimator is None:
   from transformers import pipeline
   estimator=pipeline('depth-estimation',model=args.depth_model,device=0 if args.device.startswith('cuda') else -1)
  result=estimator(image);depth=np.asarray(result['depth'].resize(image.size),np.float32);return (depth-depth.min())/(np.ptp(depth)+1e-8)
 except Exception as exc:warnings.warn(f'Depth estimation failed ({exc}); using image-y proxy.');return np.tile(np.linspace(0,1,image.height)[:,None],(1,image.width))

def write_physical_plot(image,depth,record,path,args):
 go=plotly_module();gh,gw=grid_shape(args);idx=record['indices'];x=idx%gw;y=idx//gw;px=np.minimum(image.width-1,(x+.5)*image.width/gw).astype(int);py=np.minimum(image.height-1,(y+.5)*image.height/gh).astype(int);z=depth[py,px];color=np.linalg.norm(record['features'].astype(np.float32),axis=1)
 fig=go.Figure(data=[go.Scatter3d(x=x,y=y,z=z,mode='markers',marker={'size':3,'color':color,'colorscale':'Turbo','colorbar':{'title':'Feature norm'}})]);fig.update_layout(title='Physical token coordinates (x, y, estimated depth)',scene={'xaxis_title':'Token x','yaxis_title':'Token y','zaxis_title':'Depth'});fig.write_html(path,include_plotlyjs='cdn')

def parser():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter);p.add_argument('--prompts',default=str(HERE/'e3_prompts.json'));p.add_argument('--out_dir',default='results/qwen_e4_3d_lab');p.add_argument('--case_id',type=int,default=1);p.add_argument('--object_index',type=int,default=1)
 p.add_argument('--all_prompts',action='store_true',help='Analyze every object in every prompt-suite case')
 p.add_argument('--model_id',default='Qwen/Qwen-Image-Edit-2509');p.add_argument('--lightning_repo',default='lightx2v/Qwen-Image-Lightning');p.add_argument('--lightning_weight',default='Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors');p.add_argument('--lora_scale',type=float,default=1);p.add_argument('--device',default='cuda');p.add_argument('--width',type=int,default=1024);p.add_argument('--height',type=int,default=1024);p.add_argument('--steps',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--object_seed',type=int,default=1337);p.add_argument('--true_cfg_scale',type=float,default=1);p.add_argument('--negative_prompt',default=' ');p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--missing_policy',choices=('skip','error'),default='error');p.add_argument('--max_objects',type=int)
 p.add_argument('--layers',type=int,nargs='+',default=[10,20,30,40,50,59]);p.add_argument('--tokens_per_snapshot',type=int,default=128);p.add_argument('--projection_dim',type=int,default=64);p.add_argument('--affinity_queries',type=int,default=128);p.add_argument('--affinity_keys',type=int,default=128);p.add_argument('--depth_model',default='depth-anything/Depth-Anything-V2-Small-hf');p.add_argument('--skip_depth',action='store_true');return p.parse_args()

def safe_slug(value):
 return ''.join(c if c.isalnum() else '_' for c in str(value).lower()).strip('_') or 'object'

def analyze_object(pipe,base,depth,case,item,reference,args,out,seed):
 """Run the four matched trajectories for one object against a fixed base."""
 out.mkdir(parents=True,exist_ok=True)
 generic_prompt=f'Add exactly one complete generic {item["name"]} naturally in the most plausible location. Preserve the scene.'
 replacement_prompt=f'Image 1 contains a generic {item["name"]}. Image 2 is the exact reference. Replace only the generic object with the Image 2 identity while preserving its location and the scene.'
 base.save(out/'base.png');reference.save(out/'reference.png')
 recorder=ActivationRecorder(pipe,args);affinity=QKAffinityRecorder(pipe,args);recorder.install();affinity.install();matched='Preserve Image 1 exactly without changing anything.'
 try:
  recorder.begin('base','output');call(pipe,base,matched,args,seed+1000,'latent')
  # Capture the generic trajectory during its actual generation. Previously E4
  # generated it once and then spent a duplicate pass tracing a preservation edit.
  recorder.begin('generic','output');generic=first_image(call(pipe,base,generic_prompt,args,seed+200,'pil'))
  recorder.begin('reference','condition');call(pipe,reference,matched,args,seed+1000,'latent')
  recorder.begin('replacement','output');affinity.begin();replacement=first_image(call(pipe,[generic,reference],replacement_prompt,args,seed+1000,'pil'));affinity.end()
 finally:recorder.close();affinity.close()
 generic.save(out/'generic.png');replacement.save(out/'replacement.png');pca=randomized_pca(recorder.records,args.projection_dim,seed);identity=identity_metrics(recorder.records);write_feature_plot(recorder.records,out/'feature_pca_3d.html');write_trajectory_plot(recorder.records,out/'token_trajectories_3d.html');write_affinity_surface(affinity.records,out/'reference_affinity_surface.html');save_identity_heatmaps(recorder.records,replacement,out/'identity_heatmaps',args);physical=next(r for r in recorder.records if r['trajectory']=='generic');write_physical_plot(base,depth,physical,out/'physical_scene_3d.html',args)
 compact=[{'trajectory':r['trajectory'],'layer':r['layer'],'step':r['step'],'segment':r['segment']} for r in recorder.records];metrics={'case_id':int(case['id']),'object':item['name'],'pca':pca,'identity_similarity':identity,'qk_affinity':[{k:v for k,v in r.items() if k not in {'query_indices','max_affinity'}} for r in affinity.records],'captures':compact};save_json(metrics,out/'metrics.json')
 archive={}
 for r in recorder.records:
  key=f'{r["trajectory"]}_L{r["layer"]}_T{r["step"]}';archive[key+'_features']=r['features'];archive[key+'_pca_xyz']=r['xyz'];archive[key+'_token_indices']=r['indices']
 np.savez_compressed(out/'features_and_coordinates.npz',**archive)
 return {'case_id':int(case['id']),'object':item['name'],'output':str(out),'status':'complete','identity_measurements':len(identity),'affinity_measurements':len(affinity.records)}

def main():
 args=parser();
 if diffusers.__version__!='0.40.0':warnings.warn(f'E4 Lab targets diffusers 0.40.0; found {diffusers.__version__}')
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);pf=Path(args.prompts).resolve();all_cases=e3.load_suite(pf)
 if args.all_prompts:cases=all_cases
 else:
  case=next((c for c in all_cases if int(c['id'])==args.case_id),None)
  if case is None:raise ValueError(f'Unknown case id {args.case_id}')
  if not 1<=args.object_index<=len(case['objects']):raise ValueError('object_index is one-based and outside this case')
  cases=[{'id':case['id'],'base_prompt':case['base_prompt'],'objects':[case['objects'][args.object_index-1]]}]
 save_json(vars(args),out/'config.json');total=sum(len(c['objects']) for c in cases);print(f'E4 will analyze {len(cases)} case(s), {total} object(s), with one model load.')
 pipe=load_pipe(args);refs=e3.generate_references(pipe,cases,args,out,pf);depth_estimator=None
 if not args.skip_depth:
  from transformers import pipeline
  depth_estimator=pipeline('depth-estimation',model=args.depth_model,device=0 if args.device.startswith('cuda') else -1)
 summary=[];progress=tqdm(total=total,desc='E4 object analyses',unit='object')
 for case in cases:
  case_out=out/f'case_{int(case["id"]):03d}';case_out.mkdir(parents=True,exist_ok=True);base_path=case_out/'base.png';case_seed=args.seed+int(case['id'])*10000
  if args.resume and base_path.is_file():base=fit(Image.open(base_path),(args.width,args.height))
  else:
   blank=Image.new('RGB',(args.width,args.height),'white');base=infer(pipe,[blank],'Replace blank Image 1 with: '+case['base_prompt'],args,case_seed+100);base.save(base_path)
  depth=depth_map(base,args,depth_estimator)
  for object_index,item in enumerate(case['objects'],1):
   object_out=case_out/f'object_{object_index:02d}_{safe_slug(item["name"])}';metrics_path=object_out/'metrics.json'
   if args.resume and metrics_path.is_file() and (object_out/'replacement.png').is_file():
    summary.append({'case_id':int(case['id']),'object':item['name'],'output':str(object_out),'status':'resumed'});progress.update();continue
   record=refs[e3.reference_key(item)]
   if record.get('status')!='ready':
    if args.missing_policy=='skip':summary.append({'case_id':int(case['id']),'object':item['name'],'status':'missing_reference'});progress.update();continue
    raise FileNotFoundError(record)
   reference=fit(Image.open(record['image']),(args.width,args.height));summary.append(analyze_object(pipe,base,depth,case,item,reference,args,object_out,case_seed+object_index*100));save_json({'requested':total,'results':summary},out/'summary.json');gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None;progress.update()
 progress.close();save_json({'requested':total,'completed':sum(x['status'] in {'complete','resumed'} for x in summary),'results':summary},out/'summary.json');print('Done:',out)
if __name__=='__main__':main()
