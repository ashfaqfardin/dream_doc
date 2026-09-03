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
   ids=torch.as_tensor(self.indices+start,device=x.device);features=x[0].index_select(0,ids).detach().to('cpu',dtype=torch.float32).numpy()
   self.records.append({'trajectory':self.trajectory,'layer':layer,'step':invocation,'segment':self.segment,'indices':self.indices.copy(),'features':features})
   return output
  return hook
 def install(self):self.handles=[self.blocks[i].register_forward_hook(self._hook(i)) for i in sorted(self.layers) if i<len(self.blocks)]
 def begin(self,name,segment):self.trajectory=name;self.segment=segment;self.calls={}
 def close(self):
  for h in self.handles:h.remove()

class QKAffinityRecorder:
 """Post-normalization, post-RoPE output-to-condition routing probe.

 Mass is normalized over scene and reference image keys only. It is a
 scene-vs-reference routing diagnostic, not the model's full joint attention
 denominator (which additionally contains text and noisy-output keys).
 """
 def __init__(self,pipe,args):
  self.args=args;self.layers=set(args.layers);self.blocks=list(pipe.transformer.transformer_blocks);self.output_tokens=int(np.prod(grid_shape(args)));self.pending={};self.context={};self.calls={};self.records=[];self.handles=[];self.enabled=False
 def pre_hook(self,layer):
  def hook(_m,_a,kwargs):
   if self.enabled:self.context[layer]=kwargs.get('image_rotary_emb')
  return hook
 def q_hook(self,layer):
  def hook(_m,_a,out):
   if self.enabled:self.pending[layer]=out.detach()
   return out
  return hook
 def k_hook(self,layer):
  def hook(_m,_a,out):
   if not self.enabled or layer not in self.pending:return out
   q=self.pending.pop(layer);k=out;step=self.calls.get(layer,0);self.calls[layer]=step+1;n=self.output_tokens
   if q.ndim!=3 or k.ndim!=3 or q.shape[1]<n or k.shape[1]<3*n:return out
   attn=self.blocks[layer].attn;head_dim=attn.inner_dim//attn.heads;q=q.unflatten(-1,(-1,head_dim));k=k.unflatten(-1,(-1,head_dim))
   if attn.norm_q is not None:q=attn.norm_q(q)
   if attn.norm_k is not None:k=attn.norm_k(k)
   rope=self.context.get(layer)
   if rope is not None:
    from diffusers.models.transformers.transformer_qwenimage import ROPE_PER_DEVICE
    apply_rope=ROPE_PER_DEVICE.get(q.device.type,ROPE_PER_DEVICE['cuda']);q=apply_rope(q,rope[0]);k=apply_rope(k,rope[0])
   qidx=torch.linspace(0,n-1,min(self.args.affinity_queries,n),device=q.device).round().long();qs=q[0].index_select(0,qidx);scene=k[0,n:2*n];reference=k[0,2*n:3*n]
   scene_logits=torch.einsum('qhd,khd->hqk',qs.float(),scene.float())/math.sqrt(head_dim);ref_logits=torch.einsum('qhd,khd->hqk',qs.float(),reference.float())/math.sqrt(head_dim);joint=torch.cat([scene_logits,ref_logits],dim=-1).softmax(-1);scene_mass=joint[...,:n].sum(-1).mean(0);reference_mass=joint[...,n:].sum(-1).mean(0);ratio=reference_mass/(scene_mass+reference_mass+1e-8)
   self.records.append({'layer':layer,'step':step,'query_indices':qidx.cpu().numpy(),'scene_mass':scene_mass.detach().cpu().numpy(),'reference_mass':reference_mass.detach().cpu().numpy(),'reference_ratio':ratio.detach().cpu().numpy(),'mean_scene_mass':float(scene_mass.mean()),'mean_reference_mass':float(reference_mass.mean()),'mean_reference_ratio':float(ratio.mean())})
   return out
  return hook
 def install(self):
  for layer in sorted(self.layers):
   if layer>=len(self.blocks):continue
   attn=getattr(self.blocks[layer],'attn',None)
   if attn is not None:self.handles.extend([attn.register_forward_pre_hook(self.pre_hook(layer),with_kwargs=True),attn.to_q.register_forward_hook(self.q_hook(layer)),attn.to_k.register_forward_hook(self.k_hook(layer))])
 def begin(self):self.enabled=True;self.pending={};self.context={};self.calls={}
 def end(self):self.enabled=False;self.pending={};self.context={}
 def close(self):
  for h in self.handles:h.remove()

def randomized_pca(records,projection_dim,seed):
 if not records:raise RuntimeError('No transformer activations were captured; cannot compute PCA.')
 normalized=[];nonfinite=0
 for r in records:
  x=r['features'].astype(np.float32,copy=False);nonfinite+=int((~np.isfinite(x)).sum());x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0);x=x-x.mean(0,keepdims=True);x=x/(np.sqrt(np.mean(x*x))+1e-8);normalized.append(x)
 features=np.concatenate(normalized)
 # Normalize only the numerical scale, jointly across all trajectories. This
 # preserves their relative geometry while preventing projection overflow.
 finite_abs=np.abs(features);scale=float(np.quantile(finite_abs,.999)) if finite_abs.size else 1.0
 if not np.isfinite(scale) or scale<1e-8:scale=1.0
 features=np.clip(features/scale,-1.0,1.0);d=features.shape[1];target=min(d,projection_dim);rng=np.random.default_rng(seed);projection=rng.normal(0,1/math.sqrt(target),(d,target)).astype(np.float32)
 reduced=features@projection;reduced=np.nan_to_num(reduced,nan=0.0,posinf=0.0,neginf=0.0);mean=reduced.mean(0,keepdims=True);centered=reduced-mean
 try:_,_,vt=np.linalg.svd(centered,full_matrices=False)
 except np.linalg.LinAlgError:
  # Symmetric eigendecomposition is a robust fallback for rare LAPACK SVD failures.
  covariance=centered.T@centered/max(1,len(centered)-1);values,vectors=np.linalg.eigh(covariance);vt=vectors[:,np.argsort(values)[::-1]].T
 xyz=centered@vt[:3].T
 cursor=0
 for record in records:n=len(record['features']);record['xyz']=xyz[cursor:cursor+n];cursor+=n
 return {'feature_dim':d,'projection_dim':projection.shape[1],'explained_variance':np.var(xyz,axis=0).tolist(),'nonfinite_values_replaced':nonfinite,'robust_scale_p999':scale}

def reference_centroids(records):
 return {(r['layer'],r['step']):r['features'].astype(np.float32).mean(0) for r in records if r['trajectory']=='reference'}

def identity_metrics(records):
 centers=reference_centroids(records);result=[]
 for r in records:
  if r['trajectory'] not in {'generic','replacement'}:continue
  center=centers.get((r['layer'],r['step']))
  if center is None:continue
  x=np.nan_to_num(r['features'].astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0);center=np.nan_to_num(center,nan=0.0,posinf=0.0,neginf=0.0);similarity=(x@center)/(np.linalg.norm(x,axis=1)*np.linalg.norm(center)+1e-8);similarity=np.nan_to_num(similarity,nan=0.0,posinf=0.0,neginf=0.0);r['identity_similarity']=similarity
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
 for r in records:z[steps.index(r['step']),layers.index(r['layer'])]=r['mean_reference_ratio']
 fig=go.Figure(data=[go.Surface(x=layers,y=steps,z=z,cmin=0,cmax=1)]);fig.update_layout(title='Post-RoPE conditional-image routing',scene={'xaxis_title':'Layer','yaxis_title':'Step','zaxis_title':'Reference / (scene + reference)'});fig.write_html(path,include_plotlyjs='cdn')

def save_identity_heatmaps(records,image,out,args):
 gh,gw=grid_shape(args);out.mkdir(parents=True,exist_ok=True)
 for r in records:
  if 'identity_similarity' not in r:continue
  grid=np.full(gh*gw,np.nan,np.float32);grid[r['indices']]=r['identity_similarity'];valid=np.isfinite(grid);fill=float(np.nanmedian(grid)) if valid.any() else 0;grid=np.nan_to_num(grid,nan=fill).reshape(gh,gw);lo,hi=np.quantile(grid,[.05,.95]);norm=np.clip((grid-lo)/(hi-lo+1e-8),0,1)
  color=np.zeros((gh,gw,3),np.uint8);color[...,0]=np.uint8(norm*255);color[...,2]=np.uint8((1-norm)*255);heat=Image.fromarray(color).resize(image.size,Image.Resampling.BILINEAR);Image.blend(image,heat,.45).save(out/f'{r["trajectory"]}_L{r["layer"]:02d}_T{r["step"]:02d}.png')

def save_routing_heatmaps(records,image,out,args):
 gh,gw=grid_shape(args);out.mkdir(parents=True,exist_ok=True)
 for r in records:
  grid=np.full(gh*gw,np.nan,np.float32);grid[r['query_indices']]=r['reference_ratio'];known=np.flatnonzero(np.isfinite(grid));missing=np.flatnonzero(~np.isfinite(grid))
  if missing.size and known.size:
   ky,kx=np.divmod(known,gw);my,mx=np.divmod(missing,gw);nearest=((my[:,None]-ky[None,:])**2+(mx[:,None]-kx[None,:])**2).argmin(1);grid[missing]=grid[known[nearest]]
  grid=np.nan_to_num(grid,nan=.5).reshape(gh,gw);color=np.zeros((gh,gw,3),np.uint8);color[...,0]=np.uint8(np.clip(grid,0,1)*255);color[...,2]=np.uint8(np.clip(1-grid,0,1)*255);heat=Image.fromarray(color).resize(image.size,Image.Resampling.BILINEAR);Image.blend(image,heat,.45).save(out/f'routing_L{r["layer"]:02d}_T{r["step"]:02d}.png')

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
 p.add_argument('--model_id',default='Qwen/Qwen-Image-Edit-2509');p.add_argument('--lightning_repo',default='lightx2v/Qwen-Image-Lightning');p.add_argument('--lightning_weight',default='Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors');p.add_argument('--lora_scale',type=float,default=1);p.add_argument('--device',default='cuda');p.add_argument('--width',type=int,default=1024);p.add_argument('--height',type=int,default=1024);p.add_argument('--steps',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--object_seed',type=int,default=1337);p.add_argument('--true_cfg_scale',type=float,default=1);p.add_argument('--negative_prompt',default=' ');p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--missing_policy',choices=('skip','error'),default='error');p.add_argument('--max_cases',type=int);p.add_argument('--max_objects',type=int)
 p.add_argument('--layers',type=int,nargs='+',default=[10,20,30,40,50]);p.add_argument('--tokens_per_snapshot',type=int,default=128);p.add_argument('--projection_dim',type=int,default=64);p.add_argument('--affinity_queries',type=int,default=256);p.add_argument('--affinity_keys',type=int,default=128,help='Deprecated; corrected routing uses every scene/reference key');p.add_argument('--depth_model',default='depth-anything/Depth-Anything-V2-Small-hf');p.add_argument('--skip_depth',action='store_true');return p.parse_args()

def safe_slug(value):
 return ''.join(c if c.isalnum() else '_' for c in str(value).lower()).strip('_') or 'object'

def image_role_scores(scene,reference,replacement):
 def vector(image):
  return np.asarray(image.resize((256,256),Image.Resampling.BILINEAR),np.float32).reshape(-1)/255
 s,r,o=vector(scene),vector(reference),vector(replacement)
 def compare(a,b):
  return {'rmse':float(np.sqrt(np.mean((a-b)**2))),'cosine':float((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))}
 return {'replacement_to_scene':compare(o,s),'replacement_to_reference':compare(o,r),'interpretation':'Lower RMSE and higher cosine indicate greater global resemblance; this is a collapse diagnostic, not an object-identity metric.'}

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
 if not affinity.records:raise RuntimeError('No scene/reference routing records were captured. Verify Diffusers 0.40.0 and square 1024x1024 condition-image token packing.')
 generic.save(out/'generic.png');replacement.save(out/'replacement.png');pca=randomized_pca(recorder.records,args.projection_dim,seed);identity=identity_metrics(recorder.records);write_feature_plot(recorder.records,out/'feature_pca_3d.html');write_trajectory_plot(recorder.records,out/'token_trajectories_3d.html');write_affinity_surface(affinity.records,out/'reference_affinity_surface.html');save_identity_heatmaps([r for r in recorder.records if r['trajectory']=='generic'],generic,out/'identity_heatmaps'/'generic',args);save_identity_heatmaps([r for r in recorder.records if r['trajectory']=='replacement'],replacement,out/'identity_heatmaps'/'replacement',args);save_routing_heatmaps(affinity.records,replacement,out/'routing_heatmaps',args);physical=next(r for r in recorder.records if r['trajectory']=='generic');write_physical_plot(base,depth,physical,out/'physical_scene_3d.html',args)
 compact=[{'trajectory':r['trajectory'],'layer':r['layer'],'step':r['step'],'segment':r['segment']} for r in recorder.records];routing=[{k:v for k,v in r.items() if k not in {'query_indices','scene_mass','reference_mass','reference_ratio'}} for r in affinity.records];metrics={'schema_version':2,'case_id':int(case['id']),'object':item['name'],'pca':pca,'identity_similarity':identity,'conditional_image_routing':routing,'routing_definition':'Post-QK-normalization and post-RoPE attention, normalized over scene and reference image keys only.','image_role_scores':image_role_scores(base,reference,replacement),'captures':compact};save_json(metrics,out/'metrics.json')
 archive={}
 for r in recorder.records:
  key=f'{r["trajectory"]}_L{r["layer"]}_T{r["step"]}';archive[key+'_features']=r['features'];archive[key+'_pca_xyz']=r['xyz'];archive[key+'_token_indices']=r['indices']
 np.savez_compressed(out/'features_and_coordinates.npz',**archive)
 return {'case_id':int(case['id']),'object':item['name'],'output':str(out),'status':'complete','identity_measurements':len(identity),'affinity_measurements':len(affinity.records)}

def main():
 args=parser();
 if diffusers.__version__!='0.40.0':warnings.warn(f'E4 Lab targets diffusers 0.40.0; found {diffusers.__version__}')
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);pf=Path(args.prompts).resolve();all_cases=e3.load_suite(pf)
 if args.all_prompts:
  cases=all_cases[:args.max_cases] if args.max_cases else all_cases
  if args.max_objects:cases=[{**case,'objects':case['objects'][:args.max_objects]} for case in cases]
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
   valid_resume=False
   if args.resume and metrics_path.is_file() and (object_out/'replacement.png').is_file():
    try:valid_resume=json.loads(metrics_path.read_text(encoding='utf-8')).get('schema_version',0)>=2
    except (OSError,json.JSONDecodeError):valid_resume=False
   if valid_resume:
    summary.append({'case_id':int(case['id']),'object':item['name'],'output':str(object_out),'status':'resumed'});progress.update();continue
   record=refs[e3.reference_key(item)]
   if record.get('status')!='ready':
    if args.missing_policy=='skip':summary.append({'case_id':int(case['id']),'object':item['name'],'status':'missing_reference'});progress.update();continue
    raise FileNotFoundError(record)
   reference=fit(Image.open(record['image']),(args.width,args.height));summary.append(analyze_object(pipe,base,depth,case,item,reference,args,object_out,case_seed+object_index*100));save_json({'requested':total,'results':summary},out/'summary.json');gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None;progress.update()
 progress.close();save_json({'requested':total,'completed':sum(x['status'] in {'complete','resumed'} for x in summary),'results':summary},out/'summary.json');print('Done:',out)
if __name__=='__main__':main()
