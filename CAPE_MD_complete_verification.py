# CAPE-MD complete Colab verification
# Run in Google Colab. The notebook version is generated from this script.

# %% [markdown]
# CAPE-MD: controlled synthetic verification and separate rMD17 analysis.

# %%
# Colab installation (comment out outside Colab)
import sys, subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','torch-geometric','pandas','scipy','scikit-learn','tqdm','tabulate'])

# %%
import os, gc, json, math, time, random, shutil, warnings, zipfile, platform
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from IPython.display import display, Markdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class Config:
    seed:int=20260718
    fast_mode:bool=True
    run_real_data:bool=True
    train_ensemble:bool=True
    save_checkpoints:bool=True
    syn_n_atoms:int=6
    syn_n_samples_fast:int=1400
    syn_n_samples_full:int=5000
    syn_noise_position:float=0.10
    real_dataset_name:str='revised aspirin'
    real_n_train_fast:int=400
    real_n_val_fast:int=100
    real_n_cal_fast:int=100
    real_n_test_fast:int=160
    real_n_train_full:int=1000
    real_n_val_full:int=250
    real_n_cal_full:int=250
    real_n_test_full:int=400
    real_spacing:int=10
    real_gap:int=200
    hidden_dim:int=80
    n_rbf:int=24
    cutoff_syn:float=4.5
    cutoff_real:float=5.0
    n_layers:int=3
    batch_size_syn:int=64
    batch_size_real:int=16
    epochs_syn_fast:int=30
    epochs_syn_full:int=100
    epochs_real_fast:int=20
    epochs_real_full:int=70
    lr:float=2e-3
    weight_decay:float=1e-6
    patience:int=12
    grad_clip:float=10.0
    lambda_energy:float=1.0
    lambda_force:float=20.0
    lambda_curvature:float=0.20
    lambda_rollout:float=0.01
    curvature_batch_pairs:int=6
    curvature_every:int=2
    pair_max_distance_quantile:float=0.25
    rollout_train_every:int=4
    rollout_train_steps:int=3
    rollout_eval_steps_syn:int=80
    dt_syn:float=0.004
    rollout_temperature:float=0.12
    rollout_seeds:int=8
    ensemble_size_fast:int=3
    ensemble_size_full:int=5
    alpha:float=0.10
    n_bootstrap:int=500
CFG=Config()
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE=torch.float32
RESULTS=Path('CAPE_MD_RESULTS'); FIG_DIR=RESULTS/'figures'; TABLE_DIR=RESULTS/'tables'; MODEL_DIR=RESULTS/'models'; DATA_DIR=RESULTS/'data'
for d in [RESULTS,FIG_DIR,TABLE_DIR,MODEL_DIR,DATA_DIR]: d.mkdir(parents=True,exist_ok=True)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(CFG.seed)
with open(RESULTS/'config.json','w') as f: json.dump(asdict(CFG),f,indent=2)
print('Device:',DEVICE,'PyTorch:',torch.__version__,'Fast mode:',CFG.fast_mode)

PALETTE={'DirectForceNet':'#D55E00','ConservativeNet':'#0072B2','CAPE-MD':'#009E73','CAPE-Ensemble':'#CC79A7'}
def save_show(fig,name):
    p=FIG_DIR/name; fig.savefig(p,dpi=220,bbox_inches='tight',facecolor='white'); plt.show(); plt.close(fig); print('Saved:',p)
def save_table(df,name,title):
    p=TABLE_DIR/name; df.to_csv(p,index=False); display(Markdown('### '+title)); display(df.style.format(precision=6).hide(axis='index')); print('Saved:',p)
def bootstrap_ci(x,n_boot=None,alpha=.05):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)==0:return np.nan,np.nan,np.nan
    rng=np.random.default_rng(CFG.seed); n_boot=n_boot or CFG.n_bootstrap
    vals=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(n_boot)])
    return x.mean(),np.quantile(vals,alpha/2),np.quantile(vals,1-alpha/2)
def truth_report(df):
    display(Markdown('### Empirical interpretation'))
    for metric in df.metric.unique():
        p=df[df.metric==metric].dropna(subset=['mean'])
        if len(p):
            b=p.loc[p['mean'].idxmin()]
            print(f"• {metric}: lowest observed value = {b.model} ({b['mean']:.6g}).")
    print('These conclusions apply only to this run, split, seed, architecture and budget.')

# %%
# Synthetic differentiable molecular potential

def angle_value(a,b,c,eps=1e-9):
    u=a-b; v=c-b
    co=(u*v).sum(-1)/(u.norm(dim=-1)*v.norm(dim=-1)+eps)
    return torch.acos(torch.clamp(co,-1+1e-6,1-1e-6))

def synthetic_potential(pos):
    single=pos.ndim==2
    if single: pos=pos.unsqueeze(0)
    B,N,_=pos.shape
    r0=torch.tensor([1.00,1.08,.95,1.12,1.02],device=pos.device,dtype=pos.dtype)
    kb=torch.tensor([34.,28.,38.,25.,31.],device=pos.device,dtype=pos.dtype)
    e=torch.zeros(B,device=pos.device,dtype=pos.dtype)
    for k in range(N-1):
        r=(pos[:,k]-pos[:,k+1]).norm(dim=-1); e+=.5*kb[k]*(r-r0[k])**2
    th0=torch.tensor([1.90,2.05,1.85,2.10],device=pos.device,dtype=pos.dtype)
    ka=torch.tensor([4.,3.5,4.5,3.8],device=pos.device,dtype=pos.dtype)
    for k,(i,j,l) in enumerate([(0,1,2),(1,2,3),(2,3,4),(3,4,5)]):
        th=angle_value(pos[:,i],pos[:,j],pos[:,l]); e+=.5*ka[k]*(th-th0[k])**2
    bonded={(i,i+1) for i in range(N-1)}
    for i in range(N):
        for j in range(i+1,N):
            if (i,j) in bonded or abs(i-j)==2: continue
            r=(pos[:,i]-pos[:,j]).norm(dim=-1).clamp_min(.65); sr6=(.9/r)**6
            e+=4*.12*(sr6**2-sr6)
    cen=pos-pos.mean(1,keepdim=True); e+=.005*(cen**2).sum((1,2))
    return e[0] if single else e

def energy_force(pos):
    x=pos.detach().clone().requires_grad_(True); e=synthetic_potential(x); f=-torch.autograd.grad(e.sum(),x)[0]
    return e.detach(),f.detach()
def reference_geometry(n=6):
    return torch.tensor([[.95*i,.22*((-1)**i),.10*np.sin(i)] for i in range(n)],dtype=DTYPE)
def generate_synthetic(n,seed):
    g=torch.Generator().manual_seed(seed); base=reference_geometry(CFG.syn_n_atoms)
    centres=[]
    for _ in range(8):
        q=.08*torch.randn(base.shape,generator=g); q-=q.mean(0,keepdim=True); centres.append(base+q)
    centres=torch.stack(centres); ids=torch.randint(0,len(centres),(n,),generator=g)
    pos=centres[ids]+CFG.syn_noise_position*torch.randn(n,CFG.syn_n_atoms,3,generator=g)
    pos-=pos.mean(1,keepdim=True)
    valid=torch.ones(n,dtype=torch.bool)
    for i in range(CFG.syn_n_atoms):
        for j in range(i+1,CFG.syn_n_atoms): valid&=((pos[:,i]-pos[:,j]).norm(dim=-1)>.55)
    pos=pos[valid]; e,f=energy_force(pos); z=torch.tensor([6,6,8,6,7,1],dtype=torch.long)
    return {'pos':pos,'energy':e,'force':f,'z':z}
n_syn=CFG.syn_n_samples_fast if CFG.fast_mode else CFG.syn_n_samples_full
syn=generate_synthetic(n_syn,CFG.seed)
print('Synthetic shapes:',{k:tuple(v.shape) for k,v in syn.items()})

class MolecularDataset(Dataset):
    def __init__(self,p,e,f): self.p=p; self.e=e; self.f=f
    def __len__(self): return len(self.p)
    def __getitem__(self,i): return self.p[i],self.e[i],self.f[i]
def independent_split(n,seed):
    idx=np.random.default_rng(seed).permutation(n); a=int(.60*n); b=a+int(.15*n); c=b+int(.10*n)
    return {'train':idx[:a],'val':idx[a:b],'cal':idx[b:c],'test':idx[c:]}
class Scaler:
    def fit(self,e,f): self.em=e.mean(); self.es=e.std().clamp_min(1e-6); self.fs=f.std().clamp_min(1e-6); return self
def loader(data,idx,bs,shuffle):
    idx=torch.as_tensor(idx); return DataLoader(MolecularDataset(data['pos'][idx],data['energy'][idx],data['force'][idx]),batch_size=bs,shuffle=shuffle)
def close_pairs(pos,idx,max_pairs=1200,seed=0):
    rng=np.random.default_rng(seed); idx=np.asarray(idx); m=min(max_pairs*8,max(100,len(idx)*10))
    a=rng.choice(idx,m); b=rng.choice(idx,m); keep=a!=b; a,b=a[keep],b[keep]
    d=torch.sqrt(((pos[a]-pos[b])**2).mean((1,2))).numpy(); th=np.quantile(d,CFG.pair_max_distance_quantile)
    s=np.where(d<=th)[0][:max_pairs]; return np.stack([a[s],b[s]],1),th
syn_split=independent_split(len(syn['pos']),CFG.seed)
syn_scaler=Scaler().fit(syn['energy'][syn_split['train']],syn['force'][syn_split['train']])
syn_loaders={k:loader(syn,v,CFG.batch_size_syn,k=='train') for k,v in syn_split.items()}
syn_pairs,_=close_pairs(syn['pos'],syn_split['train'],seed=CFG.seed)
print('Synthetic curvature pairs:',len(syn_pairs))

# %%
# Models
class GaussianRBF(nn.Module):
    def __init__(self,n,cutoff):
        super().__init__(); self.register_buffer('centres',torch.linspace(0,cutoff,n)); self.gamma=10/cutoff**2; self.cutoff=cutoff
    def forward(self,d):
        r=torch.exp(-self.gamma*(d.unsqueeze(-1)-self.centres)**2); env=.5*(torch.cos(math.pi*d/self.cutoff)+1)*(d<self.cutoff)
        return r*env.unsqueeze(-1)
def all_pairs(n,device): return torch.triu_indices(n,n,1,device=device)
class InvariantEnergyNet(nn.Module):
    def __init__(self,max_z,hidden,n_rbf,cutoff,n_layers):
        super().__init__(); self.emb=nn.Embedding(max_z+1,hidden); self.rbf=GaussianRBF(n_rbf,cutoff)
        seq=[nn.Linear(2*hidden+n_rbf,hidden),nn.SiLU()]
        for _ in range(n_layers-1): seq += [nn.Linear(hidden,hidden),nn.SiLU()]
        seq += [nn.Linear(hidden,hidden)]; self.pair=nn.Sequential(*seq)
        self.atom=nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
    def energy(self,pos,z):
        if pos.ndim==2: pos=pos[None]
        B,N,_=pos.shape; zb=z[None].expand(B,-1) if z.ndim==1 else z; h=self.emb(zb); i,j=all_pairs(N,pos.device)
        rij=pos[:,j]-pos[:,i]; d=rij.norm(dim=-1).clamp_min(1e-8); msg=self.pair(torch.cat([h[:,i],h[:,j],self.rbf(d)],-1))
        agg=torch.zeros(B,N,msg.shape[-1],device=pos.device,dtype=pos.dtype); agg.index_add_(1,i,msg); agg.index_add_(1,j,msg)
        return self.atom(torch.cat([h,agg],-1)).squeeze(-1).sum(1)
    def forward(self,pos,z,create_graph=False):
        x=pos.requires_grad_(True); e=self.energy(x,z); f=-torch.autograd.grad(e.sum(),x,create_graph=create_graph,retain_graph=create_graph)[0]; return e,f
class DirectForceNet(nn.Module):
    def __init__(self,max_z,hidden,n_rbf,cutoff,n_layers):
        super().__init__(); self.energy_net=InvariantEnergyNet(max_z,hidden,n_rbf,cutoff,n_layers); self.emb=nn.Embedding(max_z+1,hidden); self.rbf=GaussianRBF(n_rbf,cutoff)
        self.fm=nn.Sequential(nn.Linear(2*hidden+n_rbf,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
    def forward(self,pos,z,create_graph=False):
        if pos.ndim==2: pos=pos[None]
        B,N,_=pos.shape; zb=z[None].expand(B,-1) if z.ndim==1 else z; h=self.emb(zb); e=self.energy_net.energy(pos,z); i,j=all_pairs(N,pos.device)
        rij=pos[:,j]-pos[:,i]; d=rij.norm(dim=-1).clamp_min(1e-8); u=rij/d.unsqueeze(-1); c=self.fm(torch.cat([h[:,i],h[:,j],self.rbf(d)],-1)).squeeze(-1)
        pf=c.unsqueeze(-1)*u; force=torch.zeros_like(pos); force.index_add_(1,i,pf); force.index_add_(1,j,-pf); return e,force
def make_models(cutoff,zmax):
    kw=dict(max_z=zmax,hidden=CFG.hidden_dim,n_rbf=CFG.n_rbf,cutoff=cutoff,n_layers=CFG.n_layers)
    return {'DirectForceNet':DirectForceNet(**kw).to(DEVICE),'ConservativeNet':InvariantEnergyNet(**kw).to(DEVICE),'CAPE-MD':InvariantEnergyNet(**kw).to(DEVICE)}

def hvp(model,pos,z,v,create_graph=True):
    x=pos.detach().clone().requires_grad_(True); e=model.energy(x,z).sum(); g=torch.autograd.grad(e,x,create_graph=True)[0]; return torch.autograd.grad((g*v).sum(),x,create_graph=create_graph)[0]
def curvature_loss(model,pa,pb,fa,fb,z,fs):
    delta=pb-pa; h=hvp(model,.5*(pa+pb),z,delta,True); res=fb-fa+h; den=((fb-fa)**2).sum((1,2)).clamp_min((.05*fs)**2); return (((res**2).sum((1,2)))/den).mean()
def velocity_verlet_model(model,p0,v0,z,dt,steps,create_graph=False):
    p,v=p0,v0; ps=[]; hs=[]
    for _ in range(steps+1):
        x=p.requires_grad_(True); e=model.energy(x,z); f=-torch.autograd.grad(e.sum(),x,create_graph=create_graph,retain_graph=create_graph)[0]
        ps.append(p); hs.append(e+.5*(v**2).sum((1,2)))
        if len(ps)==steps+1: break
        vh=v+.5*dt*f; pn=p+dt*vh; xn=pn.requires_grad_(True); en=model.energy(xn,z); fn=-torch.autograd.grad(en.sum(),xn,create_graph=create_graph,retain_graph=create_graph)[0]
        v=vh+.5*dt*fn; p=pn
    return torch.stack(ps,1),torch.stack(hs,1)
def rollout_loss(model,p,z):
    v=CFG.rollout_temperature*torch.randn_like(p); v-=v.mean(1,keepdim=True); _,h=velocity_verlet_model(model,p,v,z,CFG.dt_syn,CFG.rollout_train_steps,True)
    return ((((h-h[:,[0]])/(h[:,[0]].abs()+1))[:,1:])**2).mean()

# %%
# Training and evaluation

def predict(model,ldr,z):
    model.eval(); out={k:[] for k in ['ep','fp','et','ft','pos']}
    for p,e,f in ldr:
        p=p.to(DEVICE).requires_grad_(True)
        with torch.enable_grad(): ep,fp=model(p,z.to(DEVICE),False)
        for k,v in zip(out,[ep.detach().cpu(),fp.detach().cpu(),e,f,p.detach().cpu()]): out[k].append(v)
    return {k:torch.cat(v) for k,v in out.items()}
def metrics(pr):
    ee=pr['ep']-pr['et']; fe=pr['fp']-pr['ft']; nf=pr['fp'].sum(1); cen=pr['pos']-pr['pos'].mean(1,keepdim=True); tq=torch.cross(cen,pr['fp'],dim=-1).sum(1)
    return {'Energy_MAE':ee.abs().mean().item(),'Energy_RMSE':torch.sqrt((ee**2).mean()).item(),'Force_MAE':fe.abs().mean().item(),'Force_RMSE':torch.sqrt((fe**2).mean()).item(),'NetForce_RMS':torch.sqrt((nf**2).mean()).item(),'Torque_RMS':torch.sqrt((tq**2).mean()).item()}
def sample_pair_batch(data,pairs,n):
    ids=np.random.choice(len(pairs),min(n,len(pairs)),False); p=pairs[ids]; a,b=p[:,0],p[:,1]
    return data['pos'][a].to(DEVICE),data['pos'][b].to(DEVICE),data['force'][a].to(DEVICE),data['force'][b].to(DEVICE)
def train(model,name,ldrs,data,pairs,z,scaler,epochs,prefix,use_curv=False,use_roll=False):
    opt=torch.optim.AdamW(model.parameters(),lr=CFG.lr,weight_decay=CFG.weight_decay); sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.6,patience=4)
    best=None; bestv=1e99; bad=0; hist=[]
    for ep in tqdm(range(1,epochs+1),desc='Training '+name):
        model.train(); vals=[]
        for bi,(p,e,f) in enumerate(ldrs['train']):
            p=p.to(DEVICE).requires_grad_(True); e=e.to(DEVICE); f=f.to(DEVICE); opt.zero_grad(set_to_none=True)
            pe,pf=model(p,z.to(DEVICE),True)
            le=F.smooth_l1_loss((pe-scaler.em.to(DEVICE))/scaler.es.to(DEVICE),(e-scaler.em.to(DEVICE))/scaler.es.to(DEVICE))
            lf=F.smooth_l1_loss(pf/scaler.fs.to(DEVICE),f/scaler.fs.to(DEVICE)); loss=CFG.lambda_energy*le+CFG.lambda_force*lf; lc=torch.tensor(0.,device=DEVICE); ld=torch.tensor(0.,device=DEVICE)
            if use_curv and ep%CFG.curvature_every==0:
                pa,pb,fa,fb=sample_pair_batch(data,pairs,CFG.curvature_batch_pairs); lc=curvature_loss(model,pa,pb,fa,fb,z.to(DEVICE),float(scaler.fs)); loss+=CFG.lambda_curvature*lc
            if use_roll and ep%CFG.rollout_train_every==0 and bi==0:
                ld=rollout_loss(model,p[:min(3,len(p))],z.to(DEVICE)); loss+=CFG.lambda_rollout*ld
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),CFG.grad_clip); opt.step(); vals.append([loss.item(),le.item(),lf.item(),lc.item(),ld.item()])
        vm=metrics(predict(model,ldrs['val'],z)); score=vm['Energy_RMSE']/float(scaler.es)+vm['Force_RMSE']/float(scaler.fs); sch.step(score)
        hist.append({'epoch':ep,'loss':np.mean(vals,0)[0],'energy_loss':np.mean(vals,0)[1],'force_loss':np.mean(vals,0)[2],'curvature_loss':np.mean(vals,0)[3],'rollout_loss':np.mean(vals,0)[4],'val_score':score})
        if score<bestv-1e-7: bestv=score; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=CFG.patience: print('Early stop',name,ep); break
    model.load_state_dict(best); h=pd.DataFrame(hist); h.to_csv(TABLE_DIR/f'{prefix}_{name}_history.csv',index=False)
    if CFG.save_checkpoints: torch.save(best,MODEL_DIR/f'{prefix}_{name}.pt')
    return model,h

def plot_histories(hs,prefix):
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    for n,h in hs.items(): ax.plot(h.epoch,h.val_score,label=n,color=PALETTE.get(n,'#444444'),lw=2)
    ax.set(xlabel='Epoch',ylabel='Validation standardised score',title='Validation learning curves'); ax.grid(alpha=.25); ax.legend(frameon=False); save_show(fig,prefix+'_training.png')

def curvature_eval(models,data,pairs,z):
    rows=[]
    for name,m in models.items():
        em=m if hasattr(m,'energy') else m.energy_net; vals=[]
        for s in range(0,len(pairs),12):
            q=pairs[s:s+12]; pa=data['pos'][q[:,0]].to(DEVICE); pb=data['pos'][q[:,1]].to(DEVICE); fa=data['force'][q[:,0]].to(DEVICE); fb=data['force'][q[:,1]].to(DEVICE)
            with torch.enable_grad(): hh=hvp(em,.5*(pa+pb),z.to(DEVICE),pb-pa,False)
            val=torch.sqrt(((fb-fa+hh)**2).mean((1,2)))/torch.sqrt(((fb-fa)**2).mean((1,2))).clamp_min(1e-6); vals.extend(val.detach().cpu().numpy())
        mean,lo,hi=bootstrap_ci(vals); rows.append({'model':name,'metric':'Relative_Curvature_Error','mean':mean,'ci_low':lo,'ci_high':hi})
    return pd.DataFrame(rows)

# %%
# Synthetic benchmark
syn_models=make_models(CFG.cutoff_syn,int(syn['z'].max())); syn_epochs=CFG.epochs_syn_fast if CFG.fast_mode else CFG.epochs_syn_full; syn_hist={}
for name in ['DirectForceNet','ConservativeNet','CAPE-MD']:
    syn_models[name],syn_hist[name]=train(syn_models[name],name,syn_loaders,syn,syn_pairs,syn['z'],syn_scaler,syn_epochs,'synthetic',name=='CAPE-MD',name=='CAPE-MD')
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
plot_histories(syn_hist,'synthetic')
rows=[]; syn_preds={}
for name,m in syn_models.items():
    pr=predict(m,syn_loaders['test'],syn['z']); syn_preds[name]=pr
    for k,v in metrics(pr).items(): rows.append({'model':name,'metric':k,'mean':v})
syn_metric_df=pd.DataFrame(rows); save_table(syn_metric_df.pivot(index='model',columns='metric',values='mean').reset_index(),'synthetic_pointwise.csv','Synthetic pointwise metrics')
test_pairs,_=close_pairs(syn['pos'],syn_split['test'],300,CFG.seed+1); syn_curv=curvature_eval(syn_models,syn,test_pairs,syn['z']); save_table(syn_curv,'synthetic_curvature.csv','Synthetic curvature metrics')
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
for ax,(name,pr) in zip(axes,syn_preds.items()):
    yt=pr['ft'].numpy().ravel(); yp=pr['fp'].numpy().ravel(); ids=np.linspace(0,len(yt)-1,min(2500,len(yt)),dtype=int); ax.scatter(yt[ids],yp[ids],s=8,alpha=.35,color=PALETTE[name]); lim=[min(yt.min(),yp.min()),max(yt.max(),yp.max())]; ax.plot(lim,lim,'--',color='#222222'); ax.set(title=name,xlabel='Exact force',ylabel='Predicted force'); ax.grid(alpha=.2)
fig.suptitle('Synthetic force verification',y=1.02); fig.tight_layout(); save_show(fig,'synthetic_force_scatter.png')

def verlet_true(p0,v0,dt,steps):
    p,v=p0,v0; ps=[]; hs=[]
    for _ in range(steps+1):
        x=p.detach().clone().requires_grad_(True); e=synthetic_potential(x); f=-torch.autograd.grad(e.sum(),x)[0]; ps.append(p); hs.append(e+.5*(v**2).sum((1,2)))
        if len(ps)==steps+1:break
        vh=v+.5*dt*f; pn=p+dt*vh; xn=pn.detach().clone().requires_grad_(True); en=synthetic_potential(xn); fn=-torch.autograd.grad(en.sum(),xn)[0]; v=vh+.5*dt*fn; p=pn
    return torch.stack(ps,1),torch.stack(hs,1)
rng=np.random.default_rng(CFG.seed); starts=syn['pos'][rng.choice(syn_split['test'],min(CFG.rollout_seeds,len(syn_split['test'])),False)].to(DEVICE); gen=torch.Generator(device=DEVICE).manual_seed(CFG.seed+9); v0=CFG.rollout_temperature*torch.randn(starts.shape,generator=gen,device=DEVICE); v0-=v0.mean(1,keepdim=True); truep,trueh=verlet_true(starts,v0,CFG.dt_syn,CFG.rollout_eval_steps_syn)
rollrows=[]; curves={}
for name,m in syn_models.items():
    em=m if hasattr(m,'energy') else m.energy_net; mp,mh=velocity_verlet_model(em,starts,v0,syn['z'].to(DEVICE),CFG.dt_syn,CFG.rollout_eval_steps_syn,False); curves[name]=mh.detach().cpu().numpy(); drift=((mh-mh[:,[0]]).abs()/(mh[:,[0]].abs()+1)).mean(1).detach().cpu().numpy(); tr=torch.sqrt(((mp-truep)**2).mean((2,3))).mean(1).detach().cpu().numpy()
    for metric,vals in [('Relative_Energy_Drift',drift),('Trajectory_RMSE',tr)]:
        mean,lo,hi=bootstrap_ci(vals); rollrows.append({'model':name,'metric':metric,'mean':mean,'ci_low':lo,'ci_high':hi})
rollout_df=pd.DataFrame(rollrows); save_table(rollout_df,'synthetic_rollout.csv','Synthetic rollout metrics')
fig,ax=plt.subplots(figsize=(8.5,5.2))
for name,h in curves.items():
    rel=np.abs(h-h[:,[0]])/(np.abs(h[:,[0]])+1); ax.plot(rel.mean(0),color=PALETTE[name],lw=2,label=name); ax.fill_between(np.arange(rel.shape[1]),np.quantile(rel,.25,0),np.quantile(rel,.75,0),color=PALETTE[name],alpha=.14)
ax.set_yscale('log'); ax.set(xlabel='Integrator step',ylabel='Relative energy drift',title='Synthetic rollout stability'); ax.grid(alpha=.25,which='both'); ax.legend(frameon=False); save_show(fig,'synthetic_rollout_drift.png')
truth_report(pd.concat([syn_metric_df,syn_curv[['model','metric','mean']],rollout_df[['model','metric','mean']]],ignore_index=True))

# %%
# Ensemble and split conformal helpers

def train_ensemble(data,split,ldrs,pairs,z,scaler,cutoff,zmax,prefix):
    M=CFG.ensemble_size_fast if CFG.fast_mode else CFG.ensemble_size_full; ens=[]
    epochs=(CFG.epochs_syn_fast if CFG.fast_mode else CFG.epochs_syn_full) if prefix=='synthetic' else (CFG.epochs_real_fast if CFG.fast_mode else CFG.epochs_real_full)
    for k in range(M):
        set_seed(CFG.seed+1000+k); m=InvariantEnergyNet(zmax,CFG.hidden_dim,CFG.n_rbf,cutoff,CFG.n_layers).to(DEVICE)
        m,_=train(m,f'CAPE_ensemble_{k+1}',ldrs,data,pairs,z,scaler,epochs,prefix+f'_ensemble{k+1}',True,prefix=='synthetic'); ens.append(m)
    return ens
def ensemble_predict(ens,ldr,z):
    pp=[predict(m,ldr,z) for m in ens]; es=torch.stack([p['ep'] for p in pp]); fs=torch.stack([p['fp'] for p in pp])
    return {'em':es.mean(0),'fm':fs.mean(0),'esd':es.std(0),'fsd':fs.std(0).pow(2).mean((1,2)).sqrt(),'et':pp[0]['et'],'ft':pp[0]['ft']}
def conformal(yc,muc,mut,alpha):
    s=np.abs(np.asarray(yc)-np.asarray(muc)); n=len(s); rank=min(n,int(np.ceil((n+1)*(1-alpha))))-1; q=np.sort(s)[rank]; return np.maximum(0,mut-q),mut+q,q
if CFG.train_ensemble:
    syn_ens=train_ensemble(syn,syn_split,syn_loaders,syn_pairs,syn['z'],syn_scaler,CFG.cutoff_syn,int(syn['z'].max()),'synthetic')
    ca=ensemble_predict(syn_ens,syn_loaders['cal'],syn['z']); te=ensemble_predict(syn_ens,syn_loaders['test'],syn['z'])
    yc=torch.sqrt(((ca['fm']-ca['ft'])**2).mean((1,2))).numpy(); yt=torch.sqrt(((te['fm']-te['ft'])**2).mean((1,2))).numpy(); uc=ca['fsd'].numpy(); ut=te['fsd'].numpy(); nf=max(10,len(yc)//2)
    beta=np.linalg.lstsq(np.c_[np.ones(nf),uc[:nf]],yc[:nf],rcond=None)[0]; muc=np.maximum(0,beta[0]+beta[1]*uc[nf:]); mut=np.maximum(0,beta[0]+beta[1]*ut); lo,hi,q=conformal(yc[nf:],muc,mut,CFG.alpha); cov=np.mean((yt>=lo)&(yt<=hi)); rho,pv=spearmanr(ut,yt)
    lab=(yt>np.quantile(yc,.9)).astype(int); auroc=roc_auc_score(lab,ut) if len(np.unique(lab))>1 else np.nan; aupr=average_precision_score(lab,ut) if len(np.unique(lab))>1 else np.nan
    cdf=pd.DataFrame([{'nominal_coverage':1-CFG.alpha,'empirical_coverage':cov,'mean_width':np.mean(hi-lo),'conformal_q':q,'spearman_rho':rho,'p_value':pv,'failure_AUROC':auroc,'failure_AUPRC':aupr}]); save_table(cdf,'synthetic_conformal.csv','Synthetic ensemble and conformal summary')
    fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(ut,yt,s=25,alpha=.6,color=PALETTE['CAPE-Ensemble']); xx=np.linspace(ut.min(),ut.max(),100); ax.plot(xx,np.maximum(0,beta[0]+beta[1]*xx),color='#222222',lw=2); ax.set(xlabel='Ensemble force disagreement',ylabel='Realised force RMSE',title=f'Uncertainty diagnostic: Spearman rho={rho:.3f}'); ax.grid(alpha=.25); save_show(fig,'synthetic_uncertainty.png')

# %%
# Real rMD17 analysis
if CFG.run_real_data:
    from torch_geometric.datasets import MD17
    import torch_geometric
    ds=MD17(root='data/MD17',name=CFG.real_dataset_name); print(ds,ds[0])
    pos=[]; ene=[]; frc=[]; z=ds[0].z.long()
    for item in tqdm(ds,desc='Reading rMD17'):
        pos.append(item.pos.float()); ene.append((item.energy if hasattr(item,'energy') and item.energy is not None else item.y).reshape(-1)[0].float()); frc.append((item.force if hasattr(item,'force') and item.force is not None else item.dy).float())
    real={'pos':torch.stack(pos),'energy':torch.stack(ene),'force':torch.stack(frc),'z':z}; print({k:tuple(v.shape) for k,v in real.items()})
    def blocked(total,counts,spacing,gap):
        out={}; cur=0
        for name,n in zip(['train','val','cal','test'],counts): out[name]=np.arange(cur,cur+n*spacing,spacing); cur=out[name][-1]+spacing+gap
        if max(np.concatenate(list(out.values())))>=total: raise ValueError('Split exceeds dataset; reduce counts/spacing.')
        return out
    counts=(CFG.real_n_train_fast,CFG.real_n_val_fast,CFG.real_n_cal_fast,CFG.real_n_test_fast) if CFG.fast_mode else (CFG.real_n_train_full,CFG.real_n_val_full,CFG.real_n_cal_full,CFG.real_n_test_full)
    rsp=blocked(len(real['pos']),counts,CFG.real_spacing,CFG.real_gap); save_table(pd.DataFrame([{'split':k,'n':len(v),'first':v[0],'last':v[-1],'spacing':np.min(np.diff(v))} for k,v in rsp.items()]),'real_split.csv','rMD17 blocked split')
    rsc=Scaler().fit(real['energy'][rsp['train']],real['force'][rsp['train']]); rld={k:loader(real,v,CFG.batch_size_real,k=='train') for k,v in rsp.items()}; rp=np.stack([rsp['train'][:-1],rsp['train'][1:]],1); d=torch.sqrt(((real['pos'][rp[:,1]]-real['pos'][rp[:,0]])**2).mean((1,2))).numpy(); rp=rp[d<=np.quantile(d,CFG.pair_max_distance_quantile)]
    idx=np.concatenate(list(rsp.values())); en=real['energy'][idx].numpy(); fn=torch.sqrt((real['force'][idx]**2).sum(-1)).numpy().ravel(); save_table(pd.DataFrame({'quantity':['Energy','Atomic force norm'],'mean':[en.mean(),fn.mean()],'sd':[en.std(),fn.std()],'min':[en.min(),fn.min()],'median':[np.median(en),np.median(fn)],'max':[en.max(),fn.max()]}),'real_descriptive.csv','rMD17 descriptive statistics')
    fig,ax=plt.subplots(1,2,figsize=(12,4.5)); ax[0].hist(en,bins=40,color='#0072B2',edgecolor='white'); ax[1].hist(fn,bins=50,color='#D55E00',edgecolor='white'); ax[0].set(title='Selected rMD17 energies',xlabel='Native energy unit'); ax[1].set(title='Atomic force norms',xlabel='Native force unit'); [a.grid(alpha=.18) for a in ax]; fig.tight_layout(); save_show(fig,'real_distributions.png')
    X=real['pos'][idx]; X=(X-X.mean(1,keepdim=True)).reshape(len(X),-1).numpy(); pc=PCA(2).fit_transform(X); labels=np.concatenate([[k]*len(v) for k,v in rsp.items()]); colors={'train':'#0072B2','val':'#E69F00','cal':'#CC79A7','test':'#009E73'}; fig,ax=plt.subplots(figsize=(7.5,5.5));
    for s in colors: q=labels==s; ax.scatter(pc[q,0],pc[q,1],s=18,alpha=.6,label=s,color=colors[s])
    ax.set(title='rMD17 blocked split in coordinate PCA space',xlabel='PC1',ylabel='PC2'); ax.grid(alpha=.2); ax.legend(frameon=False); save_show(fig,'real_split_pca.png')
    rmodels=make_models(CFG.cutoff_real,int(real['z'].max())); re=CFG.epochs_real_fast if CFG.fast_mode else CFG.epochs_real_full; rhs={}
    for name in ['DirectForceNet','ConservativeNet','CAPE-MD']:
        rmodels[name],rhs[name]=train(rmodels[name],name,rld,real,rp,real['z'],rsc,re,'rMD17_aspirin',name=='CAPE-MD',False); gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    plot_histories(rhs,'real')
    rr=[]; rpred={}
    for name,m in rmodels.items():
        pr=predict(m,rld['test'],real['z']); rpred[name]=pr
        for k,v in metrics(pr).items(): rr.append({'model':name,'metric':k,'mean':v})
    rdf=pd.DataFrame(rr); save_table(rdf.pivot(index='model',columns='metric',values='mean').reset_index(),'real_metrics.csv','rMD17 aspirin test metrics')
    tp=np.stack([rsp['test'][:-1],rsp['test'][1:]],1); dd=torch.sqrt(((real['pos'][tp[:,1]]-real['pos'][tp[:,0]])**2).mean((1,2))).numpy(); tp=tp[dd<=np.quantile(dd,CFG.pair_max_distance_quantile)]; rcurv=curvature_eval(rmodels,real,tp,real['z']); save_table(rcurv,'real_curvature.csv','rMD17 curvature metrics')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,(name,pr) in zip(axes,rpred.items()):
        er=(pr['fp']-pr['ft']).numpy().ravel(); ax.hist(er,bins=45,color=PALETTE[name],edgecolor='white',alpha=.85); ax.axvline(0,color='#222222',ls='--'); ax.set(title=name,xlabel='Force residual',ylabel='Frequency'); ax.grid(alpha=.18)
    fig.suptitle('rMD17 aspirin test residuals',y=1.02); fig.tight_layout(); save_show(fig,'real_force_residuals.png'); truth_report(pd.concat([rdf,rcurv[['model','metric','mean']]],ignore_index=True))
    if CFG.train_ensemble:
        rens=train_ensemble(real,rsp,rld,rp,real['z'],rsc,CFG.cutoff_real,int(real['z'].max()),'real'); ca=ensemble_predict(rens,rld['cal'],real['z']); te=ensemble_predict(rens,rld['test'],real['z']); yc=torch.sqrt(((ca['fm']-ca['ft'])**2).mean((1,2))).numpy(); yt=torch.sqrt(((te['fm']-te['ft'])**2).mean((1,2))).numpy(); uc=ca['fsd'].numpy(); ut=te['fsd'].numpy()
        L=max(10,len(yc)//10)
        def agg(y,u,L):
            n=len(y)//L; return np.array([y[i*L:(i+1)*L].mean() for i in range(n)]),np.array([u[i*L:(i+1)*L].max() for i in range(n)])
        ycb,ucb=agg(yc,uc,L); ytb,utb=agg(yt,ut,L); nf=max(3,len(ycb)//2); beta=np.linalg.lstsq(np.c_[np.ones(nf),ucb[:nf]],ycb[:nf],rcond=None)[0]; muc=np.maximum(0,beta[0]+beta[1]*ucb[nf:]); mut=np.maximum(0,beta[0]+beta[1]*utb); lo,hi,q=conformal(ycb[nf:],muc,mut,CFG.alpha); cov=np.mean((ytb>=lo)&(ytb<=hi)); rho,pv=spearmanr(utb,ytb)
        save_table(pd.DataFrame([{'block_length':L,'calibration_blocks':len(ycb),'test_blocks':len(ytb),'nominal_coverage':1-CFG.alpha,'empirical_coverage':cov,'mean_width':np.mean(hi-lo),'spearman_rho':rho,'p_value':pv,'warning':'Exact coverage requires exchangeable calibration and test blocks.'}]),'real_conformal.csv','rMD17 block-conformal summary')
        order=np.argsort(mut); fig,ax=plt.subplots(figsize=(8.5,5)); x=np.arange(len(order)); ax.errorbar(x,mut[order],yerr=[mut[order]-lo[order],hi[order]-mut[order]],fmt='o',capsize=4,color=PALETTE['CAPE-Ensemble'],label='Conformal interval'); ax.scatter(x,ytb[order],marker='x',s=65,color='#222222',label='Observed block RMSE'); ax.set(title=f'rMD17 conformal intervals; coverage={cov:.3f}',xlabel='Ordered test block',ylabel='Force RMSE'); ax.grid(alpha=.2); ax.legend(frameon=False); save_show(fig,'real_conformal_intervals.png')

# %%
# Final manifest and ZIP
frames=[]
for scope,obj in [('synthetic',globals().get('syn_metric_df')),('synthetic',globals().get('syn_curv')),('synthetic',globals().get('rollout_df')),('rMD17',globals().get('rdf')),('rMD17',globals().get('rcurv'))]:
    if isinstance(obj,pd.DataFrame) and {'model','metric','mean'}.issubset(obj.columns):
        q=obj[['model','metric','mean']].copy(); q.insert(0,'analysis',scope); frames.append(q)
if frames: save_table(pd.concat(frames,ignore_index=True),'complete_summary.csv','Complete metric summary')
manifest={'created_utc':pd.Timestamp.utcnow().isoformat(),'device':str(DEVICE),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'seed':CFG.seed,'fast_mode':CFG.fast_mode,'note':'No numerical superiority is hard-coded. Synthetic and real-data analyses are separate.'}
if CFG.run_real_data: manifest['torch_geometric']=torch_geometric.__version__
with open(RESULTS/'run_manifest.json','w') as f: json.dump(manifest,f,indent=2)
with open(RESULTS/'pip_freeze.txt','w') as f: subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,text=True)
zip_name=Path('CAPE_MD_RESULTS.zip')
if zip_name.exists(): zip_name.unlink()
with zipfile.ZipFile(zip_name,'w',zipfile.ZIP_DEFLATED) as zf:
    for p in RESULTS.rglob('*'):
        if p.is_file(): zf.write(p,p.relative_to(RESULTS.parent))
print('Complete:',zip_name.resolve(),'files:',sum(p.is_file() for p in RESULTS.rglob('*')))
if 'google.colab' in sys.modules:
    from google.colab import files
    files.download(str(zip_name))
