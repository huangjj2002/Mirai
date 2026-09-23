import copy
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import (atomic_save, digest, file_hash, run_signature, select_device,
                     seed_all, torch_load, write_json)
from .data import ExamDataset, batch_inputs, compute_stats, image_tensor, load_exams
from .metrics import evaluate_metrics
from .models import (RiskTargets, adversary_inputs, architecture, build_adversary,
                     build_model, risk_loss)


def initialize(c):
    seed_all(c['seed'])
    torch.set_num_threads(c.get('threads',4))
    device = select_device(c['device'])
    Path(c['output_dir']).mkdir(parents=True,exist_ok=True)
    return device


def normalization(c, exams, report, compute=False):
    output = Path(c['output_dir'])/'normalization.json'
    signature = digest({'size':c['image']['size'], 'sources':report['source_hashes'],
                        'images':report['image_stat_fingerprint']})
    if c['image'].get('mean') is not None and c['image'].get('std') is not None:
        stats = {'mean':float(c['image']['mean']), 'std':float(c['image']['std']), 'source':'explicit_config'}
        if not math.isfinite(stats['mean']) or not math.isfinite(stats['std']) or stats['std'] <= 0:
            raise ValueError('Invalid normalization statistics')
        return stats
    if output.exists() and not compute:
        stats = json.loads(output.read_text())
        if stats.get('signature') != signature:
            raise ValueError('Normalization metadata/images changed; run check-data --compute-stats')
        return stats
    print('Computing normalization on training images only.',flush=True)
    stats = compute_stats(exams,c)
    stats['signature'] = signature
    write_json(output,stats)
    return stats


def check_data(c, pixels=False, compute=False):
    exams,report = load_exams(c,check_pixels=pixels)
    write_json(Path(c['output_dir'])/'data_audit.json',report)
    if compute:
        normalization(c,exams,report,compute=True)
    print(json.dumps({k:report[k] for k in ['exams_by_split','train_device_image_counts','train_known_risk_factors']},indent=2))
    print(f"Excluded exams: {len(report['excluded'])}; detailed data_audit.json written.")
    return report


def loader(ds,c,stage,training=False):
    sampler = None
    if training:
        events = np.array([e['event'] for e,_ in ds.entries])
        counts = np.bincount(events,minlength=2)
        if (counts==0).any():
            raise ValueError('Training requires both observed cancer and non-event examples')
        sampler = WeightedRandomSampler(torch.tensor(1/counts[events],dtype=torch.double),len(events),replacement=True)
    return DataLoader(ds,batch_size=c[f'stage{stage}']['batch_size'],sampler=sampler,
                      num_workers=c.get('num_workers',0),shuffle=False)


def inference(model,dl,device,stage,train_exams):
    model.eval()
    by_exam = defaultdict(list)
    truths = {}
    loss_sum=0.0; mask_count=0.0
    with torch.no_grad():
        for batch in dl:
            x,rf,mb = batch_inputs(batch,device)
            logits,_,_ = model(x,rf,mb)
            mask=batch['y_mask'].to(device)
            loss_sum += float(F.binary_cross_entropy_with_logits(logits,batch['y_seq'].to(device),reduction='none').mul(mask).sum())
            mask_count += float(mask.sum())
            for i,key in enumerate(batch['key']):
                by_exam[key].append(torch.sigmoid(logits[i]).cpu().numpy())
                truths[key]=(int(batch['time'][i]),int(batch['event'][i]))
    keys=sorted(by_exam)
    probs=np.array([np.mean(by_exam[k],axis=0) for k in keys])
    times=[truths[k][0] for k in keys]; events=[truths[k][1] for k in keys]
    metrics=evaluate_metrics(times,events,probs,[e['time'] for e in train_exams],[e['event'] for e in train_exams])
    metrics['risk_loss']=loss_sum/max(mask_count,1)
    return metrics,keys,probs


def feature_signature(c,report,stats,encoder_path):
    return {'encoder_sha256':file_hash(encoder_path), 'data':report['source_hashes'],
            'image_stat_fingerprint':report['image_stat_fingerprint'],
            'normalization':stats, 'size':c['image']['size'], 'version':1}


def extract_features(c,encoder_path=None):
    device=initialize(c)
    exams,report=load_exams(c)
    stats=normalization(c,exams,report)
    encoder_path=Path(encoder_path or Path(c['output_dir'])/'stage1/best.pt')
    checkpoint=torch_load(encoder_path)
    if checkpoint['stage']!=1:
        raise ValueError('Feature extraction requires a stage1 checkpoint')
    if checkpoint['normalization']!=stats or checkpoint['signature']!=run_signature(c):
        raise ValueError('Encoder data/config does not match this run')
    model=build_model(c,1).to(device)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    model.requires_grad_(False)
    features={}
    with torch.no_grad():
        for j,e in enumerate(exams):
            vectors=[]
            # One image at a time: extraction remains usable on modest VRAM.
            for p in e['paths']:
                x=image_tensor(p,c['image'],stats).unsqueeze(0).to(device)
                rf=RiskTargets([torch.tensor(v).unsqueeze(0).to(device) for v in e['factors']],
                               torch.tensor(e['factor_known']).unsqueeze(0).to(device))
                _,hidden,_=model(x,rf)
                vectors.append(hidden[0,:512].cpu())
            features[e['key']]=torch.stack(vectors)
            if j==0 or (j+1)%100==0:
                print(f'Features: {j+1}/{len(exams)} exams',flush=True)
    out=Path(c['output_dir'])/'features.pt'
    atomic_save({'signature':feature_signature(c,report,stats,encoder_path),
                 'features':features, 'encoder_path':str(encoder_path.resolve())},out)
    print(f'Features saved: {out}',flush=True)
    return out


def get_features(c,report,stats,encoder_path=None):
    path=Path(c['output_dir'])/'features.pt'
    if not path.is_file():
        raise ValueError('Missing features.pt; run extract-features first')
    cache=torch_load(path)
    encoder_path=Path(encoder_path or Path(c['output_dir'])/'stage1/best.pt')
    if cache['signature']!=feature_signature(c,report,stats,encoder_path):
        raise ValueError('Stale feature cache: encoder, input or preprocessing changed')
    return cache['features'],str(encoder_path.resolve()),cache['signature']


def rng_state(device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}


def restore_rng(state,device):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if device.type=='cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def train_stage(c,stage,resume=None):
    device=initialize(c)
    exams,report=load_exams(c,check_pixels=True)
    write_json(Path(c['output_dir'])/'data_audit.json',report)
    stats=normalization(c,exams,report)
    features=encoder_path=feature_meta=None
    if stage==2:
        if len(report['train_device_image_counts'])<2:
            raise ValueError('Stage2 adversarial training needs at least two recognized training device classes')
        features,encoder_path,feature_meta=get_features(c,report,stats)
    out=Path(c['output_dir'])/f'stage{stage}'; out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not resume:
        raise ValueError(f'Existing training run: use --resume {out / "last.pt"} or a new output_dir')
    train_ds=ExamDataset(exams,'train',stage,c,stats,features)
    dev_ds=ExamDataset(exams,'dev',stage,c,stats,features)
    train_dl=loader(train_ds,c,stage,True); dev_dl=loader(dev_ds,c,stage)
    model=build_model(c,stage,imagenet=(stage==1 and c['stage1']['imagenet'] and not resume)).to(device)
    adv=build_adversary(model).to(device) if stage==2 else None
    s=c[f'stage{stage}']
    opt=torch.optim.Adam(model.parameters(),lr=s['lr'],weight_decay=s['weight_decay'])
    adv_opt=torch.optim.Adam(adv.parameters(),lr=s['lr'],weight_decay=s['weight_decay']) if adv else None
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.1,patience=s['lr_patience'])
    signature=run_signature(c)
    start=1; best_score=-float('inf'); history=[]; best_epoch=0
    if resume:
        ck=torch_load(resume)
        if ck['stage']!=stage or ck['signature']!=signature or ck['normalization']!=stats or ck['image_fingerprint']!=report['image_stat_fingerprint']:
            raise ValueError('Resume checkpoint is incompatible with stage/data/config/images')
        if ck['feature_signature']!=feature_meta:
            raise ValueError('Resume feature cache differs')
        model.load_state_dict(ck['model'],strict=True); opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        if adv:
            adv.load_state_dict(ck['adversary']); adv_opt.load_state_dict(ck['adversary_optimizer'])
        start=ck['epoch']+1; best_score=ck['best_score']; best_epoch=ck['best_epoch']; history=ck['history']
        restore_rng(ck['rng'],device)
        if not (out/'best.pt').exists():
            if ck['epoch']!=best_epoch:
                raise ValueError('Resume requires the run best.pt alongside last.pt')
            atomic_save(ck,out/'best.pt')
    if start>s['epochs']:
        print(f'Stage {stage} already completed {start-1} epochs.',flush=True)
        return out/'best.pt'
    write_json(out/'architecture.json',{'model':architecture(model),'adversary':architecture(adv) if adv else None})
    for epoch in range(start,s['epochs']+1):
        model.train(); opt.zero_grad(set_to_none=True)
        total=0.; updates=0; adv_updates=0; pending=[]
        for step,batch in enumerate(train_dl):
            x,rf,mb=batch_inputs(batch,device)
            logits,hidden,activ=model(x,rf,mb)
            if 'pred_rf_loss' not in activ:
                raise RuntimeError('Original model suppressed an RF loss error; refusing silent training')
            loss=risk_loss(logits,batch)+s['rf_weight']*activ['pred_rf_loss']
            if adv:
                values,targets,known=adversary_inputs(hidden,logits,batch)
                pending.append((loss,values,targets,known))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at stage {stage}, epoch {epoch}, step {step}')
            start_window=(step//s['accumulate'])*s['accumulate']
            window=min(s['accumulate'],len(train_dl)-start_window)
            if not adv:
                (loss/window).backward(); total+=float(loss.detach())
            if (step+1)%s['accumulate']==0 or step+1==len(train_dl):
                if adv:
                    # Three D steps per effective G update, also with gradient accumulation.
                    values=torch.cat([p[1][p[3]].detach() for p in pending])
                    targets=torch.cat([p[2][p[3]] for p in pending])
                    supervised=len(targets)>=2
                    if supervised:
                        adv.requires_grad_(True); adv.train()
                        for _ in range(s['adv_steps']):
                            adv_opt.zero_grad(set_to_none=True)
                            dloss=F.cross_entropy(adv(values),targets)
                            if not torch.isfinite(dloss):
                                raise FloatingPointError('Non-finite discriminator loss')
                            dloss.backward(); adv_opt.step(); adv_updates+=1
                        adv_opt.zero_grad(set_to_none=True)
                    adv.requires_grad_(False); adv.eval()
                    known_total=sum(int(p[3].sum()) for p in pending)
                    for base,v,y,k in pending:
                        loss=base/len(pending)
                        if supervised and k.any():
                            loss=loss-s['adv_weight']*F.cross_entropy(adv(v[k]),y[k],reduction='sum')/known_total
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Non-finite generator loss')
                        loss.backward(); total+=float(loss.detach())*len(pending)
                    pending.clear()
                opt.step(); opt.zero_grad(set_to_none=True); updates+=1
            if step==0 or (step+1)%50==0:
                print(f'Stage {stage} epoch {epoch} batch {step+1}/{len(train_dl)} loss={float(loss.detach()):.5f}',flush=True)
        if adv and not adv_updates:
            raise ValueError('No discriminator update was possible; check known device labels and batch size')
        metrics,_,_=inference(model,dev_dl,device,stage,train_ds.exams)
        score=metrics['mirai_legacy_c_index']
        if score is None:
            if not c['synthetic']:
                raise ValueError('Validation legacy C-index undefined. Fix cohort/censoring support; no silent metric substitution.')
            score=-metrics['risk_loss']
        scheduler.step(score)
        improved=score>best_score
        if improved:
            best_score=float(score); best_epoch=epoch
        history.append({'epoch':epoch,'train_loss':total/len(train_dl),'optimizer_updates':updates,
                        'adversary_updates':adv_updates,'dev':metrics,'lr':opt.param_groups[0]['lr']})
        ck={'format':1,'stage':stage,'epoch':epoch,'best_score':best_score,'best_epoch':best_epoch,
            'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'adversary':adv.state_dict() if adv else None,
            'adversary_optimizer':adv_opt.state_dict() if adv else None,
            'rng':rng_state(device),'config':copy.deepcopy(c),'signature':signature,
            'normalization':stats,'image_fingerprint':report['image_stat_fingerprint'],
            'encoder_path':encoder_path,'feature_signature':feature_meta,'history':history}
        atomic_save(ck,out/'last.pt')
        if improved:
            atomic_save(ck,out/'best.pt')
        write_json(out/'history.json',history)
        print(f'Stage {stage} epoch {epoch}: {json.dumps(metrics)}',flush=True)
    return out/'best.pt'


def evaluate(c,checkpoint,split):
    device=initialize(c)
    ck=torch_load(checkpoint)
    if ck['signature']!=run_signature(c):
        raise ValueError('Evaluation config/input differs from checkpoint; use its original run config')
    exams,report=load_exams(c)
    stats=ck['normalization']; stage=ck['stage']; features=None
    if ck['image_fingerprint']!=report['image_stat_fingerprint']:
        raise ValueError('Evaluation images changed since training')
    if stage==2:
        features,_,meta=get_features(c,report,stats)
        if meta!=ck['feature_signature']:
            raise ValueError('Evaluation features differ from stage2 training')
    model=build_model(c,stage).to(device); model.load_state_dict(ck['model'],strict=True)
    ds=ExamDataset(exams,split,stage,c,stats,features)
    metrics,keys,probs=inference(model,loader(ds,c,stage),device,stage,[e for e in exams if e['split']=='train'])
    out=Path(c['output_dir'])/f'stage{stage}'
    with (out/f'{split}_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['patient_exam_id']+[f'{i}_year_risk' for i in range(1,6)])
        w.writerows([[key,*p] for key,p in zip(keys,probs)])
    write_json(out/f'{split}_metrics.json',metrics)
    print(json.dumps(metrics,indent=2),flush=True)
    return metrics
