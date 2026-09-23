"""Synthetic engineering checks. These results are never clinical validation."""
import copy
import csv
import gc
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import yaml

from .config import ROOT, file_hash, load_config, seed_all, torch_load, write_json
from .data import batch_inputs, image_tensor, labels, load_exams
from .engine import check_data, evaluate, extract_features, get_features, initialize, train_stage
from .metrics import concordance
from .models import RiskTargets, architecture, build_adversary, build_model, risk_schema
from .schema_constants import VIEWS


def assert_close(a,b):
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)


def model_checks(c):
    seed_all(c['seed'])
    schema=risk_schema(c)
    assert len(schema)==34 and sum(s['size'] for s in schema)==100
    inputs=torch.randn(2,3,64,64)
    known=torch.ones(2,34,dtype=torch.bool)
    factors=[]
    for s in schema:
        v=torch.zeros(2,s['size']); v[:,0]=1
        factors.append(v)
    targets=RiskTargets(factors,known)
    batch={'time_seq':torch.zeros(2,4,dtype=torch.long),
           'view_seq':torch.tensor([[0,1,0,1]]*2),'side_seq':torch.tensor([[0,0,1,1]]*2)}
    summary={}
    for stage in (1,2):
        original=build_model(c,stage,masked=False)
        adapted=build_model(c,stage,masked=True)
        adapted.load_state_dict(original.state_dict(),strict=True)
        assert architecture(original)==architecture(adapted)
        x=inputs if stage==1 else torch.randn(2,4,512)
        original.eval(); adapted.eval()
        a=original(x,targets,batch); b=adapted(x,targets,batch)
        assert_close(a[0],b[0]); assert_close(a[1],b[1]); assert_close(a[2]['pred_rf_loss'],b[2]['pred_rf_loss'])
        (a[0].sum()+a[2]['pred_rf_loss']).backward()
        (b[0].sum()+b[2]['pred_rf_loss']).backward()
        for (n,pa),(nb,pb) in zip(original.named_parameters(),adapted.named_parameters()):
            assert n==nb and ((pa.grad is None)==(pb.grad is None))
            if pa.grad is not None:
                assert_close(pa.grad,pb.grad)
        assert b[0].shape==(2,5)
        assert torch.all(torch.diff(torch.sigmoid(b[0]),dim=-1)>=-1e-7)
        # Mask ALL factors; no factor head should receive auxiliary supervision.
        adapted.zero_grad(set_to_none=True)
        missing=RiskTargets([torch.zeros_like(v) for v in factors],torch.zeros_like(known))
        adapted.train()
        out=adapted(x,missing,batch)
        assert float(out[2]['pred_rf_loss'])==0
        out[2]['pred_rf_loss'].backward()
        pool=adapted._model.pool if stage==1 else adapted.pool
        assert all(pool._modules[f'{s["name"]}_fc'].weight.grad is None for s in schema)
        # One observed factor provides gradient only to its own auxiliary head.
        adapted.zero_grad(set_to_none=True)
        partial_known=torch.zeros_like(known); partial_known[:,0]=True
        partial=RiskTargets(factors,partial_known)
        adapted(x,partial,batch)[2]['pred_rf_loss'].backward()
        assert pool._modules[f'{schema[0]["name"]}_fc'].weight.grad is not None
        assert pool._modules[f'{schema[1]["name"]}_fc'].weight.grad is None
        summary[f'stage{stage}']=architecture(original)
        if stage==2:
            summary['adversary']=architecture(build_adversary(adapted))
            assert build_adversary(adapted).fc3.out_features==4
        del original,adapted,a,b,out
        gc.collect()
    return summary


def source_checks():
    vendor=Path(__file__).parent/'vendor'
    manifest=json.loads((vendor/'manifest.json').read_text())
    for name,hashes in manifest['files'].items():
        assert file_hash(vendor/'onconet'/name)==hashes['vendored_sha256'], f'Vendored source changed: {name}'
        original=ROOT/'code/Mirai-master/Mirai-master/onconet'/name
        if original.exists():
            assert file_hash(original)==hashes['upstream_sha256']
            expected=original.read_text(encoding='utf-8').replace('onconet.','mirai_training.vendor.onconet.')
            if name=='models/discriminator.py':
                expected=expected.replace('mirai_training.vendor.onconet.datasets.abstract_onco_dataset','mirai_training.schema_constants')
            assert expected==(vendor/'onconet'/name).read_text(encoding='utf-8')
    return {'upstream_commit':manifest['commit'],'unchanged_model_source_files':len(manifest['files'])}


def data_checks(c):
    seq,mask,event,t=labels(2,2)
    assert seq.tolist()==[0,0,1,1,1] and mask.tolist()==[1,1,1,0,0]
    seq,mask,_,_=labels(100,2)
    assert seq.tolist()==[0]*5 and mask.tolist()==[1,1,0,0,0]
    assert labels(100,0) is None
    exams,_=load_exams(c,True)
    assert exams[0]['exam_id'].startswith('9000000000000000')
    assert [Path(p).stem.split('_')[-2:] for p in exams[0]['paths']]==[['R','CC'],['R','MLO'],['L','CC'],['L','MLO']]
    # Source-compatible ranking and ties on a fully observed comparable pair.
    high=np.array([[.9]*5,[.1]*5]); low=high[::-1].copy()
    for legacy in (True,False):
        assert concordance([0,4],[1,0],high,[0,4],[1,0],legacy)==1
        assert concordance([0,4],[1,0],low,[0,4],[1,0],legacy)==0
        assert concordance([0,4],[1,0],np.ones((2,5))*.5,[0,4],[1,0],legacy)==.5
    # Inject a patient crossing splits; audit must fail before training.
    original=Path(c['data']['metadata_csv'])
    with original.open(newline='') as f:
        reader=csv.DictReader(f); fieldnames=reader.fieldnames; rs=list(reader)
    rs[4]['patient_id']=rs[0]['patient_id']; rs[4]['split_group']='test'
    bad=original.with_name('invalid_patient_split.csv')
    with bad.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rs)
    invalid=copy.deepcopy(c); invalid['data']['metadata_csv']=str(bad)
    try:
        load_exams(invalid)
    except ValueError as e:
        assert 'across splits' in str(e)
    else:
        raise AssertionError('Patient leakage was not rejected')


def generate(c,out,device):
    rng=np.random.default_rng(123)
    image_dir=out/'images'; image_dir.mkdir(parents=True)
    metadata=out/'metadata.csv'; rf=[]
    with metadata.open('w',newline='',encoding='utf-8') as f:
        fields=['patient_id','exam_id','laterality','view','file_path','years_to_cancer','years_to_last_followup','split_group','device_model']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i in range(12):
            pid=f'p{i:03d}'; eid=str(900000000000000000+i)
            split='train' if i<6 else 'dev' if i<9 else 'test'
            positive=i%2==0
            for side,view in VIEWS:
                a=rng.integers(0,20000,(64,64),dtype=np.uint16)
                a[:,:16]//=4
                path=image_dir/f'{pid}_{side}_{view}.png'
                Image.fromarray(a).save(path)
                w.writerow(dict(patient_id=pid,exam_id=eid,laterality=side,view=view,file_path=str(path),
                                years_to_cancer=i%4 if positive else 100,years_to_last_followup=5,split_group=split,
                                device_model='Selenia Dimensions' if i%3 else 'Lorad Selenia'))
            rf.append({'patient_id':pid,'exam_id':eid,'factors':{
                'age':[0,0,1,0,0,0], 'density':[0,1,0,0] if i%3 else None,
                'binary_family_history':[float(i%2)] if i%4 else None}})
    write_json(out/'risk_factors.json',rf)
    c=copy.deepcopy(c); c.pop('_config_path',None)
    c['synthetic']=True; c['device']=device; c['output_dir']=str(out/'run'); c['num_workers']=0
    c['data']={'metadata_csv':str(metadata),'risk_factors_json':str(out/'risk_factors.json'),
               'image_metadata_csv':None,'clinical_csv':None}
    c['image']={'size':[64,64],'mean':None,'std':None}
    c['stage1'].update(imagenet=False,epochs=1,batch_size=2,accumulate=2)
    c['stage2'].update(epochs=1,batch_size=2,accumulate=2)
    config=out/'smoke.yaml'; config.write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
    return load_config(config)


def run_smoke(base,full_resolution=False,device='cpu'):
    root=ROOT/'outputs'/'mirai_smoke'; root.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='check_',dir=root))
    started=time.time(); c=generate(base,out,device); initialize(c)
    source=source_checks(); shapes=model_checks(c); data_checks(c)
    check_data(c,pixels=True,compute=True)
    best1=train_stage(c,1)
    # Replay an epoch twice from identical checkpoint state; all parameter values must agree.
    resumed=copy.deepcopy(c); resumed['stage1']['epochs']=2
    branch=out/'resume_branch'; (branch/'stage1').mkdir(parents=True)
    from shutil import copy2
    copy2(best1,branch/'stage1/best.pt')
    copy2(Path(c['output_dir'])/'normalization.json',branch/'normalization.json')
    resumed['output_dir']=str(branch)
    train_stage(resumed,1,best1)
    first=torch_load(branch/'stage1/last.pt')
    train_stage(resumed,1,best1)
    second=torch_load(branch/'stage1/last.pt')
    for k,v in first['model'].items():
        assert_close(v,second['model'][k])
    del first,second
    encoder_hash=file_hash(best1)
    extract_features(c)
    best2=train_stage(c,2)
    assert file_hash(best1)==encoder_hash
    stage2_ck=torch_load(best2)
    for entry in stage2_ck['history']:
        assert entry['adversary_updates']==3*entry['optimizer_updates']
    metrics=evaluate(c,best2,'test')
    ck=torch_load(best2); m=build_model(c,2); m.load_state_dict(ck['model'],strict=True)
    m2=build_model(c,2); m2.load_state_dict(ck['model'],strict=True)
    m.eval(); m2.eval()
    batch={'time_seq':torch.zeros(2,4,dtype=torch.long),'view_seq':torch.tensor([[0,1,0,1]]*2),'side_seq':torch.tensor([[0,0,1,1]]*2)}
    x=torch.randn(2,4,512)
    with torch.no_grad():
        assert_close(m(x,None,batch)[0],m2(x,None,batch)[0])
    # Compare cached inference to the original MiraiFull.forward on actual four-view PNG inputs.
    from .vendor.onconet.models.mirai_full import MiraiFull
    exams,audit=load_exams(c)
    cache=torch_load(Path(c['output_dir'])/'features.pt')
    encoder=build_model(c,1)
    encoder.load_state_dict(torch_load(best1)['model'],strict=True)
    encoder._model.args.use_pred_risk_factors_at_test=True
    encoder.eval(); encoder.requires_grad_(False)
    frozen_before={k:v.clone() for k,v in encoder.state_dict().items()}
    assembled=torch.nn.Module(); assembled.image_encoder=encoder; assembled.transformer=m
    assembled.image_repr_dim=512
    raw=torch.stack([torch.stack([image_tensor(p,c['image'],ck['normalization']) for p in e['paths']],dim=1)
                     for e in exams[:2]])
    with torch.no_grad():
        full_logits=MiraiFull.forward(assembled,raw,None,batch)[0]
        cached_logits=m(torch.stack([cache['features'][e['key']] for e in exams[:2]]),None,batch)[0]
    assert_close(full_logits,cached_logits)
    assert all(torch.equal(v,encoder.state_dict()[k]) for k,v in frozen_before.items())
    changed=copy.deepcopy(c); changed['image']['size']=[32,32]
    try:
        get_features(changed,audit,ck['normalization'])
    except ValueError as e:
        assert 'Stale feature' in str(e)
    else:
        raise AssertionError('Stale preprocessing cache accepted')
    resolution_result=None
    if full_resolution:
        model=build_model(c,1).eval().to(device)
        example=next((out/'images').glob('*.png'))
        img=image_tensor(example,{'size':[1664,2048]},ck['normalization']).unsqueeze(0).to(device)
        targets=RiskTargets([torch.zeros(1,s['size'],device=device) for s in risk_schema(c)],torch.zeros(1,34,dtype=torch.bool,device=device))
        with torch.no_grad():
            result=model(img,targets)
        assert result[0].shape==(1,5)
        resolution_result={'input_shape':list(img.shape),'output_shape':list(result[0].shape),
                           'note':'synthetic PNG at full size; encoder inference only, no full-size backward'}
    report={'status':'passed','synthetic_only':True,'imagenet_download_tested':False,
            'device':device,'elapsed_seconds':time.time()-started,'source':source,
            'checks':['source integrity','structure equivalence','forward/loss/gradient equivalence with complete RF',
                      'missing/partial RF gradient masking','labels and patient leakage','stage1 training',
                      'deterministic epoch-boundary resume','512-D feature export','stage2 adversarial training',
                      'encoder state and BatchNorm unchanged','save/reload prediction equality','test CSV export',
                      'three discriminator updates per effective generator update',
                      'cached vs original MiraiFull four-view forward equality','stale cache rejection'],
            'full_resolution':resolution_result,'test_metrics_are_synthetic':metrics}
    write_json(out/'architecture.json',shapes); write_json(out/'report.json',report)
    print(f'SMOKE PASS: {out / "report.json"}',flush=True)
    return report
