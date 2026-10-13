"""Adaptation-only token coverage diagnostic for frozen V38 candidates.

State slices: theseus-lab-sandbox-v1, theseus-v38-token-coverage-audit-v1,
global-experiment-assurance-lifecycle-v1. No training, new prompts or holdouts.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import replacement_loss_v38_runner as parent

c,r=parent.c,parent.r
PROTOCOL='theseus-v38-token-coverage-audit-v1'
PARENT=r.BASE
BASE=r.ARTIFACTS/'theseus-lab-sandbox-v1-v38-token-coverage-audit-v1'
FREEZE=r.REPO/'docs/v38-token-coverage-audit-v1-freeze.json'
MEMBERS=('theseus_token_coverage_audit_v1.py','tests/test_token_coverage_audit_v1.py',
         'docs/v38-token-coverage-audit-v1.md','docs/replacement-loss-v38-freeze.json',
         'docs/replacement-loss-v38-results.json')


def members():
    return [{'path':p,'bytes':(r.REPO/p).stat().st_size,'sha256':r.sha(r.REPO/p)} for p in MEMBERS]


def verify():
    value=r.read(FREEZE)
    if value['members']!=members() or value['packet_sha256']!=c.digest(members()):raise ValueError('diagnostic source differs')
    return value['packet_sha256']


def rmse_sums(error,reference,count):
    if count<=0 or not all(math.isfinite(v) and v>=0 for v in (error,reference)):raise ValueError('error statistic')
    return math.sqrt(error/max(reference,count*1e-8))


def replacement(weights,site):
    prefix=str(site)+'/'
    state={key[len(prefix):]:torch.from_numpy(value.copy()) for key,value in weights.items() if key.startswith(prefix)}
    module=c.Replacement(len(state['mean']),'gpt2_gelu',0,state['mean'],state['scale'])
    module.load_state_dict(state,strict=True)
    return module.eval().requires_grad_(False)


def summary(rows):
    prefix=rmse_sums(*rows[:,:3].sum(0));final=rmse_sums(*rows[:,3:].sum(0))
    return {'prefix_nrmse':prefix,'final_nrmse':final,
            'prefix_to_final_error_ratio':prefix/max(final,1e-6),
            'prefix_coordinates':int(rows[:,2].sum()),'final_coordinates':int(rows[:,5].sum()),
            'twofold_coverage_gap':prefix>=2*final}


def run(role):
    r.mount_guard();packet=verify();r.verify_source()
    receipt=r.read(r.REPO/'docs/replacement-loss-v38-results.json')
    if r.sha(PARENT/'closeout/validation.json')!=receipt['validation_sha256'] or receipt['scientific_outcome']!='FAIL':
        raise ValueError('parent terminal binding')
    manifest=parent.manifest(PARENT/'checkpoints.json');runtime=r.runtime()
    root=BASE/role;root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def check():
        r.mount_guard()
        if time.monotonic()-start>1800:raise RuntimeError('diagnostic time cap')
    model,tokenizer=r.load_model(manifest['checkpoints'][0]);frozen=r.read(PARENT/'primary/candidate-freeze.json')
    cells={};statistics={};inputs=[]
    for case,binding in frozen['families']['gpt2_gelu']['cases'].items():
        folder=PARENT/'primary'/case;adaptation=r.read(folder/'adaptation.json')
        inputs.append({'path':str((folder/'adaptation.json').relative_to(r.ARTIFACTS)),
                       'sha256':r.sha(folder/'adaptation.json')})
        for method,stages in binding['arms'].items():
            for stage,target in stages.items():
                site=target['active_sites'][-1];stem=folder/(method+'-'+stage)
                weights=r.arrays(stem.with_name(stem.name+'-weights.npz'))
                if c.array_digest(weights)!=target['weights_sha256']:raise ValueError('candidate changed')
                captured=r.arrays(stem.with_name(stem.name+'-adaptation-inputs.npz'))
                if c.array_digest(captured)!=target['capture_sha256']:raise ValueError('capture changed')
                modules={s:replacement(weights,s) for s in target['active_sites']}
                prior={s:m for s,m in modules.items() if s!=site};row_stats=[];last_inputs=[]
                moments=torch.zeros((2,model.config.hidden_size),dtype=torch.float64);tokens=0
                with c.installed(model,'gpt2_gelu',prior),torch.no_grad():
                    for row in adaptation['encoded']['adapt']:
                        check();values=c.forward(model,'gpt2_gelu',row['ids'],[site],capture_all=True)['sites'][site]
                        x,y=values['x'],values['y'];prediction=modules[site](x)
                        a,b=prediction.double(),y.double();error=(a-b).square();reference=b.square()
                        row_stats.append([float(error[:-1].sum()),float(reference[:-1].sum()),error[:-1].numel(),
                                          float(error[-1].sum()),float(reference[-1].sum()),error[-1].numel()])
                        last_inputs.append(x[-1].numpy().copy());moments[0]+=b.sum(0);moments[1]+=b.square().sum(0);tokens+=len(x)
                if not np.array_equal(np.stack(last_inputs),captured['x']):raise ValueError('actual earlier replacement context differs')
                key=case+'/'+method+'/'+stage;rows=np.asarray(row_stats,dtype='float64');statistics[key]=rows
                value=summary(rows);variance=(moments[1]/tokens-(moments[0]/tokens).square()).clamp_min(1e-6)
                value['all_token_teacher_scale_rms']=float(variance.mean().sqrt())
                value['fitted_final_teacher_scale_rms']=float(modules[site].scale.double().square().mean().sqrt())
                value['scale_ratio']=value['all_token_teacher_scale_rms']/value['fitted_final_teacher_scale_rms']
                cells[key]=value
                print(json.dumps({'event':'adaptation_cell_complete','case':case,'method':method,'stage':stage}),flush=True)
    np.savez(root/'statistics.npz',**statistics)
    treatment=[value for key,value in cells.items() if '/jvp/' in key]
    outcome='COVERAGE_GAP_SUPPORTED' if all(v['twofold_coverage_gap'] for v in treatment) else 'MIXED_OR_UNSUPPORTED_COVERAGE_GAP'
    result={'protocol_id':PROTOCOL,'packet_sha256':packet,'state_slice':PROTOCOL,'outcome_status':outcome,
        'assurance_level':'INTERNAL_PASS','reviewer_status':'NOT_SUBMITTED','claim_ceiling':'ADAPTATION_TOKEN_COVERAGE_DIAGNOSIS_ONLY',
        'parent_packet_sha256':r.verify_source(),'parent_validation_sha256':receipt['validation_sha256'],
        'cells':cells,'statistics_sha256':c.array_digest(statistics),'statistics_file_sha256':r.sha(root/'statistics.npz'),
        'inputs':inputs,'runtime':runtime,'wall_seconds':time.monotonic()-start,
        'new_training_steps':0,'held_out_inputs_or_outcomes_read':False,'monetary_cost_usd':0,'energy':'NOT_MEASURED'}
    r.write(root/'result.json',result)
    if role=='replica':
        original=r.read(BASE/'primary/result.json');keys=('packet_sha256','parent_packet_sha256','parent_validation_sha256','cells','statistics_sha256','inputs','outcome_status')
        checks={key:original[key]==result[key] for key in keys}
        if not all(checks.values()) or original['runtime']['prefix']==runtime['prefix']:raise ValueError('diagnostic reproduction differs')
        if {k:v for k,v in original['runtime'].items() if k not in ('prefix','site_packages')}!={k:v for k,v in runtime.items() if k not in ('prefix','site_packages')}:raise ValueError('toolchain differs')
        for role2 in ('primary','replica'):
            data=r.read(BASE/role2/'result.json')
            if r.sha(BASE/role2/'statistics.npz')!=data['statistics_file_sha256']:raise ValueError('statistics changed')
        r.write(root/'reconciliation.json',{'checks':checks,'assurance_level':'REPRODUCED_LOCAL','same_owner_and_host':True,
                'primary_result_sha256':r.sha(BASE/'primary/result.json'),'replica_result_sha256':r.sha(root/'result.json')})
    return {'outcome_status':outcome,'cells':len(cells),'treatment_gap_cells':sum(v['twofold_coverage_gap'] for v in treatment),
            'result_sha256':r.sha(root/'result.json'),'assurance_level':'REPRODUCED_LOCAL' if role=='replica' else 'INTERNAL_PASS'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--role',choices=('primary','replica'),required=True)
    print(json.dumps(run(parser.parse_args().role),sort_keys=True))
