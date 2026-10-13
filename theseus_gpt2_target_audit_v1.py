"""All-layer ablations on the already visible V37R1 fixture only.

State slices: theseus-lab-sandbox-v1, theseus-gpt2-visible-target-audit-v1,
global-experiment-assurance-lifecycle-v1. No training, new data or holdouts.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import replacement_loss_v37r1_runner as parent

c,r=parent.c,parent.r
PROTOCOL='theseus-gpt2-visible-target-audit-v1'
BASE=r.ARTIFACTS/'theseus-lab-sandbox-v1-gpt2-visible-target-audit-v1'
FREEZE=r.REPO/'docs/gpt2-visible-target-audit-v1-freeze.json'
MEMBERS=('theseus_gpt2_target_audit_v1.py','docs/gpt2-visible-target-audit-v1.md',
         'tests/test_gpt2_target_audit_v1.py','docs/replacement-loss-v37r1-freeze.json')


def members():
    return [{'path':p,'bytes':(r.REPO/p).stat().st_size,'sha256':r.sha(r.REPO/p)} for p in MEMBERS]


def verify():
    value=r.read(FREEZE)
    if value['members']!=members() or value['packet_sha256']!=c.digest(members()):raise ValueError('diagnostic source differs')
    return value['packet_sha256']


def choose(metrics,depth):
    halves=(range(depth//2),range(depth//2,depth));eligible=[];selected=[]
    for half in halves:
        passing=[s for s in half if metrics[str(s)]['removal_logit_nrmse']>=.1 and metrics[str(s)]['removal_top1_change']>=.2]
        eligible.append(passing)
        if passing:selected.append(min(passing,key=lambda s:(-metrics[str(s)]['removal_logit_nrmse'],s)))
    return eligible,selected if len(selected)==2 else None


def run(role):
    r.mount_guard();packet=verify();parent.previous.activate('gpt2_gelu')
    manifest_path=r.BASE/'checkpoints.json';manifest=r.checkpoint_manifest(manifest_path)
    old=r.read(r.BASE/'primary/result.json')
    if old['outcome_status']!='NO_QUALIFIED_TARGET' or old['challenge_generated'] or old['metrics']:
        raise ValueError('parent was not the retained pre-fit stop')
    root=BASE/role;root.mkdir(parents=True,exist_ok=False);start=time.monotonic();runtime=r.runtime()
    def check():
        r.mount_guard()
        if time.monotonic()-start>1800 or sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>1024**3:
            raise RuntimeError('diagnostic cap')
    visible=r.BASE/'primary/gpt2_gelu-visible.json';rows=r.read(visible)['encoded']
    if len(rows)!=64:raise ValueError('exact visible corpus required')
    model,tokenizer=r.load_model(manifest['checkpoints'][0]);depth=len(c.layers(model,'gpt2_gelu'))
    baseline=[]
    for row in rows:check();baseline.append(c.forward(model,'gpt2_gelu',row['ids'],[])['logits'])
    baseline=np.stack(baseline);metrics={};removed=[]
    for site in range(depth):
        outputs=[]
        with c.installed(model,'gpt2_gelu',{site:c.Zero()}):
            for row in rows:check();outputs.append(c.forward(model,'gpt2_gelu',row['ids'],[])['logits'])
        outputs=np.stack(outputs);removed.append(outputs)
        metrics[str(site)]={'removal_logit_nrmse':c.nrmse(outputs,baseline),
                           'removal_top1_change':float(np.mean(outputs.argmax(1)!=baseline.argmax(1)))}
    data={'baseline':baseline,'removed':np.stack(removed)};np.savez(root/'visible-ablations.npz',**data)
    eligible,selected=choose(metrics,depth)
    report={'protocol_id':PROTOCOL,'packet_sha256':packet,'state_slice':PROTOCOL,
        'assurance_level':'INTERNAL_PASS','reviewer_status':'NOT_SUBMITTED',
        'outcome_status':'VISIBLE_CRITICAL_PAIR_AVAILABLE' if selected else 'NO_VISIBLE_CRITICAL_PAIR',
        'claim_ceiling':'REUSED_VISIBLE_TARGET_DIAGNOSIS_ONLY',
        'parent_packet_sha256':r.verify_source(),'parent_result_sha256':r.sha(r.BASE/'primary/result.json'),
        'visible_sha256':r.sha(visible),'checkpoint_manifest_sha256':r.sha(manifest_path),
        'metrics':metrics,'eligible_by_half':eligible,'recommended_sites':selected,
        'array_sha256':c.array_digest(data),'array_file_sha256':r.sha(root/'visible-ablations.npz'),
        'runtime':runtime,'wall_seconds':time.monotonic()-start,'monetary_cost_usd':0,'energy':'NOT_MEASURED',
        'held_out_outcomes_opened':False,'replacements_trained':0}
    r.write(root/'result.json',report)
    if role=='replica':
        primary=r.read(BASE/'primary/result.json')
        keys=('packet_sha256','parent_packet_sha256','parent_result_sha256','visible_sha256',
              'checkpoint_manifest_sha256','metrics','eligible_by_half','recommended_sites','array_sha256')
        checks={k:primary[k]==report[k] for k in keys}
        for role2 in ('primary','replica'):
            value=r.read(BASE/role2/'result.json')
            if r.sha(BASE/role2/'visible-ablations.npz')!=value['array_file_sha256']:raise ValueError('diagnostic arrays changed')
        if primary['runtime']['prefix']==runtime['prefix'] or not all(checks.values()):raise ValueError('diagnostic replica mismatch')
        r.write(root/'reconciliation.json',{'checks':checks,'assurance_level':'REPRODUCED_LOCAL',
            'primary_result_sha256':r.sha(BASE/'primary/result.json'),'replica_result_sha256':r.sha(root/'result.json'),
            'same_owner_and_host':True,'outcome_status':report['outcome_status']})
    return {'outcome_status':report['outcome_status'],'eligible_by_half':eligible,'recommended_sites':selected,
            'result_sha256':r.sha(root/'result.json'),'assurance_level':'REPRODUCED_LOCAL' if role=='replica' else 'INTERNAL_PASS'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--role',choices=('primary','replica'),required=True)
    print(json.dumps(run(parser.parse_args().role),sort_keys=True))
