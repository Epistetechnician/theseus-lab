"""Reused GPT-2 development pilot on two predeclared early critical MLPs.

State slices: theseus-lab-sandbox-v1, theseus-reused-gpt2-early-ladder-v38,
global-experiment-assurance-lifecycle-v1. No network or newly selected guide.
"""
import argparse
import copy
import json
import shutil
import time

import numpy as np
import replacement_loss_v37r1_runner as predecessor

c,r=predecessor.c,predecessor.r
PARENT=predecessor.previous.BASE/'gpt2_gelu'
DIAGNOSTIC=r.ARTIFACTS/'theseus-lab-sandbox-v1-gpt2-visible-target-audit-v1'
OLD_VERIFY_SOURCE_MEMBERS=r.MEMBERS
c.PROTOCOL=c.SLICE='theseus-reused-gpt2-early-ladder-v38'
c.CLAIM='LOCAL_REUSED_GPT2_EARLY_TWO_MLP_REPLACEMENT_DEVELOPMENT_ONLY'
c.FAMILIES=('gpt2_gelu',)
SITES=(0,2)
r.BASE=r.ARTIFACTS/'theseus-lab-sandbox-v1-reused-gpt2-early-ladder-v38'
r.FREEZE=r.REPO/'docs/replacement-loss-v38-freeze.json'
r.MEMBERS=(*OLD_VERIFY_SOURCE_MEMBERS,'docs/replacement-loss-v37r1-freeze.json',
           'theseus_gpt2_target_audit_v1.py','docs/gpt2-visible-target-audit-v1.md',
           'tests/test_gpt2_target_audit_v1.py','docs/gpt2-visible-target-audit-v1-freeze.json',
           'replacement_loss_v38_runner.py','tests/test_replacement_loss_v38.py',
           'docs/replacement-loss-v38-reused-early-ladder.md')
ORIGINAL_INTAKE_VERIFY=predecessor.previous.adapter.ORIGINAL_VERIFY_INTAKE


def parents():
    result=r.read(PARENT/'closeout/validation.json')
    if result['scientific_outcome']!='NO_QUALIFIED_TARGET' or result['assurance_level']!='REPRODUCED_LOCAL':
        raise ValueError('parent stop not reproduced')
    primary=r.read(DIAGNOSTIC/'primary/result.json');replica=r.read(DIAGNOSTIC/'replica/result.json')
    reconciliation=r.read(DIAGNOSTIC/'replica/reconciliation.json')
    if primary['eligible_by_half']!=[[0,2],[]] or replica['metrics']!=primary['metrics']:
        raise ValueError('predeclared visible critical pair provenance differs')
    if len(reconciliation['checks'])!=9 or not all(v is True for v in reconciliation['checks'].values()):raise ValueError('diagnostic not reconciled')
    for role,value in (('primary',primary),('replica',replica)):
        if r.sha(DIAGNOSTIC/role/'result.json')!=reconciliation[role+'_result_sha256'] or r.sha(DIAGNOSTIC/role/'visible-ablations.npz')!=value['array_file_sha256']:
            raise ValueError('diagnostic bytes changed')
    return result


def intake():
    r.mount_guard();packet=r.verify_source();parents()
    if r.BASE.exists():raise FileExistsError('V38 root already exists')
    old=r.read(PARENT/'intake.json');sets={k:set(v) for k,v in old['exclusion'].items()}
    r.collect_history(r.read(PARENT/'primary/gpt2_gelu-visible.json'),sets)
    files=copy.deepcopy(old['historical_files'])
    for path in (PARENT/'intake.json',PARENT/'checkpoints.json',PARENT/'primary/result.json',PARENT/'replica/result.json',
                 PARENT/'closeout/validation.json',PARENT/'primary/gpt2_gelu-visible.json',
                 DIAGNOSTIC/'primary/result.json',DIAGNOSTIC/'replica/result.json',DIAGNOSTIC/'replica/reconciliation.json'):
        files.append({'path':str(path.relative_to(r.ARTIFACTS)),'bytes':path.stat().st_size,'sha256':r.sha(path),'kind':'JSON_IDENTITY_RECORD'})
    exclusion={k:sorted(v) for k,v in sets.items()}
    r.write(r.BASE/'intake.json',{'protocol_id':c.PROTOCOL,'packet_sha256':packet,
        'parent_packet_sha256':r.verify_parent(),'historical_files':files,
        'quarantined_legacy_exports':old['quarantined_legacy_exports'],'exclusion':exclusion,
        'exclusion_sha256':c.digest(exclusion),'claim_ceiling':c.CLAIM,
        'guide_reuse_declared_before_freeze':True,'sites_predeclared':list(SITES)})
    for member in (*r.MEMBERS,str(r.FREEZE.relative_to(r.REPO))):
        target=r.BASE/'frozen-packet'/member;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r.REPO/member,target)
    old_manifest=r.read(PARENT/'checkpoints.json');old_entry=old_manifest['checkpoints'][0]
    source=r.contained(PARENT,old_entry['local_path']);destination=r.BASE/'checkpoint';destination.mkdir()
    for row in old_entry['files']:
        path=r.contained(source,row['path'])
        if path.stat().st_size!=row['bytes'] or r.sha(path)!=row['sha256']:raise ValueError('approved reused weight bytes differ')
        target=destination/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    entry={k:copy.deepcopy(old_entry[k]) for k in ('family','model_id','revision','files')}
    entry.update({'local_path':'checkpoint','reused_development_checkpoint':True,'sites_predeclared':list(SITES)})
    r.write(r.BASE/'checkpoints.json',{'packet_sha256':packet,'intake_sha256':r.sha(r.BASE/'intake.json'),
            'approved_parent_manifest_sha256':r.sha(PARENT/'checkpoints.json'),'checkpoints':[entry]})
    return {'packet_sha256':packet,'intake_sha256':r.sha(r.BASE/'intake.json'),'checkpoint_manifest_sha256':r.sha(r.BASE/'checkpoints.json')}


def verify_intake():
    value=ORIGINAL_INTAKE_VERIFY();parents()
    return value


def manifest(path):
    history=verify_intake();value=r.read(path);old=r.read(PARENT/'checkpoints.json')['checkpoints'][0]
    if value['packet_sha256']!=r.verify_source() or value['intake_sha256']!=r.sha(r.BASE/'intake.json') or value['approved_parent_manifest_sha256']!=r.sha(PARENT/'checkpoints.json'):
        raise ValueError('reused guide/source binding')
    if len(value['checkpoints'])!=1:raise ValueError('one reused GPT-2 checkpoint required')
    entry=value['checkpoints'][0]
    if entry['reused_development_checkpoint'] is not True or entry['sites_predeclared']!=list(SITES):raise ValueError('reuse/site declaration')
    if any(entry[k]!=old[k] for k in ('family','model_id','revision','files')):raise ValueError('reused target identity differs')
    root=r.contained(r.BASE,entry['local_path']);actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!={row['path'] for row in entry['files']} or len(actual)!=len(entry['files']):raise ValueError('asset list incomplete')
    for row in entry['files']:
        p=r.contained(root,row['path'])
        if p.stat().st_size!=row['bytes'] or r.sha(p)!=row['sha256']:raise ValueError('reused asset changed')
    return value


def qualification(metrics):
    return all(m['removal_logit_nrmse']>=.1 and m['removal_top1_change']>=.2 for m in metrics.values())


def qualify(model,family,rows,check):
    if family!='gpt2_gelu':raise ValueError('GPT-2-only pilot')
    baseline=[];energy={s:0. for s in SITES}
    for row in rows:
        check();result=c.forward(model,family,row['ids'],SITES);baseline.append(result['logits'])
        for s in SITES:energy[s]+=float(result['sites'][s]['y'].square().mean())
    baseline=np.stack(baseline);metrics={}
    for site in SITES:
        outputs=[]
        with c.installed(model,family,{site:c.Zero()}):
            for row in rows:check();outputs.append(c.forward(model,family,row['ids'],SITES)['logits'])
        outputs=np.stack(outputs)
        metrics[str(site)]={'removal_logit_nrmse':c.nrmse(outputs,baseline),
                            'removal_top1_change':float(np.mean(outputs.argmax(1)!=baseline.argmax(1)))}
    return {'sites':list(SITES),'visible_energy':{str(s):v for s,v in energy.items()},'removal':metrics,'passed':qualification(metrics)}


r.intake=intake;r.verify_intake=verify_intake;r.checkpoint_manifest=manifest;r.qualify=qualify


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('verify','intake','run','closeout'))
    parser.add_argument('--role',choices=('primary','replica'),default='primary');args=parser.parse_args()
    if args.action=='verify':answer={'packet_sha256':r.verify_source()}
    elif args.action=='intake':answer=intake()
    elif args.action=='run':answer=r.run(r.BASE/'checkpoints.json',args.role)
    else:
        import replacement_loss_v36_closeout as audit
        answer=audit.validate();r.write(r.BASE/'closeout/validation.json',answer)
    print(json.dumps(answer,sort_keys=True))
