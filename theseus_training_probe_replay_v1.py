"""Project saved Jacobians on reconstructed V39 training/fresh directions.

State slices: theseus-lab-sandbox-v1, theseus-v39-training-probe-replay-v1,
global-experiment-assurance-lifecycle-v1. No model forward or training calls.
"""
import argparse
import hashlib
import json
import math
import shutil
import time

import numpy as np
import torch
import theseus_first_mlp_diagnosis_v1 as previous

c,r=previous.c,previous.r
PROTOCOL='theseus-v39-training-probe-replay-v1'
CLAIM='ADAPTATION_TRAINING_DIRECTION_RESIDUAL_DIAGNOSIS_ONLY'
BASE=r.ARTIFACTS/'theseus-lab-sandbox-v1-v39-training-probe-replay-v1'
FREEZE=r.REPO/'docs/v39-training-probe-replay-v1-freeze.json'
MEMBERS=('theseus_training_probe_replay_v1.py','tests/test_training_probe_replay_v1.py',
    'docs/v39-training-probe-replay-v1.md','docs/v39-first-mlp-diagnosis-v1-freeze.json',
    'docs/v39-first-mlp-diagnosis-v1-results.json')


def directions(count,width,candidate_seed):
    """Original core fit consumes coordinate-mask RNG before four probes."""
    if min(count,width)<1:raise ValueError('probe dimensions')
    generator=torch.Generator().manual_seed(candidate_seed)
    torch.randint(width,(count,),generator=generator)
    return (torch.randint(2,(4,count,width),generator=generator).float()*2-1)/math.sqrt(width)


def project(jacobians,probe_directions):
    j=np.asarray(jacobians,dtype=np.float64);d=np.asarray(probe_directions,dtype=np.float64)
    if j.ndim!=3 or d.ndim!=3 or j.shape[0]!=d.shape[0] or j.shape[2]!=d.shape[2] or not np.isfinite(j).all() or not np.isfinite(d).all():
        raise ValueError('projection shape/value contract')
    return np.einsum('noi,npi->npo',j,d,optimize=False)


def member_rows():
    return [{'path':p,'bytes':(r.REPO/p).stat().st_size,'sha256':r.sha(r.REPO/p)} for p in MEMBERS]


def verify():
    previous.verify();previous.verify_archive()
    value=r.read(FREEZE)
    if value['members']!=member_rows() or value['packet_sha256']!=c.digest(member_rows()) or value['protocol_id']!=PROTOCOL:raise ValueError('replay source differs')
    parent=r.read(r.REPO/'docs/v39-first-mlp-diagnosis-v1-results.json')
    if parent['assurance_level']!='REPRODUCED_LOCAL' or r.sha(previous.BASE/'closeout/validation.json')!=parent['validation_sha256']:
        raise ValueError('parent diagnosis not reproduced/bound')
    for role in ('primary','replica'):
        if r.sha(previous.BASE/role/'result.json')!=parent[role+'_result_sha256']:raise ValueError('parent result differs')
    return value['packet_sha256']


def metrics(arrays):
    result={}
    keys={key.rsplit('/',1)[0] for key in arrays}
    for key in sorted(keys):
        teacher=arrays[key+'/teacher'];student=arrays[key+'/student']
        result[key]={'projected_nrmse':float(np.sqrt(np.sum((student-teacher)**2)/max(float(np.sum(teacher**2)),teacher.size*1e-8))),
                     'reference_rms':float(np.sqrt(np.mean(teacher**2))),'projected_vectors':32}
    selected=[row['projected_nrmse'] for key,row in result.items() if '/jvp/final/training' in key]
    return {'cells':result,'outcome_status':'TRAINING_DIRECTION_UNDERFIT_SUPPORTED' if len(selected)==3 and all(x>.20 for x in selected) else 'DIRECTION_GENERALIZATION_OR_MIXED'}


def run(role):
    start=time.monotonic();packet=verify();r.mount_guard();runtime=r.runtime()
    parent_result=r.read(previous.BASE/'primary/result.json')
    if {k:v for k,v in runtime.items() if k not in ('prefix','site_packages')}!={k:v for k,v in parent_result['runtime'].items() if k not in ('prefix','site_packages')}:raise ValueError('parent runtime differs')
    source_members=(*MEMBERS,str(FREEZE.relative_to(r.REPO)))
    if role=='primary':
        if BASE.exists():raise FileExistsError('replay root exists; no same-root retry')
        for member in source_members:
            path=BASE/'frozen-packet'/member;path.parent.mkdir(parents=True,exist_ok=True)
            r.mount_guard();shutil.copyfile(r.REPO/member,path)
    else:
        for member in source_members:
            if r.sha(BASE/'frozen-packet'/member)!=r.sha(r.REPO/member):raise ValueError('replay source archive differs')
    root=BASE/role;root.mkdir(parents=True,exist_ok=False);arrays={};bindings=[];traces={}
    index=r.read(previous.BASE/'primary/selection.json')
    if r.sha(previous.BASE/'primary/selection.json')!=parent_result['selection_sha256'] or r.sha(previous.BASE/'primary/arrays.npz')!=parent_result['arrays_file_sha256']:raise ValueError('parent arrays/selection differ')
    with np.load(previous.BASE/'primary/arrays.npz',allow_pickle=False) as jacobians:
        for case,chosen in index['cases'].items():
            case_seed=int(case.rsplit('-',1)[1]);candidate_seed=c.seed_for('gpt2_gelu',case_seed,0,'candidate')
            for method in c.METHODS:
                stem=previous.PARENT/'primary'/case/(method+'-block_1')
                captured=r.arrays(stem.with_name(stem.name+'-adaptation-inputs.npz'));trace=r.arrays(stem.with_name(stem.name+'-trace.npz'))
                binding=r.read(previous.PARENT/'primary/candidate-freeze.json')['families']['gpt2_gelu']['cases'][case]['arms'][method]['block_1']
                if c.array_digest(captured)!=binding['capture_sha256'] or c.array_digest(trace)!=binding['trace_sha256']:raise ValueError('V39 fitting inputs/trace differ')
                lengths=captured['lengths'][:,0];offsets=np.r_[0,np.cumsum(lengths)];training=directions(int(offsets[-1]),768,candidate_seed)
                fresh_seed=int.from_bytes(hashlib.sha256(c.canonical([PROTOCOL,case,'fresh'])).digest()[:8],'big')%(2**31-1)
                fresh=directions(int(offsets[-1]),768,fresh_seed)
                for label in previous.POSITIONS:
                    row_indices=[int(offsets[p])+chosen['positions'][str(p)+'/'+label] for p in chosen['selected_prompt_indices']]
                    for kind,all_probes in (('training',training),('fresh',fresh)):
                        d=all_probes[:,row_indices,:].permute(1,0,2).numpy()
                        key=case+'/'+method+'/'+label+'/'+kind
                        for name in ('teacher','student'):arrays[key+'/'+name]=project(jacobians[case+'/'+method+'/'+label+'/'+name+'_jacobian'],d)
                coeff=trace['readouts'][:,2]
                traces[case+'/'+method]={'coefficient_min':float(coeff.min()),'coefficient_median':float(np.median(coeff)),
                                       'coefficient_max':float(coeff.max()),'fraction_coefficient_below_one':float(np.mean(coeff<1))}
                bindings.append({'case':case,'method':method,'capture_sha256':binding['capture_sha256'],'trace_sha256':binding['trace_sha256'],'candidate_seed':candidate_seed})
                previous.guard(start,root)
    summary=metrics(arrays);r.mount_guard();r.save(root/'projections.npz',arrays)
    if verify()!=packet:raise ValueError('replay packet changed')
    result={'protocol_id':PROTOCOL,'state_slice':PROTOCOL,'packet_sha256':packet,'claim_ceiling':CLAIM,
        'assurance_level':'INTERNAL_PASS','reviewer_status':'NOT_SUBMITTED','metrics':summary,'trace_summaries':traces,'bindings':bindings,
        'parent_validation_sha256':r.sha(previous.BASE/'closeout/validation.json'),'parent_arrays_sha256':parent_result['arrays_sha256'],
        'arrays_sha256':c.array_digest(arrays),'arrays_file_sha256':r.sha(root/'projections.npz'),'runtime':runtime,
        'resources':previous.guard(start,root),'new_model_forwards':0,'new_training_steps':0,'new_heldout_queries':0}
    r.mount_guard();r.write(root/'result.json',result)
    if role=='replica':
        p=r.read(BASE/'primary/result.json');keys=('protocol_id','packet_sha256','metrics','trace_summaries','bindings','parent_validation_sha256','parent_arrays_sha256','arrays_sha256')
        checks={key:p[key]==result[key] for key in keys}
        if not all(checks.values()) or p['runtime']['prefix']==runtime['prefix'] or p['runtime']['site_packages']==runtime['site_packages']:raise ValueError('probe replay does not reproduce')
        if {k:v for k,v in p['runtime'].items() if k not in ('prefix','site_packages')}!={k:v for k,v in runtime.items() if k not in ('prefix','site_packages')}:raise ValueError('replay toolchain differs')
        r.mount_guard();r.write(root/'reconciliation.json',{'checks':checks,'assurance_level':'REPRODUCED_LOCAL','same_owner_and_host':True,
            'primary_result_sha256':r.sha(BASE/'primary/result.json'),'replica_result_sha256':r.sha(root/'result.json')})
    return {'outcome_status':summary['outcome_status'],'result_sha256':r.sha(root/'result.json'),'assurance_level':'REPRODUCED_LOCAL' if role=='replica' else 'INTERNAL_PASS'}


def closeout():
    verify();rows={}
    for member in (*MEMBERS,str(FREEZE.relative_to(r.REPO))):
        if r.sha(BASE/'frozen-packet'/member)!=r.sha(r.REPO/member):raise ValueError('replay archive differs')
    for role in ('primary','replica'):
        result=r.read(BASE/role/'result.json');data=r.arrays(BASE/role/'projections.npz')
        if r.sha(BASE/role/'projections.npz')!=result['arrays_file_sha256'] or c.array_digest(data)!=result['arrays_sha256']:raise ValueError('projection binding')
        if len(data)!=72 or any(a.shape!=(8,4,768) or not np.isfinite(a).all() for a in data.values()):raise ValueError('projection cohort shape')
        for key,reported in result['metrics']['cells'].items():
            p,g=data[key+'/student'],data[key+'/teacher']
            value=float(np.sqrt(np.mean(np.square(p-g))/max(float(np.mean(g*g)),1e-8)))
            if not math.isclose(value,reported['projected_nrmse'],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('independent projection formula differs')
        if metrics(data)['outcome_status']!=result['metrics']['outcome_status']:raise ValueError('replay decision differs')
        rows[role]={'projection_cells':36,'arrays_sha256':result['arrays_sha256']}
    rec=r.read(BASE/'replica/reconciliation.json')
    if len(rec['checks'])!=8 or not all(rec['checks'].values()):raise ValueError('replay reconciliation')
    p,q=[r.read(BASE/role/'result.json') for role in ('primary','replica')]
    if any(p[key]!=q[key] for key in rec['checks']):raise ValueError('replay reconciliation values differ')
    if p['runtime']['prefix']==q['runtime']['prefix'] or p['runtime']['site_packages']==q['runtime']['site_packages']:raise ValueError('replay runtime paths')
    for role in ('primary','replica'):
        if r.sha(BASE/role/'result.json')!=rec[role+'_result_sha256']:raise ValueError('replay result digest')
    answer={'protocol_id':PROTOCOL,'state_slice':PROTOCOL,'packet_sha256':verify(),'assurance_level':'REPRODUCED_LOCAL',
        'reviewer_status':'NOT_SUBMITTED','claim_ceiling':CLAIM,'outcome_status':r.read(BASE/'primary/result.json')['metrics']['outcome_status'],
        'audits':rows,'same_owner_and_host':True,'reconciliation_sha256':r.sha(BASE/'replica/reconciliation.json')}
    r.mount_guard();r.write(BASE/'closeout/validation.json',answer);return answer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('verify','run','closeout'))
    parser.add_argument('--role',choices=('primary','replica'),default='primary');args=parser.parse_args()
    if args.action=='verify':result={'packet_sha256':verify()}
    elif args.action=='run':result=run(args.role)
    else:result=closeout()
    print(json.dumps(result,sort_keys=True))
