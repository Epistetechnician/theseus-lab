"""Frozen adaptation-only Jacobian and decoder error accounting for V39.

State slices: theseus-lab-sandbox-v1, theseus-v39-first-mlp-causal-diagnosis-v1,
global-experiment-assurance-lifecycle-v1. No training or held-out fixtures.
"""
import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import sys
import time

import numpy as np
import torch
import replacement_loss_v39_runner as parent

c,r=parent.c,parent.r
PROTOCOL='theseus-v39-first-mlp-causal-diagnosis-v1'
CLAIM='ADAPTATION_FIRST_MLP_LOCAL_AND_CONTEXT_ERROR_DIAGNOSIS_ONLY'
PARENT=r.BASE
BASE=r.ARTIFACTS/'theseus-lab-sandbox-v1-v39-first-mlp-causal-diagnosis-v1'
FREEZE=r.REPO/'docs/v39-first-mlp-diagnosis-v1-freeze.json'
MEMBERS=('theseus_first_mlp_diagnosis_v1.py','tests/test_first_mlp_diagnosis_v1.py',
         'docs/v39-first-mlp-diagnosis-v1.md','docs/replacement-loss-v39-freeze.json',
         'docs/replacement-loss-v39-results.json','theseus_first_mlp_diagnosis_v1_closeout.py')
POSITIONS=('first','middle_prefix','final')
COUNTS=(1,2,4,8)
SCALES=(1.,1.4)
PROMPTS=8
MAX_SECONDS=3600
MAX_RSS=6*1024**3
MAX_BYTES=2*1024**3


def seed(*parts):
    return int.from_bytes(hashlib.sha256(c.canonical([PROTOCOL,*parts])).digest()[:8],'big')%(2**31-1)


def selection(case,count=128):
    if count<PROMPTS:raise ValueError('adaptation cohort too small')
    return sorted(torch.randperm(count,generator=torch.Generator().manual_seed(seed(case,'selection')))[:PROMPTS].tolist())


def position(length,label):
    if length<3 or label not in POSITIONS:raise ValueError('token position')
    return {'first':0,'middle_prefix':(length-1)//2,'final':length-1}[label]


def patched(x,donor,count,scale,key):
    if x.ndim!=1 or donor.shape!=x.shape or not 1<=count<=len(x):raise ValueError('finite patch shape')
    indices=torch.randperm(len(x),generator=torch.Generator().manual_seed(seed(*key,count,scale,'patch')))[:count]
    value=x.clone();value[indices]=x[indices]+scale*(donor[indices]-x[indices])
    return value


def sums(actual,reference):
    a,b=np.asarray(actual,dtype=np.float64),np.asarray(reference,dtype=np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('numeric value contract')
    return [float(np.square(a-b).sum()),float(np.square(b).sum()),int(b.size)]


def normalized(rows):
    error,reference,count=np.asarray(rows,dtype=np.float64).sum(0)
    if count<=0 or min(error,reference)<0 or not np.isfinite([error,reference,count]).all():raise ValueError('numeric sums contract')
    return math.sqrt(error/max(reference,count*1e-8))


def effects(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    if a.ndim!=2 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('effect shapes/values')
    norms=np.linalg.norm(b,axis=1);valid=norms>=1e-4
    if not valid.any():return {'resolved_fraction':0.,'median_relative_error':None,'p90_relative_error':None,'median_cosine':None}
    a,b=a[valid],b[valid];errors=np.linalg.norm(a-b,axis=1)/np.maximum(np.linalg.norm(b,axis=1),1e-6)
    cosine=np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
    return {'resolved_fraction':float(valid.mean()),'median_relative_error':float(np.median(errors)),
            'p90_relative_error':float(np.quantile(errors,.9)),'median_cosine':float(np.median(cosine))}


def jacobian(module,x,key):
    """Output-by-input Jacobian, checked with independently computed JVPs."""
    value=torch.func.jacrev(module)(x).detach()
    if value.shape!=(len(x),len(x)) or not torch.isfinite(value).all():raise ValueError('full Jacobian contract')
    gen=torch.Generator().manual_seed(seed(*key,'jacobian_checks'))
    directions=(torch.randint(2,(4,len(x)),generator=gen).float()*2-1)/math.sqrt(len(x))
    for direction in directions:
        actual=torch.func.jvp(module,(x,),(direction,))[1].detach()
        predicted=value@direction
        relative=float(torch.linalg.vector_norm((predicted-actual).double())/torch.linalg.vector_norm(actual.double()).clamp_min(1e-8))
        if relative>1e-5 or not math.isfinite(relative):raise ValueError('Jacobian orientation/JVP mismatch')
    return value.cpu().numpy().copy()


def components(actual,guide,student_guide,anchored):
    """Ordered vector accounting; the component norms are not additive."""
    values=[np.asarray(v,dtype=np.float64) for v in (actual,guide,student_guide,anchored)]
    if len({v.shape for v in values})!=1 or not all(np.isfinite(v).all() for v in values):raise ValueError('transport values')
    a,g,s,h=values
    parts={'prefix_context':a-s,'factual_anchor':s-h,'local_delta_transport':h-g}
    total=a-g
    if not np.allclose(sum(parts.values()),total,rtol=0,atol=1e-12):raise ValueError('decoder accounting identity')
    return total,parts


def members():
    return [{'path':p,'bytes':(r.REPO/p).stat().st_size,'sha256':r.sha(r.REPO/p)} for p in MEMBERS]


def archive_members():
    return sorted(set((*r.MEMBERS,*MEMBERS,str(FREEZE.relative_to(r.REPO)))))


def verify_archive():
    for member in archive_members():
        if r.sha(BASE/'frozen-packet'/member)!=r.sha(r.REPO/member):raise ValueError('diagnostic source archive changed')


def verify():
    value=r.read(FREEZE)
    if value['protocol_id']!=PROTOCOL or value['claim_ceiling']!=CLAIM or value['members']!=members() or value['packet_sha256']!=c.digest(members()):
        raise ValueError('diagnostic source/protocol changed')
    r.verify_source()
    report=r.read(r.REPO/'docs/replacement-loss-v39-results.json')
    if report['scientific_outcome']!='FAIL' or report['assurance_level']!='REPRODUCED_LOCAL':raise ValueError('parent terminal state')
    for field,path in (('validation_sha256','closeout/validation.json'),('candidate_freeze_sha256','primary/candidate-freeze.json'),
                       ('checkpoint_manifest_sha256','checkpoints.json'),('primary_result_sha256','primary/result.json'),
                       ('replica_result_sha256','replica/result.json'),('reconciliation_sha256','replica/reconciliation.json')):
        if r.sha(PARENT/path)!=report[field]:raise ValueError('parent terminal binding')
    return value['packet_sha256']


def guard(start,root):
    r.mount_guard()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    if time.monotonic()-start>MAX_SECONDS or rss>MAX_RSS or size>MAX_BYTES:raise RuntimeError('diagnostic resource cap')
    if shutil.disk_usage(r.ARTIFACTS).free<5*1024**3:raise RuntimeError('diagnostic free-space reserve')
    return {'wall_seconds':time.monotonic()-start,'peak_rss_bytes':rss,'artifact_bytes':size,
            'paid_execution_usd':0,'energy':'NOT_MEASURED'}


def guide_manifest():
    """Verify the used guide without reopening ancestor held-out files."""
    r.mount_guard();value=r.read(PARENT/'checkpoints.json')
    report=r.read(r.REPO/'docs/replacement-loss-v39-results.json')
    if value['packet_sha256']!=r.verify_source() or r.sha(PARENT/'checkpoints.json')!=report['checkpoint_manifest_sha256'] or len(value['checkpoints'])!=1:
        raise ValueError('approved parent guide binding')
    entry=value['checkpoints'][0];expected=report['checkpoint']
    if entry['family']!=expected['architecture_family'] or entry['model_id']!=expected['model_id'] or entry['revision']!=expected['revision']:
        raise ValueError('approved guide identity differs')
    root=r.contained(PARENT,entry['local_path'])
    if {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}!={row['path'] for row in entry['files']}:
        raise ValueError('guide file inventory differs')
    for row in entry['files']:
        path=r.contained(root,row['path'])
        if path.stat().st_size!=row['bytes'] or r.sha(path)!=row['sha256']:raise ValueError('guide bytes differ')
    return value


def write(path,value):
    r.mount_guard()
    if not path.resolve().is_relative_to(BASE.resolve()):raise ValueError('diagnostic write boundary')
    r.write(path,value)


def restore(weights):
    if any(not key.startswith('0/') for key in weights):raise ValueError('first-stage-only weights')
    state={key[2:]:torch.from_numpy(value.copy()) for key,value in weights.items()}
    module=c.Replacement(len(state['mean']),'gpt2_gelu',0,state['mean'],state['scale'])
    module.load_state_dict(state,strict=True)
    return module.eval().requires_grad_(False)


def summarize(data,index):
    cells={}
    for case in index['cases']:
        for method in c.METHODS:
            for label in POSITIONS:
                key=case+'/'+method+'/'+label;prefix=key+'/'
                zero_control=effects(np.zeros_like(data[prefix+'teacher_delta']),data[prefix+'teacher_delta'])
                if zero_control['resolved_fraction']>0 and abs(zero_control['median_relative_error']-1.)>1e-12:raise ValueError('zero-delta control')
                cells[key]={'value_nrmse':normalized([sums(a,b) for a,b in zip(data[prefix+'student_value'],data[prefix+'teacher_value'])]),
                            'jacobian_nrmse':normalized([sums(a,b) for a,b in zip(data[prefix+'student_jacobian'],data[prefix+'teacher_jacobian'])]),
                            'finite_effects':effects(data[prefix+'student_delta'],data[prefix+'teacher_delta']),
                            'analytic_zero_delta_control':zero_control,
                            'sampled_prompts':PROMPTS,'finite_patches':PROMPTS*len(COUNTS)*len(SCALES)}
    transport={}
    for case in index['cases']:
        for method in c.METHODS:
            prefix=case+'/'+method+'/transport/';a,g,s,h=[data[prefix+k] for k in ('actual','guide','student_guide','anchored')]
            total,parts=components(a,g,s,h);den=float(np.square(total).sum())
            transport[case+'/'+method]={'actual_effects':effects(a,g),'student_in_guide_context':effects(s,g),
                'anchored_student_delta':effects(h,g),
                'component_relative_rms':{k:math.sqrt(float(np.square(v).sum())/max(den,1e-12)) for k,v in parts.items()},
                'error_projection_shares':{k:float(np.sum(v*total))/den if den>=1e-12 else None for k,v in parts.items()},
                'projection_shares_are_ordered_not_unique_causal_attribution':True}
    final=[cells[case+'/jvp/final'] for case in index['cases']]
    mismatch=all(v['jacobian_nrmse']>.20 and v['finite_effects']['resolved_fraction']>=.90
                 and v['finite_effects']['median_relative_error'] is not None and v['finite_effects']['median_relative_error']>.20 for v in final)
    return {'cells':cells,'transport':transport,'outcome_status':'LOCAL_DERIVATIVE_MISMATCH_SUPPORTED' if mismatch else 'MIXED_OR_LOCAL_FIT_REQUIRES_FURTHER_DIAGNOSIS'}


def run(role):
    start=time.monotonic();packet=verify();guard(start,BASE)
    manifest=guide_manifest();runtime=r.runtime();guard(start,BASE)
    parent_artifacts=r.read(PARENT/'primary/result.json')['artifacts']
    runtime_binding=next(row for row in parent_artifacts if row['path']=='runtime.json')
    if r.sha(PARENT/'primary/runtime.json')!=runtime_binding['sha256']:raise ValueError('parent runtime artifact changed')
    prior_runtime=r.read(PARENT/'primary/runtime.json')
    if {k:v for k,v in runtime.items() if k not in ('prefix','site_packages')}!={k:v for k,v in prior_runtime.items() if k not in ('prefix','site_packages')}:
        raise ValueError('runtime differs from the parent study')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    if role=='primary':
        if BASE.exists():raise FileExistsError('diagnostic root exists; no same-root retry')
        for member in archive_members():
            target=BASE/'frozen-packet'/member;target.parent.mkdir(parents=True,exist_ok=True)
            r.mount_guard();shutil.copyfile(r.REPO/member,target)
    else:
        verify_archive()
    root=BASE/role;root.mkdir(parents=True,exist_ok=False)
    try:
        model,tokenizer=r.load_model(manifest['checkpoints'][0]);teacher=c.original_mlp(model,'gpt2_gelu',0)
        teacher_digest=c.array_digest(c.state(teacher))
        freeze=r.read(PARENT/'primary/candidate-freeze.json');data={};index={'cases':{},'input_bindings':[],'patch_order':[[n,z] for n in COUNTS for z in SCALES],
            'native_width':model.config.hidden_size,'vocab_size':model.config.vocab_size}
        for case,binding in freeze['families']['gpt2_gelu']['cases'].items():
            folder=PARENT/'primary'/case;adaptation=r.read(folder/'adaptation.json');encoded=adaptation['encoded']
            selected=selection(case);index['cases'][case]={'selected_prompt_indices':selected,'positions':{}}
            index['input_bindings'].append({'path':str((folder/'adaptation.json').relative_to(r.ARTIFACTS)),'sha256':r.sha(folder/'adaptation.json')})
            for method in c.METHODS:
                target=binding['arms'][method]['block_1'];stem=folder/(method+'-block_1')
                if target['active_sites']!=[0]:raise ValueError('first-stage target')
                if teacher_digest!=target['original_target_digests']['0']:raise ValueError('loaded original MLP differs')
                weights=r.arrays(stem.with_name(stem.name+'-weights.npz'));captures=r.arrays(stem.with_name(stem.name+'-adaptation-inputs.npz'))
                if c.array_digest(weights)!=target['weights_sha256'] or c.array_digest(captures)!=target['capture_sha256']:raise ValueError('parent candidate/capture changed')
                student=restore(weights);offsets=np.r_[0,np.cumsum(captures['lengths'][:,0])];donor_offsets=np.r_[0,np.cumsum(captures['lengths'][:,1])]
                local={label:{k:[] for k in ('teacher_value','student_value','teacher_jacobian','student_jacobian','teacher_delta','student_delta')} for label in POSITIONS}
                transported={key:[] for key in ('actual','guide','student_guide','anchored','local_teacher_delta','local_student_delta')};noop_errors=[]
                for prompt in selected:
                    guard(start,root);row=encoded['adapt'][prompt];donor=encoded['donor'][prompt]
                    original=c.forward(model,'gpt2_gelu',row['ids'],[0],capture_all=True)
                    donor_values=c.forward(model,'gpt2_gelu',donor['ids'],[0],capture_all=True)['sites'][0]['x']
                    values=original['sites'][0];xfull=values['x'];length=len(xfull)
                    if not np.array_equal(xfull.numpy(),captures['x'][offsets[prompt]:offsets[prompt+1]]) or not np.array_equal(donor_values.numpy(),captures['donor_source'][donor_offsets[prompt]:donor_offsets[prompt+1]]):
                        raise ValueError('actual first-MLP adaptation context differs')
                    with c.installed(model,'gpt2_gelu',{0:student}):
                        factual=c.forward(model,'gpt2_gelu',row['ids'],[0])
                    teacher_base=values['y'][-1];student_base=factual['sites'][0]['y']
                    guide_logits=original['logits'].astype(np.float64);actual_logits=factual['logits'].astype(np.float64)
                    for label in POSITIONS:
                        pos=position(length,label);index['cases'][case]['positions'][str(prompt)+'/'+label]=pos
                        x=xfull[pos];d=donor_values[pos*(len(donor_values)-1)//(length-1)]
                        arrays=local[label]
                        with torch.no_grad():yt,ys=teacher(x),student(x)
                        arrays['teacher_value'].append(yt.numpy().copy());arrays['student_value'].append(ys.numpy().copy())
                        arrays['teacher_jacobian'].append(jacobian(teacher,x,(case,prompt,label,'teacher')))
                        arrays['student_jacobian'].append(jacobian(student,x,(case,prompt,label,'student')))
                        for count in COUNTS:
                            for scale in SCALES:
                                guard(start,root);xp=patched(x,d,count,scale,(case,prompt,label))
                                with torch.no_grad():dt=teacher(xp)-yt;ds=student(xp)-ys
                                arrays['teacher_delta'].append(dt.numpy().copy());arrays['student_delta'].append(ds.numpy().copy())
                                if label!='final':continue
                                oracle=c.forward(model,'gpt2_gelu',row['ids'],[0],(0,'input',xp))
                                with c.installed(model,'gpt2_gelu',{0:student}):
                                    candidate=c.forward(model,'gpt2_gelu',row['ids'],[0],(0,'input',xp))
                                local_t=oracle['sites'][0]['y']-teacher_base;local_s=candidate['sites'][0]['y']-student_base
                                sg_base=c.forward(model,'gpt2_gelu',row['ids'],[0],(0,'output',student_base))['logits'].astype(np.float64)
                                sg_patch=c.forward(model,'gpt2_gelu',row['ids'],[0],(0,'output',candidate['sites'][0]['y']))['logits'].astype(np.float64)
                                anchor=c.forward(model,'gpt2_gelu',row['ids'],[0],(0,'output',teacher_base+local_s))['logits'].astype(np.float64)
                                transported['actual'].append(candidate['logits'].astype(np.float64)-actual_logits)
                                transported['guide'].append(oracle['logits'].astype(np.float64)-guide_logits)
                                transported['student_guide'].append(sg_patch-sg_base)
                                transported['anchored'].append(anchor-guide_logits)
                                transported['local_teacher_delta'].append(local_t.numpy().copy());transported['local_student_delta'].append(local_s.numpy().copy())
                                if count==1 and scale==1.:
                                    with c.installed(model,'gpt2_gelu',{0:teacher}):
                                        noop=c.forward(model,'gpt2_gelu',row['ids'],[0],(0,'input',xp))
                                    error=float(np.max(np.abs(noop['logits'].astype(np.float64)-oracle['logits'].astype(np.float64))))
                                    if error!=0:raise ValueError('actual no-op control differs')
                                    noop_errors.append(error)
                    print(json.dumps({'event':'diagnostic_prompt_complete','case':case,'method':method,'prompt_index':prompt}),flush=True)
                for label,arrays in local.items():
                    for key,rows in arrays.items():data[case+'/'+method+'/'+label+'/'+key]=np.stack(rows)
                for key,rows in transported.items():data[case+'/'+method+'/transport/'+key]=np.stack(rows)
                data[case+'/'+method+'/noop_errors']=np.asarray(noop_errors,dtype=np.float64)
                guard(start,root)
        summary=summarize(data,index)
        write(root/'selection.json',index);guard(start,root)
        if (root/'arrays.npz').exists():raise FileExistsError('diagnostic arrays exist')
        r.save(root/'arrays.npz',data);guard(start,root)
        if verify()!=packet:raise ValueError('diagnostic packet changed during execution')
        verify_archive()
        result={'protocol_id':PROTOCOL,'state_slice':PROTOCOL,'packet_sha256':packet,'claim_ceiling':CLAIM,
            'assurance_level':'INTERNAL_PASS','reviewer_status':'NOT_SUBMITTED','parent_packet_sha256':r.verify_source(),
            'parent_validation_sha256':r.sha(PARENT/'closeout/validation.json'),'metrics':summary,
            'selection_sha256':r.sha(root/'selection.json'),'arrays_sha256':c.array_digest(data),'arrays_file_sha256':r.sha(root/'arrays.npz'),
            'runtime':runtime,'parent_runtime_sha256':r.sha(PARENT/'primary/runtime.json'),
            'offline_environment':{'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
            'resources':guard(start,root),'new_training_steps':0,'new_heldout_queries':0,
            'row_level_heldout_files_read':False,'data_partition':'V39_ADAPTATION_ONLY',
            'controls':{'actual_noop_reruns':sum(len(v) for k,v in data.items() if k.endswith('/noop_errors')),'all_noop_errors_exact_zero':True,
                'analytic_zero_delta_relative_error_on_resolved_reference':1.,'full_jacobians_checked_against_four_jvps_each':True}}
        write(root/'result.json',result)
        if role=='replica':
            original=r.read(BASE/'primary/result.json')
            keys=('protocol_id','packet_sha256','parent_packet_sha256','parent_validation_sha256','metrics','selection_sha256','arrays_sha256','controls')
            checks={key:original[key]==result[key] for key in keys}
            if original['runtime']['prefix']==runtime['prefix'] or original['runtime']['site_packages']==runtime['site_packages']:raise ValueError('replica path not distinct')
            if {k:v for k,v in original['runtime'].items() if k not in ('prefix','site_packages')}!={k:v for k,v in runtime.items() if k not in ('prefix','site_packages')}:raise ValueError('replica toolchain differs')
            if not all(checks.values()):raise ValueError('diagnostic reproduction differs')
            for role2 in ('primary','replica'):
                receipt=r.read(BASE/role2/'result.json')
                if r.sha(BASE/role2/'arrays.npz')!=receipt['arrays_file_sha256'] or r.sha(BASE/role2/'selection.json')!=receipt['selection_sha256']:raise ValueError('diagnostic artifacts changed')
            write(root/'reconciliation.json',{'checks':checks,'assurance_level':'REPRODUCED_LOCAL','same_owner_and_host':True,
                  'primary_result_sha256':r.sha(BASE/'primary/result.json'),'replica_result_sha256':r.sha(root/'result.json')})
        return {'outcome_status':summary['outcome_status'],'assurance_level':'REPRODUCED_LOCAL' if role=='replica' else 'INTERNAL_PASS',
                'result_sha256':r.sha(root/'result.json'),'local_cells':len(summary['cells']),'transport_cells':len(summary['transport'])}
    except BaseException as error:
        try:write(root/'error.json',{'protocol_id':PROTOCOL,'outcome_status':'INCONCLUSIVE','exception_type':type(error).__name__,'retry_same_root_permitted':False})
        except (FileExistsError,FileNotFoundError):pass
        raise


def closeout():
    import theseus_first_mlp_diagnosis_v1_closeout as numeric
    verify();verify_archive();checks={}
    for role in ('primary','replica'):
        root=BASE/role;result=r.read(root/'result.json');index=r.read(root/'selection.json');data=r.arrays(root/'arrays.npz')
        if r.sha(root/'arrays.npz')!=result['arrays_file_sha256'] or c.array_digest(data)!=result['arrays_sha256'] or r.sha(root/'selection.json')!=result['selection_sha256']:raise ValueError('closeout artifact binding')
        numeric.compare(numeric.recompute(data,index),result['metrics'])
        if len(index['cases'])!=3 or any(value['selected_prompt_indices']!=selection(case) for case,value in index['cases'].items()):raise ValueError('selection differs')
        if len(result['metrics']['cells'])!=18 or len(result['metrics']['transport'])!=6:raise ValueError('diagnostic cohort incomplete')
        expected_keys=set();width=index['native_width'];vocab=index['vocab_size']
        if width!=768:raise ValueError('guide width changed')
        for case in index['cases']:
            for method in c.METHODS:
                for label in POSITIONS:
                    prefix=case+'/'+method+'/'+label+'/'
                    for key,shape in (('teacher_value',(PROMPTS,width)),('student_value',(PROMPTS,width)),
                        ('teacher_jacobian',(PROMPTS,width,width)),('student_jacobian',(PROMPTS,width,width)),
                        ('teacher_delta',(PROMPTS*8,width)),('student_delta',(PROMPTS*8,width))):
                        expected_keys.add(prefix+key)
                        if data[prefix+key].shape!=shape:raise ValueError('local array shape incomplete')
                prefix=case+'/'+method+'/transport/'
                for key in ('actual','guide','student_guide','anchored','local_teacher_delta','local_student_delta'):
                    expected_keys.add(prefix+key)
                    if data[prefix+key].shape!=(PROMPTS*8,width if key.startswith('local_') else vocab):raise ValueError('transport array shape incomplete')
                expected_keys.add(case+'/'+method+'/noop_errors')
        if set(data)!=expected_keys:raise ValueError('diagnostic array inventory differs')
        for key,values in data.items():
            if not np.isfinite(values).all():raise ValueError('nonfinite persisted array')
            if key.endswith('/noop_errors') and (values.shape!=(PROMPTS,) or np.any(values!=0)):raise ValueError('no-op rows')
        checks[role]={'local_cells':18,'transport_cells':6,'full_jacobians':288,'finite_local_pairs':1152,'decoder_pairs':384,'arrays_sha256':result['arrays_sha256']}
    rec=r.read(BASE/'replica/reconciliation.json')
    if len(rec['checks'])!=8 or not all(v is True for v in rec['checks'].values()):raise ValueError('reconciliation incomplete')
    originals=[r.read(BASE/role/'result.json') for role in ('primary','replica')]
    for key in rec['checks']:
        if originals[0][key]!=originals[1][key]:raise ValueError('reconciliation values differ')
    runtimes=[x['runtime'] for x in originals]
    if runtimes[0]['prefix']==runtimes[1]['prefix'] or runtimes[0]['site_packages']==runtimes[1]['site_packages']:raise ValueError('runtime paths not segregated')
    if {k:v for k,v in runtimes[0].items() if k not in ('prefix','site_packages')}!={k:v for k,v in runtimes[1].items() if k not in ('prefix','site_packages')}:raise ValueError('runtime toolchains differ')
    for role in ('primary','replica'):
        if rec[role+'_result_sha256']!=r.sha(BASE/role/'result.json'):raise ValueError('result reconciliation binding')
    answer={'protocol_id':PROTOCOL,'packet_sha256':verify(),'state_slice':PROTOCOL,'assurance_level':'REPRODUCED_LOCAL',
        'reviewer_status':'NOT_SUBMITTED','claim_ceiling':CLAIM,'audits':checks,
        'outcome_status':r.read(BASE/'primary/result.json')['metrics']['outcome_status'],'same_owner_and_host':True,
        'reconciliation_sha256':r.sha(BASE/'replica/reconciliation.json')}
    write(BASE/'closeout/validation.json',answer)
    return answer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('verify','run','closeout'))
    parser.add_argument('--role',choices=('primary','replica'),default='primary');args=parser.parse_args()
    if args.action=='verify':value={'packet_sha256':verify()}
    elif args.action=='run':value=run(args.role)
    else:value=closeout()
    print(json.dumps(value,sort_keys=True))
