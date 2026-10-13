"""Separate persisted-array scoring and custody audit for V36.

State slices: theseus-lab-sandbox-v1, theseus-pretrained-whole-block-transfer-v36,
global-experiment-assurance-lifecycle-v1. No authority is conferred by this audit.
"""
import json
import numpy as np

import replacement_loss_v36_core as c
import replacement_loss_v36_runner as r


def relative(prediction,reference):
    resolved=np.linalg.norm(reference,axis=1)>=1e-4
    if not resolved.any():
        return {'resolved_fraction':0.,'median_error':None,'p90_error':None,'median_cosine':None,'passed':False}
    p,g=prediction[resolved].astype('float64'),reference[resolved].astype('float64')
    errors=np.linalg.norm(p-g,axis=1)/np.maximum(np.linalg.norm(g,axis=1),1e-6)
    angles=np.sum(p*g,axis=1)/np.maximum(np.linalg.norm(p,axis=1)*np.linalg.norm(g,axis=1),1e-12)
    out={'resolved_fraction':float(resolved.mean()),'median_error':float(np.median(errors)),
         'p90_error':float(np.quantile(errors,.9)),'median_cosine':float(np.median(angles))}
    out['passed']=out['resolved_fraction']>=.9 and out['median_error']<=.2 and out['p90_error']<=.5 and out['median_cosine']>=.9
    return out


def rmse(actual,reference):
    return float(np.sqrt(np.mean((actual.astype('float64')-reference.astype('float64'))**2)/max(float(np.mean(reference.astype('float64')**2)),1e-8)))


def cell(prediction,oracle,rescues,active,stratum):
    prefix=stratum+'/';p,g=prediction[prefix+'logits'],oracle[prefix+'logits']
    out={'output_nrmse':rmse(p,g),'top1_agreement':float(np.mean(p.argmax(1)==g.argmax(1))),
         'nll_increase':float(np.mean(prediction[prefix+'nll']-oracle[prefix+'nll'])),
         'joint_effect':relative(prediction[prefix+'joint_effect'],oracle[prefix+'joint_effect']),
         'effects':{},'activation':{},'rescue':{},'rescue_prediction':{}}
    for site in active:
        key=str(site)
        out['effects'][key]=relative(prediction[prefix+'effect/'+key],oracle[prefix+'effect/'+key])
        out['activation'][key]=rmse(prediction[prefix+'activation/'+key],oracle[prefix+'activation/'+key])
        out['rescue'][key]=rmse(rescues[prefix+key],g)
        out['rescue_prediction'][key]=rmse(p,rescues[prefix+key])
    out['passed']=bool(out['output_nrmse']<=.1 and out['top1_agreement']>=.95 and out['nll_increase']<=.03
                      and out['joint_effect']['passed'] and all(e['passed'] for e in out['effects'].values())
                      and max(out['activation'].values())<=.15 and max(out['rescue'].values())<=.1
                      and max(out['rescue_prediction'].values())<=.1)
    return out


def verify_run(root):
    result=r.read(root/'result.json');cells=0;readouts=0
    for item in result['artifacts']:
        p=r.contained(root,item['path'])
        if p.stat().st_size!=item['bytes'] or r.sha(p)!=item['sha256']:
            raise ValueError('artifact hash/size failed')
    if result['challenge_generated']:
        preparation=r.read(root/'candidate-freeze.json')
        challenge=r.read(root/'challenge.json');seal=r.read(root/'prediction-freeze.json')
        if preparation!=result['preparation'] or challenge['candidate_freeze_sha256']!=c.digest(preparation):
            raise ValueError('candidate/challenge seal failed')
        if seal['challenge_sha256']!=r.sha(root/'challenge.json'):
            raise ValueError('challenge/prediction seal failed')
    if set(result['preparation']['families'])!=set(c.FAMILIES):
        raise ValueError('architecture cohort incomplete')
    for family,prep in result['preparation']['families'].items():
        if result['challenge_generated'] and set(prep['cases'])!={family+'-'+str(seed) for seed in c.SEEDS}:
            raise ValueError('seed cohort incomplete')
        for case,binding in prep['cases'].items():
            oracle=r.arrays(root/case/'oracle.npz')
            if c.array_digest(oracle)!=result['outcome_bindings'][case]['oracle_sha256']:
                raise ValueError('oracle binding failed')
            noops=r.arrays(root/case/'noop.npz');noop_rescues=r.arrays(root/case/'noop-rescues.npz')
            if c.array_digest(noops)!=result['outcome_bindings'][case]['noop_sha256'] or c.array_digest(noop_rescues)!=result['outcome_bindings'][case]['noop_rescues_sha256']:
                raise ValueError('no-op binding failed')
            sites=prep['qualification']['sites']
            positive={s:cell(noops,oracle,noop_rescues,sites,s) for s in c.STRATA}
            if set(binding['arms'])!=set(c.METHODS):
                raise ValueError('comparison arms incomplete')
            for method,stages in binding['arms'].items():
                if set(stages)!={'block_1','block_2'}:
                    raise ValueError('ladder stages incomplete')
                if set(seal['bindings'][case][method])!=set(stages)|{'random','wrong','zero'}:
                    raise ValueError('prediction seal entries differ')
                for label,h in seal['bindings'][case][method].items():
                    prediction=r.arrays(root/case/(method+'-'+label+'-predictions.npz'))
                    if c.array_digest(prediction)!=h:
                        raise ValueError('prediction binding failed')
                for stage,target in stages.items():
                    if target['active_sites']!=sites[:int(stage[-1])]:
                        raise ValueError('stage sites differ')
                    stem=root/case/(method+'-'+stage)
                    if c.array_digest(r.arrays(stem.with_name(stem.name+'-weights.npz')))!=target['weights_sha256']:
                        raise ValueError('candidate binding failed')
                    if c.array_digest(r.arrays(stem.with_name(stem.name+'-adaptation-inputs.npz')))!=target['capture_sha256']:
                        raise ValueError('capture binding failed')
                    trace=r.arrays(stem.with_name(stem.name+'-trace.npz'))
                    if c.array_digest(trace)!=target['trace_sha256']:
                        raise ValueError('trace binding failed')
                    t=trace['readouts'];expected=np.minimum(1.,.5*t[:,0]/np.maximum(t[:,1],1e-12)) if method=='jvp' else np.ones(len(t))
                    if not np.isfinite(t).all() or not np.array_equal(t[:,2],expected) or np.any(t[:,1]*t[:,2]>.5*t[:,0]+1e-8):
                        raise ValueError('gradient cap failed')
                    readouts+=len(t)
                    prediction=r.arrays(stem.with_name(stem.name+'-predictions.npz'));rescues=r.arrays(stem.with_name(stem.name+'-rescues.npz'))
                    if c.array_digest(rescues)!=result['outcome_bindings'][case]['rescues'][method+'/'+stage]:
                        raise ValueError('rescue binding failed')
                    reference=dict(oracle);active=target['active_sites']
                    if len(active)==1:
                        for stratum in c.STRATA:reference[stratum+'/joint_effect']=oracle[stratum+'/effect/'+str(active[0])]
                    for stratum in c.STRATA:
                        if cell(prediction,reference,rescues,active,stratum)!=result['metrics'][case][method]['cells'][stage][stratum]:
                            raise ValueError('recomputed numeric cell differs')
                        cells+=1
                sensitivity={}
                for stratum in c.STRATA:
                    treatment=result['metrics'][case][method]['cells']['block_2'][stratum]['joint_effect']['median_error']
                    errors=[relative(r.arrays(root/case/(method+'-'+label+'-predictions.npz'))[stratum+'/joint_effect'],oracle[stratum+'/joint_effect'])['median_error'] for label in ('zero','wrong','random')]
                    sensitivity[stratum]=treatment is not None and all(e is not None and e>=treatment+.1 for e in errors)
                reported=result['metrics'][case][method]
                passed=all(v['passed'] for ss in reported['cells'].values() for v in ss.values()) and all(sensitivity.values()) and all(v['passed'] for v in positive.values())
                if positive!=reported['positive_controls'] or sensitivity!=reported['control_sensitivity'] or passed!=reported['passed']:
                    raise ValueError('control or method verdict differs')
    qualified=all(p['qualification']['passed'] for p in result['preparation']['families'].values())
    if qualified:
        if not result['challenge_generated'] or not result['metrics']:
            raise ValueError('qualified targets have no assessment')
        expected='PASS' if all(m['jvp']['passed'] for m in result['metrics'].values()) else 'FAIL'
    else:
        expected='NO_QUALIFIED_TARGET'
        if result['challenge_generated'] or result['metrics']:
            raise ValueError('unqualified target was assessed')
    if result['outcome_status']!=expected:
        raise ValueError('aggregate verdict differs')
    return {'manifested_files':len(result['artifacts']),'native_cells':cells,'gradient_readouts':readouts,
            'final_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}


def validate():
    r.verify_intake();audits={role:verify_run(r.BASE/role) for role in ('primary','replica')}
    primary,replica=[r.read(r.BASE/role/'result.json') for role in ('primary','replica')]
    receipt=r.read(r.BASE/'replica/reconciliation.json')
    for role in ('primary','replica'):
        if receipt[role+'_result_sha256']!=r.sha(r.BASE/role/'result.json'):
            raise ValueError('reconciliation result digest differs')
    if set(receipt['checks'])!=set(r.RECONCILIATION_KEYS):
        raise ValueError('reconciliation keys incomplete')
    runtimes=[r.read(r.BASE/role/'runtime.json') for role in ('primary','replica')]
    if runtimes[0]['prefix']==runtimes[1]['prefix'] or runtimes[0]['site_packages']==runtimes[1]['site_packages']:
        raise ValueError('reproduction runtime not distinct')
    if {k:v for k,v in runtimes[0].items() if k not in ('prefix','site_packages')}!={k:v for k,v in runtimes[1].items() if k not in ('prefix','site_packages')}:
        raise ValueError('reproduction runtime toolchains differ')
    for key,passed in receipt['checks'].items():
        if passed is not True or primary[key]!=replica[key]:
            raise ValueError('reconciliation is false')
    return {'protocol_id':c.PROTOCOL,'state_slice':c.SLICE,'packet_sha256':r.verify_source(),
            'scientific_outcome':primary['outcome_status'],'assurance_level':'REPRODUCED_LOCAL',
            'reviewer_status':'NOT_SUBMITTED','same_owner_and_host':True,'audits':audits,
            'claim_ceiling':c.CLAIM,'reconciliation_sha256':r.sha(r.BASE/'replica/reconciliation.json')}


if __name__=='__main__':
    result=validate();path=r.BASE/'closeout/validation.json'
    if path.exists():
        if r.read(path)!=result:raise ValueError('existing closeout differs')
    else:r.write(path,result)
    print(json.dumps({'scientific_outcome':result['scientific_outcome'],'assurance_level':result['assurance_level'],
                      'validation_sha256':r.sha(path),'audits':result['audits']},sort_keys=True))
