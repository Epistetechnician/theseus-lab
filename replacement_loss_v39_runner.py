"""Explicit all-token fitting adapter; native V36 evaluation is unchanged.

State slices: theseus-lab-sandbox-v1, theseus-reused-gpt2-all-token-ladder-v39,
global-experiment-assurance-lifecycle-v1. Offline, reused guide, external data.
"""
import argparse
import copy
import json
import shutil

import numpy as np
import torch
import replacement_loss_v38_runner as previous

c,r=previous.c,previous.r
f=r.f
PARENT=r.BASE
DIAGNOSTIC=r.ARTIFACTS/'theseus-lab-sandbox-v1-v38-token-coverage-audit-v1'
OLD_MEMBERS=r.MEMBERS
c.PROTOCOL=c.SLICE='theseus-reused-gpt2-all-token-ladder-v39'
c.CLAIM='LOCAL_REUSED_GPT2_EARLY_ALL_TOKEN_REPLACEMENT_DEVELOPMENT_ONLY'
r.BASE=r.ARTIFACTS/'theseus-lab-sandbox-v1-reused-gpt2-all-token-ladder-v39'
r.FREEZE=r.REPO/'docs/replacement-loss-v39-freeze.json'
r.MEMBERS=(*OLD_MEMBERS,'docs/replacement-loss-v38-freeze.json','docs/replacement-loss-v38-results.json',
    'theseus_token_coverage_audit_v1.py','tests/test_token_coverage_audit_v1.py',
    'docs/v38-token-coverage-audit-v1.md','docs/v38-token-coverage-audit-v1-freeze.json',
    'docs/v38-token-coverage-audit-v1-results.json','replacement_loss_v39_runner.py',
    'tests/test_replacement_loss_v39.py','docs/replacement-loss-v39-all-token-ladder.md')
SITES=previous.SITES
LIMIT=96
ORIGINAL_RECORDS,ORIGINAL_ENCODE=f.records,f.encode
ORIGINAL_FORWARD,ORIGINAL_FIT,ORIGINAL_SAVE=c.forward,c.fit,r.save


class FitTokenIDs(list):
    """Live adaptation marker; JSON retains the exact original token list."""
    def __init__(self, ids, partition):
        if partition not in ('adapt','donor'):raise ValueError('fitting partition')
        super().__init__(ids);self.partition=partition


def records(family,seed,partition,count,entropy='preparation'):
    rows=ORIGINAL_RECORDS(family,seed,partition,count,entropy)
    if partition in ('adapt','donor'):
        for row in rows:row['fit_partition']=partition
    return rows


def encode(tokenizer,rows,seen_text,seen_tokens,limit=LIMIT):
    encoded=ORIGINAL_ENCODE(tokenizer,rows,seen_text,seen_tokens,limit)
    for source,target in zip(rows,encoded):
        if 'fit_partition' in source:
            target['ids']=FitTokenIDs(target['ids'],source['fit_partition'])
            target['fit_partition']=source['fit_partition']
    return encoded


def padded(x):
    """Mask is capture metadata and never a model feature."""
    if x.ndim!=2 or not 2<=len(x)<=LIMIT or not torch.isfinite(x).all():
        raise ValueError('all-token capture shape/value')
    value=torch.zeros((LIMIT,x.shape[1]+1),dtype=x.dtype,device=x.device)
    value[:len(x),:-1]=x;value[:len(x),-1]=1
    return value


def forward(model,family,ids,sites,intervention=None,capture_all=False):
    if not isinstance(ids,FitTokenIDs):
        return ORIGINAL_FORWARD(model,family,ids,sites,intervention,capture_all)
    if intervention is not None:raise ValueError('fitting marker in held-out intervention')
    value=ORIGINAL_FORWARD(model,family,ids,sites,capture_all=True)
    for site in sites:value['sites'][site]['x']=padded(value['sites'][site]['x'])
    return value


def unpack(value):
    value=np.asarray(value)
    if value.ndim!=3 or value.shape[1]!=LIMIT or value.shape[2]<2 or value.dtype!=np.float32 or not np.isfinite(value).all():
        raise ValueError('masked fitting capture contract')
    mask=value[:,:,-1]
    if not np.isin(mask,(0,1)).all():raise ValueError('capture mask is not binary')
    lengths=mask.sum(1).astype(np.int64)
    if np.any(lengths<2) or not np.array_equal(mask,np.arange(LIMIT)[None,:]<lengths[:,None]):
        raise ValueError('capture mask is not a valid contiguous prefix')
    if np.any(value[:,:,:-1][mask==0]!=0):raise ValueError('nonzero capture padding')
    return [value[i,:n,:-1].copy() for i,n in enumerate(lengths)]


def pool(x,donor):
    """Every adaptation token once; donor mapping preserves both endpoints."""
    if np.shape(x)!=np.shape(donor):raise ValueError('unpaired capture shapes')
    a,b=unpack(x),unpack(donor)
    if not a:raise ValueError('empty fitting pool')
    lengths=np.asarray([(len(v),len(w)) for v,w in zip(a,b)],dtype=np.int64)
    paired=[w[np.arange(len(v))*(len(w)-1)//(len(v)-1)] for v,w in zip(a,b)]
    return {'x':np.concatenate(a),'donor':np.concatenate(paired),'donor_source':np.concatenate(b),
            'lengths':lengths,'pool_schema':np.asarray([1],dtype=np.int64)}


def fit(teacher,x,donor,family,seed,method,budget=lambda:None,updates=c.UPDATES):
    arrays=pool(x.detach().cpu().numpy(),donor.detach().cpu().numpy())
    return ORIGINAL_FIT(teacher,torch.from_numpy(arrays['x']),torch.from_numpy(arrays['donor']),
                        family,seed,method,budget,updates)


def save(path,value):
    if path.name.endswith('-adaptation-inputs.npz'):
        if set(value)!={'x','donor'}:raise ValueError('unexpected adaptation capture fields')
        value=pool(value['x'],value['donor'])
    return ORIGINAL_SAVE(path,value)


def parents():
    report=r.read(r.REPO/'docs/replacement-loss-v38-results.json')
    if report['scientific_outcome']!='FAIL' or report['assurance_level']!='REPRODUCED_LOCAL':
        raise ValueError('V38 terminal failure not preserved')
    for field,path in (('primary_result_sha256',PARENT/'primary/result.json'),
                       ('replica_result_sha256',PARENT/'replica/result.json'),
                       ('reconciliation_sha256',PARENT/'replica/reconciliation.json'),
                       ('validation_sha256',PARENT/'closeout/validation.json')):
        if r.sha(path)!=report[field]:raise ValueError('V38 terminal bytes differ')
    diagnostic=r.read(r.REPO/'docs/v38-token-coverage-audit-v1-results.json')
    if diagnostic['outcome_status']!='COVERAGE_GAP_SUPPORTED' or diagnostic['assurance_level']!='REPRODUCED_LOCAL':
        raise ValueError('coverage hypothesis not reproduced')
    for role in ('primary','replica'):
        path=DIAGNOSTIC/role/'result.json';value=r.read(path)
        if r.sha(path)!=diagnostic[role+'_result_sha256'] or r.sha(DIAGNOSTIC/role/'statistics.npz')!=value['statistics_file_sha256']:
            raise ValueError('coverage diagnostic changed')
    reconciliation=r.read(DIAGNOSTIC/'replica/reconciliation.json')
    if r.sha(DIAGNOSTIC/'replica/reconciliation.json')!=diagnostic['reconciliation_sha256'] or len(reconciliation['checks'])!=7 or not all(v is True for v in reconciliation['checks'].values()):
        raise ValueError('coverage reconciliation differs')


def intake():
    r.mount_guard();packet=r.verify_source();parents()
    if r.BASE.exists():raise FileExistsError('V39 root already exists')
    old=r.read(PARENT/'intake.json');sets={k:set(v) for k,v in old['exclusion'].items()}
    files={row['path']:copy.deepcopy(row) for row in old['historical_files']}
    # Explicit bounded predecessor roots. Read old fixture identities only for
    # exclusion; they cannot become new evaluation or training examples.
    paths=[p for p in PARENT.rglob('*.json') if 'frozen-packet' not in p.parts]
    paths.extend(DIAGNOSTIC.rglob('*.json'))
    for path in sorted(paths):
        if path.stat().st_size>64*1024**2:raise ValueError('history JSON size cap')
        r.collect_history(r.read(path),sets)
        key=str(path.relative_to(r.ARTIFACTS))
        files[key]={'path':key,'sha256':r.sha(path),'kind':'JSON_IDENTITY_RECORD'}
    exclusion={k:sorted(v) for k,v in sets.items()}
    r.write(r.BASE/'intake.json',{'protocol_id':c.PROTOCOL,'packet_sha256':packet,
        'parent_packet_sha256':r.verify_parent(),'historical_files':list(files.values()),
        'quarantined_legacy_exports':old['quarantined_legacy_exports'],'exclusion':exclusion,
        'exclusion_sha256':c.digest(exclusion),'claim_ceiling':c.CLAIM,
        'guide_reuse_declared_before_freeze':True,'sites_predeclared':list(SITES),
        'coverage_diagnostic_reconciliation_sha256':r.sha(DIAGNOSTIC/'replica/reconciliation.json')})
    for member in (*r.MEMBERS,str(r.FREEZE.relative_to(r.REPO))):
        target=r.BASE/'frozen-packet'/member;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r.REPO/member,target)
    old_entry=r.read(PARENT/'checkpoints.json')['checkpoints'][0]
    source=r.contained(PARENT,old_entry['local_path']);destination=r.BASE/'checkpoint';destination.mkdir()
    for row in old_entry['files']:
        path=r.contained(source,row['path'])
        if path.stat().st_size!=row['bytes'] or r.sha(path)!=row['sha256']:raise ValueError('approved reused guide changed')
        target=destination/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    entry={k:copy.deepcopy(old_entry[k]) for k in ('family','model_id','revision','files')}
    entry.update({'local_path':'checkpoint','reused_development_checkpoint':True,'sites_predeclared':list(SITES)})
    r.write(r.BASE/'checkpoints.json',{'packet_sha256':packet,'intake_sha256':r.sha(r.BASE/'intake.json'),
        'approved_parent_manifest_sha256':r.sha(PARENT/'checkpoints.json'),'checkpoints':[entry]})
    return {'packet_sha256':packet,'intake_sha256':r.sha(r.BASE/'intake.json'),
            'checkpoint_manifest_sha256':r.sha(r.BASE/'checkpoints.json')}


def verify_intake():
    value=previous.ORIGINAL_INTAKE_VERIFY();parents()
    return value


def manifest(path):
    verify_intake();path=r.contained(r.BASE,str(path.relative_to(r.BASE)))
    value=r.read(path);old=r.read(PARENT/'checkpoints.json')['checkpoints'][0]
    if value['packet_sha256']!=r.verify_source() or value['intake_sha256']!=r.sha(r.BASE/'intake.json') or value['approved_parent_manifest_sha256']!=r.sha(PARENT/'checkpoints.json'):
        raise ValueError('reused guide/source binding')
    if len(value['checkpoints'])!=1:raise ValueError('one reused GPT-2 checkpoint required')
    entry=value['checkpoints'][0]
    if entry['reused_development_checkpoint'] is not True or entry['sites_predeclared']!=list(SITES):raise ValueError('reuse/site declaration')
    if any(entry[k]!=old[k] for k in ('family','model_id','revision','files')):raise ValueError('reused guide differs')
    root=r.contained(r.BASE,entry['local_path']);actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!={row['path'] for row in entry['files']} or len(actual)!=len(entry['files']):raise ValueError('asset list incomplete')
    for row in entry['files']:
        p=r.contained(root,row['path'])
        if p.stat().st_size!=row['bytes'] or r.sha(p)!=row['sha256']:raise ValueError('reused asset changed')
    return value


def capture_audit(root):
    result=r.read(root/'result.json');cells=0;tokens={}
    for family,binding in result['preparation']['families'].items():
        for case,details in binding['cases'].items():
            encoded=r.read(root/case/'adaptation.json')['encoded']
            expected=np.asarray([(len(a['ids']),len(b['ids'])) for a,b in zip(encoded['adapt'],encoded['donor'])],dtype=np.int64)
            if expected.shape!=(128,2) or any(row['fit_partition']!=key for key,rows in encoded.items() for row in rows):
                raise ValueError('source adaptation roles/count differ')
            for method,stages in details['arms'].items():
                for stage,target in stages.items():
                    data=r.arrays(root/case/(method+'-'+stage+'-adaptation-inputs.npz'))
                    if set(data)!={'x','donor','donor_source','lengths','pool_schema'} or not np.array_equal(data['pool_schema'],[1]) or not np.array_equal(data['lengths'],expected):
                        raise ValueError('pool schema/source token lengths differ')
                    width=768;na,nd=expected.sum(0)
                    if data['x'].shape!=(na,width) or data['donor'].shape!=(na,width) or data['donor_source'].shape!=(nd,width):
                        raise ValueError('all-token coverage shape differs')
                    if any(data[k].dtype!=np.float32 or not np.isfinite(data[k]).all() for k in ('x','donor','donor_source')):
                        raise ValueError('all-token capture values differ')
                    ao=do=0
                    for la,ld in expected:
                        indices=np.arange(la)*(ld-1)//(la-1)
                        if not np.array_equal(data['donor'][ao:ao+la],data['donor_source'][do+indices]):
                            raise ValueError('donor normalized-position pairing differs')
                        ao+=la;do+=ld
                    if c.array_digest(data)!=target['capture_sha256']:raise ValueError('pooled capture digest differs')
                    tokens[case+'/'+method+'/'+stage]={'adapt_tokens':int(na),'donor_source_tokens':int(nd)};cells+=1
    return {'capture_cells':cells,'tokens':tokens,'all_adaptation_tokens_once':True,
            'native_feature_width_without_mask':768,'endpoint_preserving_donor_pairing':True}


def closeout():
    import replacement_loss_v36_closeout as audit
    result=audit.validate()
    result['all_token_capture_audits']={role:capture_audit(r.BASE/role) for role in ('primary','replica')}
    if result['all_token_capture_audits']['primary']!=result['all_token_capture_audits']['replica']:
        raise ValueError('pooled capture reproduction differs')
    r.write(r.BASE/'closeout/validation.json',result)
    return result


f.records=records;f.encode=encode;c.forward=forward;c.fit=fit;r.save=save
r.intake=intake;r.verify_intake=verify_intake;r.checkpoint_manifest=manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('verify','intake','run','closeout'))
    parser.add_argument('--role',choices=('primary','replica'),default='primary');args=parser.parse_args()
    if args.action=='verify':answer={'packet_sha256':r.verify_source()}
    elif args.action=='intake':answer=intake()
    elif args.action=='run':answer=r.run(r.BASE/'checkpoints.json',args.role)
    else:answer=closeout()
    print(json.dumps(answer,sort_keys=True))
