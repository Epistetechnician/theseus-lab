"""Prospective weight-held-out development transfer; family gates are separate.

State slices: theseus-lab-sandbox-v1, theseus-pretrained-development-transfer-v37,
theseus-public-checkpoint-acquisition-v37,
global-experiment-assurance-lifecycle-v1. Public static assets opt-in; local CPU.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import re
import shutil
import time

import torch
import replacement_loss_v36r1_runner as adapter
import theseus_checkpoint_exposure_v1 as exposure

c,r=adapter.c,adapter.r
FAMILIES=tuple(c.FAMILIES)
c.PROTOCOL=c.SLICE='theseus-pretrained-development-transfer-v37'
c.CLAIM='LOCAL_METADATA_KNOWN_WEIGHT_HELD_OUT_PRETRAINED_TWO_MLP_INTERFACE_ONLY'
BASE=r.ARTIFACTS/'theseus-lab-sandbox-v1-pretrained-development-transfer-v37'
r.BASE=BASE
r.FREEZE=r.REPO/'docs/replacement-loss-v37-freeze.json'
r.MEMBERS=(*r.MEMBERS,'docs/replacement-loss-v36r1-freeze.json',
           'docs/replacement-loss-v36r2-freeze.json','docs/replacement-loss-v36r2-results.json',
           *exposure.MEMBERS[:3], 'docs/checkpoint-exposure-v1-freeze.json',
           'docs/checkpoint-exposure-v1-results.json', 'replacement_loss_v37_runner.py',
           'tests/test_replacement_loss_v37.py','docs/replacement-loss-v37-development-transfer.md')
adapter.RUNTIMES=r.RUNTIME_VOLUME/'ResearchRuntimes/theseus-lab-v37/toolchains'
CATALOG=r.ARTIFACTS/'theseus-lab-sandbox-v1-replacement-loss-v27/v27-hub-screen.json'
# Per-family hard scope; independent jobs permit retaining each honest outcome.
r.MAX_SECONDS=8*3600


def audit_receipt():
    binding=r.read(r.REPO/'docs/checkpoint-exposure-v1-results.json')
    if exposure.sha(exposure.BASE/'result.json')!=binding['result_sha256']:
        raise ValueError('exposure audit receipt changed')
    return exposure.read(exposure.BASE/'result.json')


def metadata_eligible(identity,audit):
    return audit['identities'].get(identity.lower(),{}).get('status')=='METADATA_ONLY_IN_RETAINED_CUSTODY'


def catalog_candidates(catalog,family,audit,limit=200):
    """Fixed retained metadata, popularity then identity; never guide outputs."""
    pool={}
    for query in catalog['family_screens'][family]['queries']:
        for row in query['result_projection']:
            identity=row['model_id']
            if row.get('model_type')!=adapter.FAMILY_TYPE[family] or not metadata_eligible(identity,audit):continue
            if adapter.REJECT_NAMES.search(identity) or row.get('private') or row.get('gated'):continue
            old=pool.get(identity.lower())
            if old is None or row.get('downloads',0)>old.get('downloads',0):pool[identity.lower()]=row
    return sorted(pool.values(),key=lambda row:(-row.get('downloads',0),row['model_id'].lower()))[:limit]


def intake():
    r.mount_guard();packet=r.verify_source();parent=r.verify_parent();audit=audit_receipt()
    if BASE.exists():raise FileExistsError('V37 intake already exists')
    sets={k:set() for k in ('model_ids','text_hashes','token_hashes','prior_digests')};files=[];opaque=[]
    for folder in sorted(r.ARTIFACTS.glob('theseus*')):
        if not folder.is_dir() or folder==BASE:continue
        for path in sorted(folder.rglob('*.json')):
            relative=str(path.relative_to(r.ARTIFACTS));resolved=path.resolve()
            if not resolved.is_relative_to(r.ARTIFACTS.resolve()):raise ValueError('intake custody escape')
            value,row=adapter.classify(path,relative);files.append(row)
            if value is None and row['kind']=='QUARANTINED_OPAQUE_LEGACY_EXPORT':opaque.append(row)
            else:r.collect_history(value,sets)
    registry=r.read(r.REPO/'checkpoint_manifests.json');r.collect_history(registry,sets)
    sets['model_ids'].update(k.lower() for k in registry)
    # A new, explicit scope: metadata exposure is allowed; unknowns stay closed.
    allowed={k for k,v in audit['identities'].items() if v['status']=='METADATA_ONLY_IN_RETAINED_CUSTODY'}
    sets['model_ids'].difference_update(allowed)
    sets['prior_digests'].update(audit['weight_digests'])
    exclusion={k:sorted(v) for k,v in sets.items()}
    r.write(BASE/'intake.json',{'protocol_id':c.PROTOCOL,'packet_sha256':packet,
        'parent_packet_sha256':parent,'historical_files':files,'quarantined_legacy_exports':opaque,
        'exclusion':exclusion,'exclusion_sha256':c.digest(exclusion),
        'exposure_audit_sha256':exposure.sha(exposure.BASE/'result.json'),
        'retained_catalog_sha256':r.sha(CATALOG),'metadata_known_scope':True,
        'checkpoint_selection_status':'NOT_SELECTED','claim_ceiling':c.CLAIM})
    for member in (*r.MEMBERS,str(r.FREEZE.relative_to(r.REPO))):
        target=BASE/'frozen-packet'/member;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(r.REPO/member,target)
    return {'packet_sha256':packet,'intake_sha256':r.sha(BASE/'intake.json'),
            'historical_files':len(files),'metadata_only_allowed':len(allowed),
            'unknown_or_exposed_exclusions':len(exclusion['model_ids'])}


def verify_intake():
    r.mount_guard();receipt=r.read(BASE/'intake.json')
    if receipt['packet_sha256']!=r.verify_source() or receipt['exclusion_sha256']!=c.digest(receipt['exclusion']):
        raise ValueError('V37 intake binding')
    if receipt['parent_packet_sha256']!=r.verify_parent():raise ValueError('V35 parent differs')
    audit_receipt()
    if receipt['exposure_audit_sha256']!=exposure.sha(exposure.BASE/'result.json') or receipt['retained_catalog_sha256']!=r.sha(CATALOG):
        raise ValueError('audit/catalog binding')
    for member in receipt['historical_files']:
        if r.sha(r.contained(r.ARTIFACTS,member['path']))!=member['sha256']:raise ValueError('historical input differs')
    for member in r.freeze_material():
        if r.sha(BASE/'frozen-packet'/member['path'])!=member['sha256']:raise ValueError('source archive differs')
    if r.sha(BASE/'frozen-packet'/r.FREEZE.relative_to(r.REPO))!=r.sha(r.FREEZE):raise ValueError('freeze archive differs')
    if r.BASE!=BASE and r.sha(r.BASE/'intake.json')!=r.sha(BASE/'intake.json'):raise ValueError('family intake differs')
    return receipt


def checkpoint_manifest(path):
    history=verify_intake();manifest=r.read(path);audit=audit_receipt()
    if not Path(path).resolve().is_relative_to(r.BASE.resolve()):raise ValueError('manifest custody escape')
    if manifest['packet_sha256']!=history['packet_sha256'] or manifest['intake_sha256']!=r.sha(BASE/'intake.json'):
        raise ValueError('cohort not bound to intake/source')
    entries=manifest['checkpoints']
    if len(entries)!=len(c.FAMILIES) or {e['family'] for e in entries}!=set(c.FAMILIES):raise ValueError('family manifest mismatch')
    digests=set();seen=set()
    for entry in entries:
        identity=entry['model_id'].lower()
        if identity in seen or identity in history['exclusion']['model_ids'] or not metadata_eligible(identity,audit):raise ValueError('checkpoint exposure or duplicate')
        seen.add(identity)
        if not re.fullmatch('[0-9a-f]{40}',entry['revision']):raise ValueError('revision not pinned')
        for flag in ('base_model','selected_after_source_freeze','no_recorded_weight_or_execution_exposure','rights_to_load_locally'):
            if entry.get(flag) is not True:raise ValueError('checkpoint attestation '+flag)
        root=r.contained(r.BASE,entry['local_path'])
        actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        if actual!={row['path'] for row in entry['files']} or len(actual)!=len(entry['files']):raise ValueError('tree list incomplete')
        if sum(row['bytes'] for row in entry['files'])>int(1.1*1024**3):raise ValueError('asset byte cap')
        weights=[]
        for row in entry['files']:
            p=r.contained(root,row['path'])
            if p.suffix in ('.py','.bin','.pt','.pth'):raise ValueError('custom/pickle unsupported')
            if p.stat().st_size!=row['bytes'] or r.sha(p)!=row['sha256']:raise ValueError('asset bytes differ')
            if p.suffix=='.safetensors':weights.append(row['sha256'])
        if not weights or any(h in history['exclusion']['prior_digests'] or h in digests for h in weights):raise ValueError('old/duplicate/absent weights')
        digests.update(weights)
    return manifest


r.intake=intake
r.verify_intake=verify_intake
r.checkpoint_manifest=checkpoint_manifest


def acquire(execute):
    if not execute:raise PermissionError('explicit public acquisition opt-in required')
    history=verify_intake();audit=audit_receipt();root=BASE/'acquisition';root.mkdir(exist_ok=False)
    start=time.monotonic()
    from huggingface_hub import HfApi,ModelCard,hf_hub_download,snapshot_download
    from transformers import AutoConfig,AutoModelForCausalLM
    api=HfApi(endpoint='https://huggingface.co',token=False)
    catalog=r.read(CATALOG);selected=[];screens=[]
    def check():
        r.mount_guard()
        if time.monotonic()-start>adapter.ACQUISITION_SECONDS or sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>adapter.ACQUISITION_BYTES:raise RuntimeError('acquisition cap')
    try:
        for family in FAMILIES:
            candidates=catalog_candidates(catalog,family,audit)
            r.write(root/(family+'-catalog.json'),{'rows':candidates,'retained_catalog_sha256':r.sha(CATALOG)})
            for candidate in candidates:
                check();identity=candidate['model_id'];row={'family':family,'model_id':identity}
                info=api.model_info(identity,revision=candidate['revision'],files_metadata=True,timeout=30,token=False)
                if info.private or info.gated:row['reason']='PRIVATE_OR_GATED';screens.append(row);continue
                if info.sha!=candidate['revision'] or not re.fullmatch('[0-9a-f]{40}',info.sha):raise ValueError('revision mismatch')
                siblings=[s for s in info.siblings if '/' not in s.rfilename and any(fnmatch.fnmatch(s.rfilename,p) for p in adapter.PATTERNS)]
                names={s.rfilename for s in siblings};weights=[s for s in siblings if s.rfilename.endswith('.safetensors')]
                if not weights or any(s.size is None for s in siblings) or sum(s.size for s in siblings)>int(1.1*1024**3):row['reason']='SAFE_WEIGHT_OR_BYTE_LIMIT'
                elif not {'config.json','README.md'}<=names:row['reason']='METADATA_MISSING'
                elif not ('tokenizer.json' in names or 'tokenizer.model' in names or {'vocab.json','merges.txt'}<=names):row['reason']='TOKENIZER_MISSING'
                elif any(s.size>2*1024**2 for s in siblings if s.rfilename in ('config.json','README.md')):row['reason']='METADATA_BYTE_LIMIT'
                if row.get('reason'):screens.append(row);continue
                shas=[s.lfs.get('sha256') if isinstance(s.lfs,dict) else getattr(s.lfs,'sha256',None) for s in weights]
                if any(h is None or h in history['exclusion']['prior_digests'] for h in shas):row['reason']='OLD_OR_UNBOUND_WEIGHT';screens.append(row);continue
                local=root/'metadata'/family/identity.replace('/','--');local.mkdir(parents=True,exist_ok=False)
                for filename in ('config.json','README.md'):
                    hf_hub_download(identity,filename,revision=info.sha,local_dir=local,token=False,etag_timeout=30)
                config=r.read(local/'config.json');readme=(local/'README.md').read_text()
                card=ModelCard.load(local/'README.md').data.to_dict()
                row['reason']=adapter.public_base_reason(identity,card,readme) or adapter.config_reason(config,family)
                if row['reason']:screens.append(row);continue
                cfg=AutoConfig.from_pretrained(local,local_files_only=True,trust_remote_code=False)
                with torch.device('meta'):skeleton=AutoModelForCausalLM.from_config(cfg,trust_remote_code=False)
                parameters=sum(p.numel() for p in skeleton.parameters());del skeleton
                if parameters>250_000_000:row['reason']='PARAMETER_LIMIT';screens.append(row);continue
                chosen={'family':family,'model_id':identity,'revision':info.sha,'license':card['license'],'parameters':parameters,
                    'files':[{'path':s.rfilename,'bytes':s.size,'weight_sha256':shas[weights.index(s)] if s in weights else None} for s in siblings],
                    'readme_sha256':r.sha(local/'README.md'),'config_sha256':r.sha(local/'config.json')}
                selected.append(chosen);row['reason']='NOMINATED';screens.append(row);break
        r.write(root/'screen.json',{'packet_sha256':r.verify_source(),'screens':screens,'selected':selected})
        if not selected:
            answer={'outcome_status':'NO_ELIGIBLE_CHECKPOINT','qualified_families':0,'weight_downloads_opened':False}
            r.write(root/'result.json',answer);return answer
        r.write(root/'selection-freeze.json',{'selected':selected,'screen_sha256':r.sha(root/'screen.json'),
                 'cohort_scope':'FULL_THREE_FAMILY' if len(selected)==3 else 'PARTIAL_DEVELOPMENT_ONLY'})
        # Each family is an independent finite study. A missing family does not
        # erase a runnable family, and cannot be reported as nine-case success.
        for entry in selected:
            check();family=entry['family'];family_root=BASE/family;local=family_root/'checkpoint'
            local.mkdir(parents=True,exist_ok=False)
            snapshot_download(entry['model_id'],revision=entry['revision'],local_dir=local,
                allow_patterns=[row['path'] for row in entry['files']],max_workers=2,token=False,etag_timeout=30)
            check()
            if (local/'.cache').exists():shutil.rmtree(local/'.cache')
            actual=[{'path':str(p.relative_to(local)),'bytes':p.stat().st_size,'sha256':r.sha(p)} for p in sorted(local.rglob('*')) if p.is_file()]
            lookup={row['path']:row for row in actual}
            if set(lookup)!={row['path'] for row in entry['files']}:raise ValueError('download file list differs')
            for expected in entry['files']:
                row=lookup[expected['path']]
                if row['bytes']!=expected['bytes'] or expected['weight_sha256'] is not None and row['sha256']!=expected['weight_sha256']:raise ValueError('download bytes differ')
            if lookup['README.md']['sha256']!=entry['readme_sha256'] or lookup['config.json']['sha256']!=entry['config_sha256']:raise ValueError('provenance bytes differ')
            shutil.copyfile(BASE/'intake.json',family_root/'intake.json')
            r.write(family_root/'checkpoints.json',{'packet_sha256':r.verify_source(),'intake_sha256':r.sha(BASE/'intake.json'),
                'selection_sha256':r.sha(root/'selection-freeze.json'),'checkpoints':[{'family':family,
                'model_id':entry['model_id'],'revision':entry['revision'],'local_path':'checkpoint','files':actual,
                'base_model':True,'selected_after_source_freeze':True,'no_recorded_weight_or_execution_exposure':True,
                'rights_to_load_locally':True,'metadata_exposure':'KNOWN_BEFORE_PROTOCOL_FREEZE'}]})
            activate(family);checkpoint_manifest(family_root/'checkpoints.json');activate(None)
        answer={'outcome_status':'COHORT_READY' if len(selected)==3 else 'PARTIAL_DEVELOPMENT_COHORT_READY',
                'qualified_families':len(selected),'weight_downloads_opened':True,'monetary_cost_usd':0,
                'wall_seconds':time.monotonic()-start,'energy':'NOT_MEASURED'}
        r.write(root/'result.json',answer);return answer
    except BaseException as error:
        r.write(root/'error.json',{'outcome_status':'INCONCLUSIVE','exception_type':type(error).__name__,
                                 'retry_same_root_permitted':False,'monetary_cost_usd':0})
        raise


def activate(family):
    if family is not None and family not in FAMILIES:raise ValueError('unknown family')
    r.BASE=BASE/family if family else BASE;c.FAMILIES=(family,) if family else FAMILIES


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','intake','acquire','bootstrap','runtime-check','run','closeout'))
    parser.add_argument('--execute-public-acquisition',action='store_true')
    parser.add_argument('--role',choices=('primary','replica'),default='primary')
    parser.add_argument('--family',choices=FAMILIES)
    args=parser.parse_args()
    if args.action=='verify':answer={'packet_sha256':r.verify_source()}
    elif args.action=='intake':answer=intake()
    elif args.action=='acquire':answer=acquire(args.execute_public_acquisition)
    elif args.action=='bootstrap':answer=adapter.bootstrap()
    elif args.action=='runtime-check':answer=r.runtime()
    else:
        if args.family is None:parser.error('scientific actions require --family')
        activate(args.family)
        if args.action=='run':answer=r.run(r.BASE/'checkpoints.json',args.role)
        else:
            import replacement_loss_v36_closeout as closeout
            answer=closeout.validate();r.write(r.BASE/'closeout/validation.json',answer)
    print(json.dumps(answer,sort_keys=True))
