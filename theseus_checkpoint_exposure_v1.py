"""Retained-custody exposure audit, never a claim about unrecorded execution.

State slices: theseus-lab-sandbox-v1,
theseus-checkpoint-exposure-feasibility-v1,
global-experiment-assurance-lifecycle-v1. No network or model loading.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import time

REPO = Path(__file__).resolve().parent
ARTIFACTS = Path('/Volumes/PrimaryED/ResearchArtifacts')
PREVIOUS = ARTIFACTS/'theseus-lab-sandbox-v1-pretrained-whole-block-transfer-v36r2'
BASE = ARTIFACTS/'theseus-lab-sandbox-v1-checkpoint-exposure-feasibility-v1'
FREEZE = REPO/'docs/checkpoint-exposure-v1-freeze.json'
MEMBERS = ('theseus_checkpoint_exposure_v1.py', 'tests/test_checkpoint_exposure_v1.py',
           'docs/checkpoint-exposure-v1.md', 'docs/replacement-loss-v36r2-results.json',
           'docs/replacement-loss-v36r2-freeze.json', 'checkpoint_manifests.json')
ID = re.compile(r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')
CLASSES = ('METADATA', 'UNKNOWN', 'WEIGHT_RECEIPT', 'EXECUTION_RECEIPT')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def members():
    return [{'path':p, 'bytes':(REPO/p).stat().st_size, 'sha256':sha(REPO/p)} for p in MEMBERS]


def verify():
    packet=read(FREEZE)
    if packet['members']!=members() or packet['packet_sha256']!=digest(members()):
        raise ValueError('exposure source freeze differs')
    return packet['packet_sha256']


def context_class(filename, pointer, obj):
    """Metadata file sizes/download popularity are not weight or execution evidence."""
    name=Path(filename).name
    if 'frozen-packet/' in filename:
        return None  # Source copies are not new exposure events.
    if name=='architecture-preflight.json' or name=='candidate-preparation.json':
        if any(k in obj for k in ('parameter_count','selected_layer','selected_layers','tokenizer_class')):
            return 'EXECUTION_RECEIPT'
    if any(k in obj for k in ('outcome_array_sha256','outcome_sha256','treatment_effects','primary_effects')):
        return 'EXECUTION_RECEIPT'
    if name=='outcome-report.json' and '/checkpoint/' in pointer:
        return 'EXECUTION_RECEIPT'
    if 'download-manifest' in name and obj.get('downloaded_bytes',0)>0 and obj.get('downloaded_files'):
        return 'WEIGHT_RECEIPT'
    if any(k in obj for k in ('weights_sha256','raw_weight_file_sha256','weight_sha256s')):
        return 'WEIGHT_RECEIPT'
    if 'prior_' in pointer or 'known_prior' in pointer or name.startswith('prior-'):
        return 'UNKNOWN'
    if 'screen' in name or name.endswith('-catalog.json'):
        return 'METADATA'
    if name in ('config.json','README.json'):
        return 'METADATA'
    return 'UNKNOWN'


def identity_mentions(value, filename, pointer=''):
    if isinstance(value,dict):
        for key,child in value.items():
            ptr=pointer+'/'+key
            if key in ('model_id','checkpoint_id') and isinstance(child,str) and ID.fullmatch(child):
                kind=context_class(filename,ptr,value)
                if kind:yield child.lower(),kind,ptr
            if key=='ids' and filename.endswith('-catalog.json') and isinstance(child,list):
                for i,item in enumerate(child):
                    if isinstance(item,str) and ID.fullmatch(item):yield item.lower(),'METADATA',ptr+'/'+str(i)
            yield from identity_mentions(child,filename,ptr)
    elif isinstance(value,list):
        for i,child in enumerate(value):
            yield from identity_mentions(child,filename,pointer+'/'+str(i))


def exposure_status(kinds):
    if 'EXECUTION_RECEIPT' in kinds:return 'EXECUTION_RECORDED'
    if 'WEIGHT_RECEIPT' in kinds or 'WEIGHT_FILE' in kinds:return 'WEIGHTS_RECORDED'
    if 'UNKNOWN' in kinds or not kinds:return 'UNKNOWN'
    return 'METADATA_ONLY_IN_RETAINED_CUSTODY'


def audit():
    if not all(os.path.ismount(p) for p in ('/Volumes/PrimaryED','/Volumes/Backup ED')):
        raise FileNotFoundError('approved volumes not mounted')
    packet=verify()
    if BASE.exists():raise FileExistsError('audit root already exists; use verify-result')
    start=time.monotonic();parent=read(PREVIOUS/'intake.json')
    sources={row['path']:row for row in parent['historical_files']}
    for path in sorted((PREVIOUS/'acquisition').rglob('*.json')):
        relative=str(path.relative_to(ARTIFACTS))
        sources[relative]={'path':relative,'bytes':path.stat().st_size,'sha256':sha(path),'kind':'JSON_IDENTITY_RECORD'}
    sources[str((PREVIOUS/'intake.json').relative_to(ARTIFACTS))]={
        'path':str((PREVIOUS/'intake.json').relative_to(ARTIFACTS)),
        'bytes':(PREVIOUS/'intake.json').stat().st_size,'sha256':sha(PREVIOUS/'intake.json'),
        'kind':'JSON_IDENTITY_RECORD'}
    records={identity:{'counts':collections.Counter(),'examples':{}} for identity in parent['exclusion']['model_ids']}
    evidence_hash=hashlib.sha256();pointers=0;opaque=[]
    def check():
        if time.monotonic()-start>900:raise RuntimeError('audit 900 second cap')
        if not os.path.ismount('/Volumes/PrimaryED'):raise FileNotFoundError('artifact mount lost')
    def add(identity,kind,reference):
        nonlocal pointers
        record=records.setdefault(identity,{'counts':collections.Counter(),'examples':{}})
        record['counts'][kind]+=1
        record['examples'].setdefault(kind,reference)
        evidence_hash.update((json.dumps([identity,kind,reference],sort_keys=True)+'\n').encode());pointers+=1
    for relative,row in sorted(sources.items()):
        check();path=(ARTIFACTS/relative).resolve()
        if not path.is_relative_to(ARTIFACTS.resolve()):raise ValueError('historical custody escape')
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('historical input differs')
        if row['kind']=='QUARANTINED_OPAQUE_LEGACY_EXPORT':opaque.append(row);continue
        if path.stat().st_size>64*1024**2:raise ValueError('JSON size cap')
        for identity,kind,pointer in identity_mentions(read(path),relative):
            add(identity,kind,{'path':relative,'sha256':row['sha256'],'pointer':pointer})
    # Theseus-owned trees only; no imports from another project's asset custody.
    # Candidate modules are also hashed and excluded by digest, but a candidate
    # filename by itself is not attributed to a guide identity.
    weight_files=[];cached={};physical_bytes=0
    for folder in sorted(ARTIFACTS.glob('theseus*')):
        if not folder.is_dir() or folder==BASE:continue
        for directory,dirs,names in os.walk(folder,followlinks=False):
            dirs.sort()
            for name in sorted(names):
                if not name.endswith(('.safetensors','.bin','.pt','.pth')):continue
                check();path=Path(directory)/name;resolved=path.resolve()
                if not resolved.is_relative_to(ARTIFACTS.resolve()):raise ValueError('weight custody escape')
                if not resolved.is_file():raise ValueError('broken asset link')
                key=str(resolved)
                if key not in cached:
                    physical_bytes+=resolved.stat().st_size
                    if physical_bytes>36*1024**3:raise RuntimeError('audit weight byte cap')
                    cached[key]=sha(resolved)
                relative=str(path.relative_to(ARTIFACTS))
                row={'path':relative,'bytes':path.stat().st_size,'sha256':cached[key]}
                weight_files.append(row)
                for part in path.parts:
                    if part.startswith('models--'):
                        identity='/'.join(part[len('models--'):].split('--')).lower()
                        if ID.fullmatch(identity):add(identity,'WEIGHT_FILE',row)
    registry=read(REPO/'checkpoint_manifests.json')
    for identity,obj in sorted(registry.items()):
        add(identity.lower(),'WEIGHT_RECEIPT',{'path':'checkpoint_manifests.json','sha256':sha(REPO/'checkpoint_manifests.json')})
    identities={identity:{'status':exposure_status(record['counts']),
                         'counts':dict(record['counts']),'examples':record['examples']}
                for identity,record in sorted(records.items())}
    result={'protocol_id':'theseus-checkpoint-exposure-feasibility-v1','packet_sha256':packet,
        'source_revision':'202f459d6d89df9a038d78d58e0af825d287fcd8',
        'assurance_level':'INTERNAL_PASS','reviewer_status':'NOT_SUBMITTED',
        'reproducibility_status':'NOT_RUN_SEPARATELY','claim_ceiling':'RETAINED_CUSTODY_EXPOSURE_CLASSIFICATION_ONLY',
        'prior_intake_sha256':sha(PREVIOUS/'intake.json'),'sources':list(sources.values()),
        'identities':identities,'status_counts':dict(collections.Counter(v['status'] for v in identities.values())),
        'evidence_pointers':pointers,'evidence_stream_sha256':evidence_hash.hexdigest(),
        'weight_files':weight_files,'weight_digests':sorted({row['sha256'] for row in weight_files}),
        'opaque_inputs':opaque,'audit_seconds':time.monotonic()-start,
        'hashed_unique_weight_bytes':physical_bytes,'monetary_cost_usd':0,
        'limitations':['Unrecorded, removed, remote, alias and other-project exposures remain unknown.',
          'Receipt classifications are conservative exclusions, not independently authenticated execution.',
          'Metadata-only means no positive or unknown exposure record in this bound retained input set.']}
    write(BASE/'result.json',result)
    for member in (*MEMBERS,str(FREEZE.relative_to(REPO))):
        destination=BASE/'frozen-packet'/member
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((REPO/member).read_bytes())
    return {'result_sha256':sha(BASE/'result.json'),'status_counts':result['status_counts'],
            'sources':len(sources),'weight_paths':len(weight_files),'audit_seconds':result['audit_seconds']}


def verify_result():
    value=read(BASE/'result.json')
    if value['packet_sha256']!=verify():raise ValueError('audit packet binding')
    for row in [*value['sources'],*value['weight_files']]:
        path=ARTIFACTS/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('audit input changed')
    for member in members():
        if sha(BASE/'frozen-packet'/member['path'])!=member['sha256']:raise ValueError('audit archive differs')
    return {'result_sha256':sha(BASE/'result.json'),'verified':True,'status_counts':value['status_counts']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('audit','verify','verify-result'))
    args=parser.parse_args()
    answer={'packet_sha256':verify()} if args.action=='verify' else audit() if args.action=='audit' else verify_result()
    print(json.dumps(answer,sort_keys=True))
