"""Cache-first continuation of stopped V37; no loss or gate changes.

State slices: theseus-lab-sandbox-v1, theseus-pretrained-development-transfer-v37r1,
theseus-public-checkpoint-acquisition-v37r1, global-experiment-assurance-lifecycle-v1.
"""
import argparse
import json
from pathlib import Path

import replacement_loss_v37_runner as previous

c,r=previous.c,previous.r
STOPPED_BASE=previous.BASE
CACHE_MANIFEST=r.REPO/'docs/replacement-loss-v37-metadata-cache-v1.json'
ORIGINAL_CANDIDATES=previous.catalog_candidates
c.PROTOCOL=c.SLICE='theseus-pretrained-development-transfer-v37r1'
previous.BASE=r.BASE=r.ARTIFACTS/'theseus-lab-sandbox-v1-pretrained-development-transfer-v37r1'
r.FREEZE=r.REPO/'docs/replacement-loss-v37r1-freeze.json'
r.MEMBERS=(*r.MEMBERS,'docs/replacement-loss-v37-freeze.json','docs/replacement-loss-v37-results.json',
           'docs/replacement-loss-v37-metadata-cache-v1.json','replacement_loss_v37r1_runner.py',
           'tests/test_replacement_loss_v37r1.py','docs/replacement-loss-v37r1-cache-continuation.md')
previous.adapter.RUNTIMES=r.RUNTIME_VOLUME/'ResearchRuntimes/theseus-lab-v37r1/toolchains'


def cache_verified():
    manifest=r.read(CACHE_MANIFEST)
    if r.sha(STOPPED_BASE/'acquisition/error.json')!=manifest['terminal_error_sha256']:
        raise ValueError('stopped acquisition error differs')
    if (STOPPED_BASE/'acquisition/selection-freeze.json').exists() or any(STOPPED_BASE.rglob('*.safetensors')):
        raise ValueError('stopped acquisition unexpectedly progressed')
    for row in manifest['files']:
        path=r.contained(STOPPED_BASE,row['path'])
        if path.stat().st_size!=row['bytes'] or r.sha(path)!=row['sha256']:raise ValueError('cached public metadata differs')
    return manifest


def cache_candidates(catalog,family,audit,limit=200):
    from huggingface_hub import ModelCard
    cache_verified()
    result=[]
    for row in ORIGINAL_CANDIDATES(catalog,family,audit,limit):
        local=STOPPED_BASE/'acquisition/metadata'/family/row['model_id'].replace('/','--')
        if not all((local/name).is_file() for name in ('config.json','README.md')):continue
        card=ModelCard.load(local/'README.md').data.to_dict()
        reason=previous.adapter.public_base_reason(row['model_id'],card,(local/'README.md').read_text())
        reason=reason or previous.adapter.config_reason(r.read(local/'config.json'),family)
        if not reason:result.append(row)
    if len(result)>30:raise ValueError('cache-first API candidate cap')
    return result


previous.catalog_candidates=cache_candidates


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','intake','acquire','bootstrap','runtime-check','run','closeout'))
    parser.add_argument('--execute-public-acquisition',action='store_true')
    parser.add_argument('--role',choices=('primary','replica'),default='primary')
    parser.add_argument('--family',choices=previous.FAMILIES)
    args=parser.parse_args()
    cache_verified()
    if args.action=='verify':answer={'packet_sha256':r.verify_source()}
    elif args.action=='intake':answer=previous.intake()
    elif args.action=='acquire':answer=previous.acquire(args.execute_public_acquisition)
    elif args.action=='bootstrap':answer=previous.adapter.bootstrap()
    elif args.action=='runtime-check':answer=r.runtime()
    else:
        if args.family is None:parser.error('scientific actions require --family')
        previous.activate(args.family)
        if args.action=='run':answer=r.run(r.BASE/'checkpoints.json',args.role)
        else:
            import replacement_loss_v36_closeout as audit
            answer=audit.validate();r.write(r.BASE/'closeout/validation.json',answer)
    print(json.dumps(answer,sort_keys=True))
