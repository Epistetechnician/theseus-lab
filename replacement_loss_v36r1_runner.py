"""Fresh V36R1 continuation; V36 bytes and terminal outcome remain immutable.

State slices: theseus-lab-sandbox-v1,
theseus-pretrained-whole-block-transfer-v36r1,
theseus-public-checkpoint-acquisition-v36r1,
global-experiment-assurance-lifecycle-v1.
Only public static assets may be acquired; no inference API or paid workers.
"""
from __future__ import annotations

import argparse
import fnmatch
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import venv

import torch

import replacement_loss_v36_core as c
import replacement_loss_v36_runner as r

ORIGINAL_MEMBERS = r.MEMBERS
ORIGINAL_VERIFY_INTAKE = r.verify_intake
LEGACY_ROOT = 'theseus-lab-sandbox-v1-replacement-loss-v8-cloud-run-replay-v2/'
OPAQUE = {
    LEGACY_ROOT+'cloud-run-logs-final-v2.json': (550,'2ee43ebd7db99520d6274cf5e0930df48b435749530d24ae69007e6d88d613de'),
    LEGACY_ROOT+'cloud-run-logs-progress-v2.json': (0,'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
    LEGACY_ROOT+'cloud-run-task-start-v2.json': (0,'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
}
c.PROTOCOL = 'theseus-pretrained-whole-block-transfer-v36r1'
c.SLICE = c.PROTOCOL
r.BASE = r.ARTIFACTS/'theseus-lab-sandbox-v1-pretrained-whole-block-transfer-v36r1'
r.FREEZE = r.REPO/'docs/replacement-loss-v36r1-freeze.json'
r.MEMBERS = (*ORIGINAL_MEMBERS, 'docs/replacement-loss-v36-freeze.json',
             'docs/replacement-loss-v36-results.json', 'replacement_loss_v36r1_runner.py',
             'tests/test_replacement_loss_v36r1.py', 'docs/replacement-loss-v36r1-continuation.md')
RUNTIMES = r.RUNTIME_VOLUME/'ResearchRuntimes/theseus-lab-v36r1/toolchains'
LICENSES = {'apache-2.0','mit','bsd-3-clause','bsd-2-clause','cc0-1.0','cc-by-4.0'}
PATTERNS = ('config.json','generation_config.json','tokenizer*.json','tokenizer.model',
            'special_tokens_map.json','added_tokens.json','vocab.json','vocab.txt','merges.txt',
            '*.safetensors','model.safetensors.index.json','README.md')
FAMILY_TYPE = {'opt_relu':'opt','gpt2_gelu':'gpt2','llama_silu_gated':'llama'}
REJECT_NAMES = re.compile(r'random|dummy|instruct|chat|fine[-_]?tun|sft|lora|gguf|gptq|awq',re.I)
CATALOG_LIMIT = 200
ACQUISITION_SECONDS = 1800
ACQUISITION_BYTES = 8*1024**3


def classify(path, relative, opaque=OPAQUE):
    """Unknown malformed records fail; only exact known exports are quarantined."""
    size=path.stat().st_size
    if size>64*1024**2:
        raise ValueError('historical JSON byte ceiling')
    h=r.sha(path)
    if relative in opaque:
        if (size,h)!=opaque[relative]:
            raise ValueError('known opaque legacy export changed')
        return None, {'path':relative,'bytes':size,'sha256':h,'kind':'QUARANTINED_OPAQUE_LEGACY_EXPORT'}
    value=json.loads(path.read_text())
    return value, {'path':relative,'bytes':size,'sha256':h,'kind':'JSON_IDENTITY_RECORD'}


def intake():
    r.mount_guard();packet=r.verify_source();parent=r.verify_parent()
    if r.BASE.exists():
        raise FileExistsError('fresh continuation intake already exists')
    sets={key:set() for key in ('model_ids','text_hashes','token_hashes','prior_digests')}
    files=[];quarantined=[]
    for folder in sorted(r.ARTIFACTS.glob('theseus*')):
        if not folder.is_dir() or folder.resolve()==r.BASE.resolve():continue
        for path in sorted(folder.rglob('*.json')):
            if not path.resolve().is_relative_to(r.ARTIFACTS.resolve()):
                raise ValueError('historical custody escape')
            relative=str(path.relative_to(r.ARTIFACTS))
            value,row=classify(path,relative);files.append(row)
            if value is None and row['kind']=='QUARANTINED_OPAQUE_LEGACY_EXPORT':quarantined.append(row)
            else:r.collect_history(value,sets)
    if {row['path'] for row in quarantined}!=set(OPAQUE):
        raise ValueError('known legacy export custody incomplete')
    registry=r.read(r.REPO/'checkpoint_manifests.json');r.collect_history(registry,sets)
    sets['model_ids'].update(key.lower() for key in registry)
    exclusion={key:sorted(value) for key,value in sets.items()}
    r.write(r.BASE/'intake.json',{'protocol_id':c.PROTOCOL,'packet_sha256':packet,
       'parent_packet_sha256':parent,'historical_files':files,'quarantined_legacy_exports':quarantined,
       'exclusion':exclusion,'exclusion_sha256':c.digest(exclusion),
       'checkpoint_selection_status':'NOT_SELECTED','claim_ceiling':c.CLAIM})
    for member in (*r.MEMBERS,str(r.FREEZE.relative_to(r.REPO))):
        target=r.BASE/'frozen-packet'/member;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(r.REPO/member,target)
    return {'packet_sha256':packet,'intake_sha256':r.sha(r.BASE/'intake.json'),
            'historical_files':len(files),'quarantined':len(quarantined)}


def verify_intake():
    receipt=ORIGINAL_VERIFY_INTAKE()
    rows=receipt['quarantined_legacy_exports']
    if len(rows)!=len(OPAQUE) or {row['path'] for row in rows}!=set(OPAQUE):
        raise ValueError('opaque quarantine inventory differs')
    for row in rows:
        if (row['bytes'],row['sha256'])!=OPAQUE[row['path']] or row['kind']!='QUARANTINED_OPAQUE_LEGACY_EXPORT':
            raise ValueError('opaque quarantine binding differs')
    return receipt


r.intake=intake
r.verify_intake=verify_intake


def expected_packages():
    return dict(line.split('==') for line in (r.REPO/'requirements-v36.lock.txt').read_text().splitlines() if line and not line.startswith('#'))


def bootstrap():
    r.mount_guard();r.verify_source()
    if RUNTIMES.exists():raise FileExistsError('runtime root already exists; preserve it')
    source=Path(torch.__file__).resolve().parent.parent;entries={}
    for name,version in expected_packages().items():
        distribution=importlib.metadata.distribution(name)
        if distribution.version!=version:raise ValueError('source dependency version differs')
        for item in distribution.files or ():
            path=Path(distribution.locate_file(item)).resolve()
            if not path.is_relative_to(source) or path.suffix=='.pyc':continue
            if not path.is_file():raise ValueError('source dependency file missing')
            entries[str(path.relative_to(source))]={'bytes':path.stat().st_size,'sha256':r.sha(path)}
    if sum(value['bytes'] for value in entries.values())*2>1024**3:
        raise ValueError('runtime-copy byte cap')
    for role in ('primary','replica'):
        destination=RUNTIMES/role
        venv.EnvBuilder(with_pip=False,symlinks=True).create(destination)
        site=destination/'lib/python3.12/site-packages'
        for relative,row in entries.items():
            target=site/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/relative,target)
            if target.stat().st_size!=row['bytes'] or r.sha(target)!=row['sha256']:
                raise ValueError('dependency copy failed')
    receipt={'protocol_id':c.PROTOCOL,'packet_sha256':r.verify_source(),'roles':['primary','replica'],
             'files_per_runtime':len(entries),'dependency_files_sha256':c.digest(entries),
             'copied_bytes_per_runtime':sum(row['bytes'] for row in entries.values()),'packages':expected_packages()}
    r.write(RUNTIMES/'bootstrap.json',receipt)
    return receipt


def public_base_reason(identity,card,readme):
    """Conservative publisher-provenance screen, never independent authority."""
    if REJECT_NAMES.search(identity):return 'DERIVATIVE_OR_RANDOM_NAME'
    if not isinstance(card.get('license'),str) or card['license'] not in LICENSES:return 'LICENSE_NOT_IN_DECLARED_SET'
    if card.get('base_model'):return 'DERIVED_FROM_ANOTHER_CHECKPOINT'
    if not re.search(r'(pre[- ]?trained|trained) from scratch',readme,re.I):
        return 'FROM_SCRATCH_PROVENANCE_NOT_DECLARED'
    return None


def config_reason(config,family):
    if config.get('model_type')!=FAMILY_TYPE[family]:return 'ARCHITECTURE'
    if config.get('auto_map') or config.get('quantization_config'):return 'CUSTOM_CODE_OR_QUANTIZED'
    width=config.get('hidden_size',config.get('n_embd',0))
    depth=config.get('num_hidden_layers',config.get('n_layer',0))
    vocabulary=config.get('vocab_size',0)
    if any(type(value) is not int for value in (width,depth,vocabulary)) or not 8<=width<=1024 or depth<4 or not 2<=vocabulary<=65536:return 'SHAPE_LIMIT'
    activation=config.get('activation_function',config.get('hidden_act'))
    if activation not in {'opt_relu':('relu',),'gpt2_gelu':('gelu','gelu_new','gelu_fast'),
                          'llama_silu_gated':('silu',)}[family]:return 'ACTIVATION'
    return None


def acquire(execute):
    """Bounded metadata-only nomination, then one fixed cohort's static assets."""
    if not execute:raise PermissionError('explicit --execute-public-acquisition required')
    history=verify_intake();root=r.BASE/'acquisition'
    root.mkdir(exist_ok=False);start=time.monotonic()
    from huggingface_hub import HfApi,ModelCard,hf_hub_download,snapshot_download
    from transformers import AutoConfig,AutoModelForCausalLM
    for name,version in expected_packages().items():
        if importlib.metadata.version(name)!=version:raise ValueError('operator dependency version differs')
    if platform.python_version()!='3.12.13':raise ValueError('operator Python version differs')
    r.write(root/'operator-runtime.json',{'python':platform.python_version(),'prefix':sys.prefix,
            'packages':expected_packages(),**r.dependency_fingerprint()})
    api=HfApi(endpoint='https://huggingface.co',token=False)
    def check():
        r.mount_guard()
        if time.monotonic()-start>ACQUISITION_SECONDS or sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>ACQUISITION_BYTES:
            raise RuntimeError('acquisition resource cap')
    screens=[];selected=[]
    try:
        for family,query in (('opt_relu','opt'),('gpt2_gelu','gpt2'),('llama_silu_gated','llama')):
            check()
            catalog=list(api.list_models(search=query,pipeline_tag='text-generation',filter='transformers',
                num_parameters='max:250M',sort='created_at',limit=CATALOG_LIMIT,full=True,
                cardData=True,fetch_config=True,token=False))
            # A declared metadata catalog; no model output is used for selection.
            r.write(root/(family+'-catalog.json'),{'ids':[item.id for item in catalog]})
            chosen=None
            for item in sorted(catalog,key=lambda value:value.id.lower()):
                check();identity=item.id
                reason=None
                if identity.lower() in history['exclusion']['model_ids']:reason='RECORDED_IDENTITY'
                elif REJECT_NAMES.search(identity):reason='DERIVATIVE_OR_RANDOM_NAME'
                elif (item.config or {}).get('model_type')!=FAMILY_TYPE[family]:reason='ARCHITECTURE'
                card=item.card_data.to_dict() if item.card_data is not None else {}
                if reason is None and (not isinstance(card.get('license'),str) or card['license'] not in LICENSES):reason='LICENSE_NOT_IN_DECLARED_SET'
                if reason is None and card.get('base_model'):reason='DERIVED_FROM_ANOTHER_CHECKPOINT'
                row={'family':family,'model_id':identity,'reason':reason}
                if reason:screens.append(row);continue
                info=api.model_info(identity,files_metadata=True,timeout=30,token=False)
                if info.private or info.gated:row['reason']='PRIVATE_OR_GATED';screens.append(row);continue
                if not re.fullmatch('[0-9a-f]{40}',info.sha or ''):raise ValueError('unpinned model revision')
                files=[s for s in info.siblings if '/' not in s.rfilename and any(fnmatch.fnmatch(s.rfilename,p) for p in PATTERNS)]
                weights=[s for s in files if s.rfilename.endswith('.safetensors')]
                if not weights or any(s.size is None for s in files) or sum(s.size for s in files)>int(1.1*1024**3):
                    row['reason']='SAFE_WEIGHT_OR_FILE_LIMIT';screens.append(row);continue
                names={s.rfilename for s in files}
                if not ('tokenizer.json' in names or 'tokenizer.model' in names or {'vocab.json','merges.txt'}<=names):
                    row['reason']='TOKENIZER_MISSING';screens.append(row);continue
                if any(s.size>2*1024**2 for s in files if s.rfilename in ('config.json','README.md')):
                    row['reason']='METADATA_BYTE_LIMIT';screens.append(row);continue
                weight_shas=[getattr(s.lfs,'sha256',None) if not isinstance(s.lfs,dict) else s.lfs.get('sha256') for s in weights]
                if any(h is None or h in history['exclusion']['prior_digests'] for h in weight_shas):
                    row['reason']='REUSED_OR_UNBOUND_WEIGHT';screens.append(row);continue
                local=root/'metadata'/family/identity.replace('/','--');local.mkdir(parents=True,exist_ok=False)
                for filename in ('config.json','README.md'):
                    if filename not in {s.rfilename for s in files}:
                        row['reason']='METADATA_INCOMPLETE';break
                    hf_hub_download(identity,filename,revision=info.sha,local_dir=local,token=False,
                        endpoint='https://huggingface.co',etag_timeout=30)
                if row['reason']:screens.append(row);continue
                config=r.read(local/'config.json');readme=(local/'README.md').read_text()
                try:pinned_card=ModelCard.load(local/'README.md').data.to_dict()
                except (ValueError,TypeError):
                    row['reason']='MODEL_CARD_SCHEMA';screens.append(row);continue
                row['reason']=public_base_reason(identity,pinned_card,readme) or config_reason(config,family)
                if row['reason']:screens.append(row);continue
                cfg=AutoConfig.from_pretrained(local,local_files_only=True,trust_remote_code=False)
                with torch.device('meta'):
                    skeleton=AutoModelForCausalLM.from_config(cfg,trust_remote_code=False)
                parameters=sum(p.numel() for p in skeleton.parameters());del skeleton
                if parameters>250_000_000:
                    row['reason']='PARAMETER_LIMIT';screens.append(row);continue
                chosen={'family':family,'model_id':identity,'revision':info.sha,'license':pinned_card['license'],
                    'parameters':parameters,'files':[{'path':s.rfilename,'bytes':s.size,
                      'weight_sha256':weight_shas[weights.index(s)] if s in weights else None} for s in files],
                    'readme_sha256':r.sha(local/'README.md'),'config_sha256':r.sha(local/'config.json')}
                row['reason']='NOMINATED';screens.append(row);break
            if chosen is not None:selected.append(chosen)
        r.write(root/'screen.json',{'protocol_id':c.PROTOCOL,'packet_sha256':r.verify_source(),
            'intake_sha256':r.sha(r.BASE/'intake.json'),'screens':screens,'selected':selected})
        if len(selected)!=3:
            result={'outcome_status':'NO_ELIGIBLE_CHECKPOINT_COHORT','qualified_families':len(selected),
                'weight_downloads_opened':False,'monetary_cost_usd':0,'energy':'NOT_MEASURED'}
            r.write(root/'result.json',result);return result
        # Bind all nominations before downloading any candidate weight.
        r.write(root/'selection-freeze.json',{'selected':selected,'screen_sha256':r.sha(root/'screen.json')})
        checkpoints=[]
        for entry in selected:
            check();local=r.BASE/'checkpoints'/entry['family']
            local.mkdir(parents=True,exist_ok=False)
            snapshot_download(entry['model_id'],revision=entry['revision'],local_dir=local,
                allow_patterns=[row['path'] for row in entry['files']],max_workers=2,
                token=False,endpoint='https://huggingface.co',etag_timeout=30)
            check()
            # Drop only Hub download bookkeeping created by this adapter;
            # actual public inputs and candidate weights are retained.
            bookkeeping=local/'.cache'
            if bookkeeping.exists():shutil.rmtree(bookkeeping)
            actual=[{'path':str(p.relative_to(local)),'bytes':p.stat().st_size,'sha256':r.sha(p)} for p in sorted(local.rglob('*')) if p.is_file()]
            lookup={row['path']:row for row in actual}
            if set(lookup)!={row['path'] for row in entry['files']}:
                raise ValueError('acquired file list differs')
            for expected in entry['files']:
                row=lookup[expected['path']]
                if row['bytes']!=expected['bytes'] or (expected['weight_sha256'] is not None and row['sha256']!=expected['weight_sha256']):
                    raise ValueError('pinned public artifact differs')
            if lookup['README.md']['sha256']!=entry['readme_sha256'] or lookup['config.json']['sha256']!=entry['config_sha256']:
                raise ValueError('pinned model provenance differs')
            checkpoints.append({'family':entry['family'],'model_id':entry['model_id'],'revision':entry['revision'],
                'local_path':str(local.relative_to(r.BASE)),'files':actual,'base_model':True,
                'selected_after_source_freeze':True,'not_used_for_protocol_design':True,'rights_to_load_locally':True})
        manifest=r.BASE/'checkpoints.json'
        r.write(manifest,{'packet_sha256':r.verify_source(),'intake_sha256':r.sha(r.BASE/'intake.json'),
            'checkpoints':checkpoints,'selection_sha256':r.sha(root/'selection-freeze.json')})
        r.checkpoint_manifest(manifest)
        result={'outcome_status':'COHORT_READY','checkpoint_manifest_sha256':r.sha(manifest),
                'monetary_cost_usd':0,'energy':'NOT_MEASURED','wall_seconds':time.monotonic()-start}
        r.write(root/'result.json',result);return result
    except BaseException as error:
        r.write(root/'error.json',{'outcome_status':'INCONCLUSIVE','exception_type':type(error).__name__,
             'retry_same_root_permitted':False,'monetary_cost_usd':0,'energy':'NOT_MEASURED'})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','status','intake','bootstrap','runtime-check','acquire','run','closeout'))
    parser.add_argument('--execute-public-acquisition',action='store_true')
    parser.add_argument('--role',choices=('primary','replica'),default='primary')
    args=parser.parse_args()
    try:
        if args.action=='verify':answer={'packet_sha256':r.verify_source()}
        elif args.action=='status':answer=r.status()
        elif args.action=='intake':answer=intake()
        elif args.action=='bootstrap':answer=bootstrap()
        elif args.action=='runtime-check':answer=r.runtime()
        elif args.action=='acquire':answer=acquire(args.execute_public_acquisition)
        elif args.action=='run':answer=r.run(r.BASE/'checkpoints.json',args.role)
        else:
            import replacement_loss_v36_closeout as audit
            answer=audit.validate();r.write(r.BASE/'closeout/validation.json',answer)
        print(json.dumps(answer,sort_keys=True))
    except BaseException as error:
        if isinstance(error,SystemExit):raise
        print(json.dumps({'outcome_status':'INCONCLUSIVE','exception_type':type(error).__name__,
                         'retry_same_root_permitted':False}),flush=True)
        raise SystemExit(1)
