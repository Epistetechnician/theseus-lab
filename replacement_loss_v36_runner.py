"""Custody-gated V36 lifecycle. All study arrays/text/weights stay external.

State slices: theseus-lab-sandbox-v1, theseus-pretrained-whole-block-transfer-v36,
global-experiment-assurance-lifecycle-v1. Runtime is local/offline CPU only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import secrets
import shutil
import sys
import time

import numpy as np
import torch

import replacement_loss_v36_core as c
import replacement_loss_v36_fixtures as f


REPO = Path(__file__).resolve().parent
PRIMARY_VOLUME = Path('/Volumes/PrimaryED')
RUNTIME_VOLUME = Path('/Volumes/Backup ED')
ARTIFACTS = PRIMARY_VOLUME / 'ResearchArtifacts'
BASE = ARTIFACTS / 'theseus-lab-sandbox-v1-pretrained-whole-block-transfer-v36'
FREEZE = REPO / 'docs/replacement-loss-v36-freeze.json'
PARENT = ARTIFACTS / 'theseus-lab-sandbox-v1-replacement-loss-v35'
MEMBERS = ('replacement_loss_v36_core.py', 'replacement_loss_v36_fixtures.py',
           'replacement_loss_v36_runner.py', 'replacement_loss_v36_closeout.py', 'tests/test_replacement_loss_v36.py',
           'docs/replacement-loss-v36-pretrained-transfer.md', 'requirements-v36.lock.txt',
           'docs/replacement-loss-v35-results.json', 'docs/replacement-loss-v35-freeze.json',
           'checkpoint_manifests.json')
MAX_RSS = 6 * 1024**3
MAX_BYTES = 24 * 1024**3
MAX_SECONDS = 24 * 3600
RECONCILIATION_KEYS = ('protocol_id','state_slice','packet_sha256','outcome_status','claim_ceiling',
                       'challenge_generated','preparation','metrics','outcome_bindings')


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('JSON object required')
    return value


def write(path, value):
    path = Path(path)
    if path.is_relative_to(BASE):
        mount_guard()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as file:
        file.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def freeze_material():
    return [{'path':p, 'bytes':(REPO/p).stat().st_size, 'sha256':sha(REPO/p)} for p in MEMBERS]


def verify_source():
    packet = read(FREEZE)
    if packet['members'] != freeze_material() or packet['packet_sha256'] != c.digest(packet['members']):
        raise ValueError('frozen source differs')
    if packet['claim_ceiling'] != c.CLAIM or packet['protocol_id'] != c.PROTOCOL:
        raise ValueError('frozen protocol differs')
    return packet['packet_sha256']


def mount_guard():
    if not all(p.is_dir() and os.path.ismount(p) for p in (PRIMARY_VOLUME, RUNTIME_VOLUME)):
        raise FileNotFoundError('approved external artifact/runtime volumes are not mounted')
    if not ARTIFACTS.is_dir():
        raise FileNotFoundError('existing artifact custody is absent')


def contained(root, relative):
    if Path(relative).is_absolute():
        raise ValueError('relative custody path required')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path.is_relative_to(REPO):
        raise ValueError('custody path escape or repository asset')
    return path


def verify_parent():
    report = read(REPO/'docs/replacement-loss-v35-results.json')
    if report['scientific_outcome'] != 'PASS' or report['assurance_level'] != 'REPRODUCED_LOCAL':
        raise ValueError('parent scientific gate')
    for phase, receipt in report['phases'].items():
        for role in ('primary','replica'):
            root = PARENT/phase/role
            if sha(root/'result.json') != receipt['result_digests'][role]:
                raise ValueError('parent result binding')
            for member in read(root/'result.json')['artifact_manifest']:
                path = contained(root, member['path'])
                if path.stat().st_size != member['bytes'] or sha(path) != member['sha256']:
                    raise ValueError('parent custody artifact differs')
        if sha(PARENT/phase/'replica/reconciliation.json') != receipt['reconciliation_sha256']:
            raise ValueError('parent reconciliation binding')
        if sha(PARENT/'closeout'/(phase+'-validation.json')) != receipt['validation_sha256']:
            raise ValueError('parent closeout binding')
    return report['packet_sha256']


def collect_history(value, sets):
    """Read predecessor records without emitting prompts, arrays, or identities."""
    if isinstance(value, list):
        for child in value:
            collect_history(child, sets)
    elif isinstance(value, dict):
        for key, child in value.items():
            if key in ('model_id','checkpoint_id') and isinstance(child,str):
                sets['model_ids'].add(child.strip().lower())
            elif key in ('input_text','prompt','text') and isinstance(child,str):
                sets['text_hashes'].add(f.text_hash(child))
            elif key in ('ids','input_ids','prompt_ids') and isinstance(child,list) and child and all(type(v) is int for v in child):
                sets['token_hashes'].add(c.digest(child))
            elif key in ('text_sha256','prompt_sha256') and isinstance(child,str):
                sets['text_hashes'].add(child)
            if isinstance(child,str) and len(child)==64 and all(x in '0123456789abcdef' for x in child):
                sets['prior_digests'].add(child)
            collect_history(child, sets)


def intake():
    mount_guard()
    packet = verify_source()
    parent = verify_parent()
    if BASE.exists():
        raise FileExistsError('V36 intake already exists; preserve exact identity')
    sets = {k:set() for k in ('model_ids','text_hashes','token_hashes','prior_digests')}
    files = []
    # Snapshot only predecessor roots. Never re-enumerate after own replicas
    # exist; the explicit immutable list prevents V28's expanding-ledger bug.
    for root in sorted(ARTIFACTS.glob('theseus*')):
        if not root.is_dir() or root.resolve()==BASE.resolve():
            continue
        for path in sorted(root.rglob('*.json')):
            if not path.resolve().is_relative_to(ARTIFACTS.resolve()):
                raise ValueError('predecessor custody link escapes artifact root')
            if path.stat().st_size > 64*1024**2:
                raise ValueError('predecessor JSON exceeds fixed intake bound')
            collect_history(json.loads(path.read_text()), sets)
            files.append({'path':str(path.relative_to(ARTIFACTS)), 'sha256':sha(path)})
    collect_history(read(REPO/'checkpoint_manifests.json'), sets)
    for key in read(REPO/'checkpoint_manifests.json'):
        sets['model_ids'].add(key.lower())
    exclusion = {k:sorted(v) for k,v in sets.items()}
    payload = {'protocol_id':c.PROTOCOL, 'packet_sha256':packet, 'parent_packet_sha256':parent,
               'historical_files':files, 'exclusion':exclusion, 'exclusion_sha256':c.digest(exclusion),
               'checkpoint_selection_status':'NOT_SELECTED', 'claim_ceiling':c.CLAIM}
    write(BASE/'intake.json',payload)
    for member in (*MEMBERS, str(FREEZE.relative_to(REPO))):
        target = BASE/'frozen-packet'/member
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/member,target)
    return {'packet_sha256':packet, 'historical_files':len(files), 'intake_sha256':sha(BASE/'intake.json')}


def verify_intake():
    mount_guard()
    receipt = read(BASE/'intake.json')
    if receipt['packet_sha256'] != verify_source() or receipt['exclusion_sha256'] != c.digest(receipt['exclusion']):
        raise ValueError('intake binding changed')
    if receipt['parent_packet_sha256']!=verify_parent():
        raise ValueError('intake parent binding changed')
    for member in receipt['historical_files']:
        if sha(contained(ARTIFACTS,member['path'])) != member['sha256']:
            raise ValueError('historical custody changed')
    for member in freeze_material():
        if sha(BASE/'frozen-packet'/member['path']) != member['sha256']:
            raise ValueError('source archive changed')
    if sha(BASE/'frozen-packet'/FREEZE.relative_to(REPO))!=sha(FREEZE):
        raise ValueError('source freeze archive changed')
    return receipt


def checkpoint_manifest(path):
    history = verify_intake()
    path = Path(path).resolve()
    if not path.is_relative_to(BASE.resolve()):
        raise ValueError('checkpoint manifest must be inside V36 custody')
    manifest = read(path)
    if manifest['packet_sha256'] != history['packet_sha256'] or manifest['intake_sha256'] != sha(BASE/'intake.json'):
        raise ValueError('selection not bound to frozen source and historical intake')
    entries = manifest['checkpoints']
    if len(entries)!=3 or {e['family'] for e in entries}!=set(c.FAMILIES):
        raise ValueError('exactly one checkpoint per declared architecture required')
    seen, weight_shas = set(), set()
    for e in entries:
        identity = e['model_id'].strip().lower()
        if not identity or identity in history['exclusion']['model_ids'] or identity in seen:
            raise ValueError('checkpoint is reused or duplicate')
        seen.add(identity)
        if len(e['revision'])!=40 or any(x not in '0123456789abcdef' for x in e['revision']):
            raise ValueError('full revision required')
        for flag in ('base_model','selected_after_source_freeze','not_used_for_protocol_design','rights_to_load_locally'):
            if e.get(flag) is not True:
                raise ValueError('checkpoint attestation missing: '+flag)
        root=contained(BASE,e['local_path'])
        if not root.is_dir() or not e['files']:
            raise ValueError('checkpoint custody is missing')
        actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        if len(e['files'])!=len(actual) or actual != {r['path'] for r in e['files']}:
            raise ValueError('checkpoint tree manifest incomplete')
        weights=[]
        if sum(row['bytes'] for row in e['files'])>int(1.1*1024**3):
            raise ValueError('checkpoint file-byte ceiling')
        for row in e['files']:
            p=contained(root,row['path'])
            if p.suffix=='.py' or p.suffix in ('.bin','.pt','.pth'):
                raise ValueError('remote/custom code and pickle weights unsupported')
            if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
                raise ValueError('checkpoint artifact digest differs')
            if p.suffix=='.safetensors':
                weights.append(row['sha256'])
        if not weights or any(h in history['exclusion']['prior_digests'] or h in weight_shas for h in weights):
            raise ValueError('checkpoint weight bytes reused or absent')
        weight_shas.update(weights)
    return manifest


def dependency_fingerprint():
    """Hash installed dependency files, excluding caches and external shims."""
    site=Path(torch.__file__).resolve().parent.parent
    rows={}
    for distribution in importlib.metadata.distributions():
        for member in distribution.files or ():
            path=Path(distribution.locate_file(member)).resolve()
            if not path.is_relative_to(site) or path.suffix=='.pyc':
                continue
            if not path.is_file():
                raise ValueError('installed dependency file is missing')
            relative=str(path.relative_to(site))
            rows[relative]={'bytes':path.stat().st_size,'sha256':sha(path)}
    if not rows:
        raise ValueError('installed dependency inventory is empty')
    return {'dependency_files_sha256':c.digest(rows),'dependency_file_count':len(rows),
            'python_executable_sha256':sha(Path(sys.executable).resolve())}


def runtime():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    inventory={d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()}
    expected=dict(line.split('==') for line in (REPO/'requirements-v36.lock.txt').read_text().splitlines() if line and not line.startswith('#'))
    if inventory!=expected or platform.python_version()!='3.12.13':
        raise ValueError('exact isolated V36 runtime lock required')
    cfg=Path(sys.prefix)/'pyvenv.cfg'
    if not cfg.is_file() or 'include-system-site-packages = false' not in cfg.read_text().lower():
        raise ValueError('isolated runtime required')
    if not Path(sys.prefix).resolve().is_relative_to(RUNTIME_VOLUME.resolve()):
        raise ValueError('study runtime must use approved external runtime custody')
    return {'prefix':sys.prefix,'site_packages':str(Path(torch.__file__).resolve().parent.parent),
            'python':platform.python_version(),'packages':inventory,'platform':platform.platform(),'threads':1,'device':'cpu',
            **dependency_fingerprint()}


def load_model(entry):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root=contained(BASE,entry['local_path'])
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,trust_remote_code=False,
                                            dtype=torch.float32,attn_implementation='eager',use_safetensors=True)
    expected={'opt_relu':'opt','gpt2_gelu':'gpt2','llama_silu_gated':'llama'}[entry['family']]
    if model.config.model_type!=expected or sum(p.numel() for p in model.parameters())>250_000_000:
        raise ValueError('architecture or 250M parameter contract')
    activation=getattr(model.config,'activation_function',getattr(model.config,'hidden_act',None))
    valid={'opt_relu':('relu',),'gpt2_gelu':('gelu','gelu_new','gelu_fast'),'llama_silu_gated':('silu',)}
    if activation not in valid[entry['family']]:
        raise ValueError('activation contract')
    model.eval().requires_grad_(False)
    if model.config.hidden_size<8 or model.config.hidden_size>1024 or len(c.layers(model,entry['family']))<4 or model.config.vocab_size>65536:
        raise ValueError('hidden-width/depth/vocabulary contract')
    return model,tokenizer


def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def save(path,value):
    if path.is_relative_to(BASE):
        mount_guard()
    if path.exists():
        raise FileExistsError('array artifact already exists')
    np.savez(path,**value)
    return c.array_digest(value)


def budget(start, root):
    mount_guard()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if time.monotonic()-start>MAX_SECONDS or rss>MAX_RSS or size>MAX_BYTES:
        raise RuntimeError('frozen study resource cap')
    return {'wall_seconds':time.monotonic()-start,'peak_rss_bytes':rss,'artifact_bytes':size,
            'artifact_volume_free_bytes':shutil.disk_usage(BASE).free,'monetary_cost_usd':0,'energy':'NOT_MEASURED'}


def reserve_space():
    """Conservative per-run reserve; measurement is not a disk reservation."""
    if shutil.disk_usage(BASE).free<MAX_BYTES+1024**3:
        raise RuntimeError('artifact free-space reserve is below the frozen cap plus 1 GiB')


def qualify(model,family,rows,check):
    sites=list(range(len(c.layers(model,family))))
    baseline=[]; energy={s:0. for s in sites}
    for row in rows:
        check();p=c.forward(model,family,row['ids'],sites);baseline.append(p['logits'])
        for s in sites:energy[s]+=float(p['sites'][s]['y'].square().mean())
    halves=(sites[:len(sites)//2],sites[len(sites)//2:])
    chosen=[min(half,key=lambda s:(-energy[s],s)) for half in halves]
    metrics={}
    for s in chosen:
        outputs=[]
        with c.installed(model,family,{s:c.Zero()}):
            for row in rows:
                check();outputs.append(c.forward(model,family,row['ids'],chosen)['logits'])
        b,o=np.stack(baseline),np.stack(outputs)
        metrics[str(s)]={'removal_logit_nrmse':c.nrmse(o,b),'removal_top1_change':float(np.mean(o.argmax(1)!=b.argmax(1)))}
    passed=all(m['removal_logit_nrmse']>=.1 and m['removal_top1_change']>=.2 for m in metrics.values())
    return {'sites':chosen,'visible_energy':{str(s):v for s,v in energy.items()},'removal':metrics,'passed':passed}


def predict(model,family,rows,donors,active,entropy,case,check):
    result={}
    for stratum in c.STRATA:
        collected={}
        def add(key,value):collected.setdefault(stratum+'/'+key,[]).append(np.asarray(value))
        for i,(base,donor) in enumerate(zip(rows[stratum],donors[stratum],strict=True)):
            check();b=c.forward(model,family,base['ids'],active);d=c.forward(model,family,donor['ids'],active)
            add('logits',b['logits']);add('nll',b['nll'])
            changes=[]
            for site in active:
                count=(1,2,4,8)[c.STRATA.index(stratum)]
                vector=c.intervention_vector(b['sites'][site]['x'],d['sites'][site]['x'],count,
                                            c.seed_for(case,stratum,i,site,entropy),stratum=='adversarial')
                changes.append((site,'input',vector))
                p=c.forward(model,family,base['ids'],active,changes[-1])
                add('effect/'+str(site),p['logits']-b['logits'])
                add('activation/'+str(site),b['sites'][site]['y'].numpy())
            joint=c.forward(model,family,base['ids'],active,changes)
            add('joint_effect',joint['logits']-b['logits'])
        result.update({k:np.stack(v) for k,v in collected.items()})
    return result


def restore(model,family,rows,prediction,active,check):
    result={}
    for stratum in c.STRATA:
        for site in active:
            values=[]
            for i,row in enumerate(rows[stratum]):
                check();h=torch.from_numpy(prediction[stratum+'/activation/'+str(site)][i])
                values.append(c.forward(model,family,row['ids'],active,(site,'output',h))['logits'])
            result[stratum+'/'+str(site)]=np.stack(values)
    return result


def run(manifest_path,role):
    if role not in ('primary','replica'):
        raise ValueError('role')
    manifest=checkpoint_manifest(manifest_path);rt=runtime();root=BASE/role
    if role=='replica':
        pr=read(BASE/'primary/runtime.json')
        if pr['prefix']==rt['prefix'] or pr['site_packages']==rt['site_packages']:
            raise ValueError('replica environment not distinct')
        if {k:v for k,v in pr.items() if k not in ('prefix','site_packages')}!={k:v for k,v in rt.items() if k not in ('prefix','site_packages')}:
            raise ValueError('replica toolchain differs')
    root.mkdir(exist_ok=False)
    start=time.monotonic();check=lambda:budget(start,root)
    try:
        write(root/'runtime.json',rt)
        history=read(BASE/'intake.json')['exclusion'];seen_text=set(history['text_hashes']);seen_tokens=set(history['token_hashes'])
        prepared={};bindings={'packet_sha256':verify_source(),'intake_sha256':sha(BASE/'intake.json'),'checkpoint_manifest_sha256':sha(manifest_path),'families':{}}
        for entry in sorted(manifest['checkpoints'],key=lambda e:c.FAMILIES.index(e['family'])):
            family=entry['family'];model,tokenizer=load_model(entry)
            visible=f.records(family,36100,'visible',64);encoded=f.encode(tokenizer,visible,seen_text,seen_tokens)
            write(root/(family+'-visible.json'),{'records':visible,'encoded':encoded})
            qualification=qualify(model,family,encoded,check)
            prepared[family]={'model':model,'tokenizer':tokenizer,'qualification':qualification,'cases':{}}
            bindings['families'][family]={'qualification':qualification,'visible_sha256':c.digest(encoded),'cases':{}}
            print(json.dumps({'event':'target_qualification','family':family,'passed':qualification['passed'],'sites':qualification['sites']}),flush=True)
        if not all(p['qualification']['passed'] for p in prepared.values()):
            outcome='NO_QUALIFIED_TARGET';metrics={};outcome_bindings={};challenge_generated=False
        else:
            reserve_space()
            for family,item in prepared.items():
                model=item['model'];sites=item['qualification']['sites']
                originals={s:c.original_mlp(model,family,s).eval() for s in sites}
                for seed in c.SEEDS:
                    case=family+'-'+str(seed);case_root=root/case;case_root.mkdir()
                    corpus={key:f.records(family,seed,key,128) for key in ('adapt','donor')}
                    encoded={key:f.encode(item['tokenizer'],v,seen_text,seen_tokens) for key,v in corpus.items()}
                    write(case_root/'adaptation.json',{'corpus':corpus,'encoded':encoded})
                    case_binding={'fixture_sha256':c.digest(encoded),'arms':{}};stages={}
                    for method in c.METHODS:
                        current={};stages[method]={};case_binding['arms'][method]={}
                        for index,site in enumerate(sites):
                            captures={key:[] for key in encoded}
                            with c.installed(model,family,current):
                                for key,rows in encoded.items():
                                    for row in rows:
                                        check();captures[key].append(c.forward(model,family,row['ids'],[site])['sites'][site]['x'])
                            x,d=torch.stack(captures['adapt']),torch.stack(captures['donor'])
                            module,trace=c.fit(originals[site],x,d,family,c.seed_for(family,seed,index,'candidate'),method,check)
                            current[site]=module;stage='block_'+str(index+1);stages[method][stage]=copy.deepcopy(current)
                            capture_hash=save(case_root/(method+'-'+stage+'-adaptation-inputs.npz'),{'x':x.numpy(),'donor':d.numpy()})
                            weights={str(s)+'/'+k:v for s,m in current.items() for k,v in c.state(m).items()}
                            wh=save(case_root/(method+'-'+stage+'-weights.npz'),weights)
                            th=save(case_root/(method+'-'+stage+'-trace.npz'),{'readouts':trace})
                            with c.installed(model,family,current):
                                removed={id(p) for s in current for p in originals[s].parameters()}
                                if removed & {id(p) for p in model.parameters()}:
                                    raise ValueError('original selected MLP parameter remains installed')
                            case_binding['arms'][method][stage]={'weights_sha256':wh,'trace_sha256':th,'capture_sha256':capture_hash,'active_sites':list(current),'original_target_digests':{str(s):c.array_digest(c.state(originals[s])) for s in current}}
                        print(json.dumps({'event':'candidates_frozen','case':case,'method':method}),flush=True)
                    item['cases'][case]={'seed':seed,'stages':stages}
                    bindings['families'][family]['cases'][case]=case_binding
            write(root/'candidate-freeze.json',bindings)
            if role=='replica':
                if bindings!=read(BASE/'primary/candidate-freeze.json'):
                    raise ValueError('replica candidates differ before entropy reveal')
                entropy=read(BASE/'primary/challenge.json')['entropy']
            else:entropy=secrets.token_hex(32)
            queries={};query_binding={}
            for family,item in prepared.items():
                for case,value in item['cases'].items():
                    rows={s:f.records(family,value['seed'],s,32,entropy) for s in c.STRATA}
                    donor_rows={s:f.records(family,value['seed']+1000,s,32,entropy) for s in c.STRATA}
                    encoded={s:f.encode(item['tokenizer'],rows[s],seen_text,seen_tokens) for s in c.STRATA}
                    donors={s:f.encode(item['tokenizer'],donor_rows[s],seen_text,seen_tokens) for s in c.STRATA}
                    queries[case]=(encoded,donors);query_binding[case]=c.digest([encoded,donors])
                    write(root/case/'challenge-fixtures.json',{'rows':rows,'donors':donor_rows,'encoded':encoded,'encoded_donors':donors})
            write(root/'challenge.json',{'entropy':entropy,'candidate_freeze_sha256':c.digest(bindings),'fixtures':query_binding})
            prediction_binding={}
            for family,item in prepared.items():
                model=item['model'];sites=item['qualification']['sites']
                for case,value in item['cases'].items():
                    rows,donors=queries[case];prediction_binding[case]={}
                    for method,stages in value['stages'].items():
                        prediction_binding[case][method]={}
                        for stage,modules in stages.items():
                            with c.installed(model,family,modules):
                                prediction=predict(model,family,rows,donors,list(modules),entropy,case,check)
                            prediction_binding[case][method][stage]=save(root/case/(method+'-'+stage+'-predictions.npz'),prediction)
                        full=stages['block_2']
                        controls={'zero':{s:c.Zero() for s in sites},'wrong':{sites[0]:full[sites[1]],sites[1]:full[sites[0]]},
                                  'random':{s:c.Replacement(full[s].width,family,c.seed_for(case,s,'random'),full[s].mean,full[s].scale).eval().requires_grad_(False) for s in sites}}
                        for label,modules in controls.items():
                            with c.installed(model,family,modules):prediction=predict(model,family,rows,donors,sites,entropy,case,check)
                            prediction_binding[case][method][label]=save(root/case/(method+'-'+label+'-predictions.npz'),prediction)
                    print(json.dumps({'event':'predictions_sealed','case':case}),flush=True)
            write(root/'prediction-freeze.json',{'bindings':prediction_binding,'challenge_sha256':sha(root/'challenge.json')})
            if role=='replica' and read(root/'prediction-freeze.json')!=read(BASE/'primary/prediction-freeze.json'):
                raise ValueError('replica prediction seal differs before oracle')
            metrics={};outcome_bindings={}
            for family,item in prepared.items():
                model=item['model'];sites=item['qualification']['sites']
                for case,value in item['cases'].items():
                    rows,donors=queries[case];oracle=predict(model,family,rows,donors,sites,entropy,case,check)
                    oracle_hash=save(root/case/'oracle.npz',oracle);metrics[case]={}
                    outcome_bindings[case]={'oracle_sha256':oracle_hash,'rescues':{}}
                    # Actual decoder rerun with the original MLPs reinstalled.
                    # This control is assessed after candidate prediction sealing.
                    noops={s:c.original_mlp(model,family,s).eval() for s in sites}
                    with c.installed(model,family,noops):
                        positive_prediction=predict(model,family,rows,donors,sites,entropy,case,check)
                    positive_rescues=restore(model,family,rows,positive_prediction,sites,check)
                    outcome_bindings[case]['noop_sha256']=save(root/case/'noop.npz',positive_prediction)
                    outcome_bindings[case]['noop_rescues_sha256']=save(root/case/'noop-rescues.npz',positive_rescues)
                    positive=c.score(positive_prediction,oracle,positive_rescues,sites)
                    for method,stages in value['stages'].items():
                        cells={}
                        for stage,modules in stages.items():
                            prediction=arrays(root/case/(method+'-'+stage+'-predictions.npz'))
                            rescues=restore(model,family,rows,prediction,list(modules),check)
                            rescue_hash=save(root/case/(method+'-'+stage+'-rescues.npz'),rescues)
                            outcome_bindings[case]['rescues'][method+'/'+stage]=rescue_hash
                            reference=dict(oracle)
                            if len(modules)==1:
                                for stratum in c.STRATA:
                                    reference[stratum+'/joint_effect']=oracle[stratum+'/effect/'+str(sites[0])]
                            cells[stage]=c.score(prediction,reference,rescues,list(modules))
                        controls={label:arrays(root/case/(method+'-'+label+'-predictions.npz')) for label in ('zero','wrong','random')}
                        sensitivity={}
                        for s in c.STRATA:
                            treatment=cells['block_2'][s]['joint_effect']['median_error']
                            errors=[c.effect_metrics(p[s+'/joint_effect'],oracle[s+'/joint_effect'])['median_error'] for p in controls.values()]
                            sensitivity[s]=treatment is not None and all(e is not None and e>=treatment+.1 for e in errors)
                        passed=all(v['passed'] for ss in cells.values() for v in ss.values()) and all(sensitivity.values()) and all(v['passed'] for v in positive.values())
                        metrics[case][method]={'cells':cells,'positive_controls':positive,'control_sensitivity':sensitivity,'passed':passed}
                    print(json.dumps({'event':'evaluated','case':case,'method_passes':{k:v['passed'] for k,v in metrics[case].items()}}),flush=True)
            outcome='PASS' if all(case['jvp']['passed'] for case in metrics.values()) else 'FAIL';challenge_generated=True
        result={'protocol_id':c.PROTOCOL,'state_slice':c.SLICE,'packet_sha256':verify_source(),'outcome_status':outcome,
                'assurance_level':'INTERNAL_PASS','reviewer_status':'NOT_SUBMITTED','claim_ceiling':c.CLAIM,
                'challenge_generated':challenge_generated,'preparation':bindings,'metrics':metrics,'outcome_bindings':outcome_bindings,'resources':check()}
        result['artifacts']=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file()]
        write(root/'result.json',result)
        if role=='replica':
            primary=read(BASE/'primary/result.json')
            checks={k:primary[k]==result[k] for k in RECONCILIATION_KEYS}
            for role2 in ('primary','replica'):
                for row in read(BASE/role2/'result.json')['artifacts']:
                    p=contained(BASE/role2,row['path'])
                    if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('terminal artifact changed')
            if not all(checks.values()):raise ValueError('local reproduction differs')
            write(root/'reconciliation.json',{'checks':checks,'assurance_level':'REPRODUCED_LOCAL','scientific_outcome':outcome,
                  'same_owner_and_host':True,'primary_result_sha256':sha(BASE/'primary/result.json'),'replica_result_sha256':sha(root/'result.json')})
        return {'outcome_status':outcome,'result_sha256':sha(root/'result.json'),'assurance_level':'REPRODUCED_LOCAL' if role=='replica' else 'INTERNAL_PASS'}
    except BaseException as error:
        try:
            write(root/'error.json',{'protocol_id':c.PROTOCOL,'outcome_status':'INCONCLUSIVE','exception_type':type(error).__name__,
                                   'resources':{'monetary_cost_usd':0,'energy':'NOT_MEASURED'},'retry_same_root_permitted':False})
        except (FileNotFoundError,FileExistsError):
            print(json.dumps({'outcome_status':'INCONCLUSIVE','error_receipt':'CUSTODY_UNAVAILABLE_OR_ALREADY_RECORDED','retry_same_root_permitted':False}),flush=True)
        raise


def status():
    return {'protocol_id':c.PROTOCOL,'source_freeze_present':FREEZE.exists(),
            'primary_volume_mounted':PRIMARY_VOLUME.is_dir() and os.path.ismount(PRIMARY_VOLUME),
            'runtime_volume_mounted':RUNTIME_VOLUME.is_dir() and os.path.ismount(RUNTIME_VOLUME),
            'custody_present':BASE.is_dir(),'claim_ceiling':c.CLAIM,'reviewer_status':'NOT_SUBMITTED'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('status','verify','intake','run'))
    parser.add_argument('--checkpoint-manifest',type=Path)
    parser.add_argument('--role',choices=('primary','replica'),default='primary')
    args=parser.parse_args()
    if args.action=='status':answer=status()
    elif args.action=='verify':answer={'packet_sha256':verify_source()}
    elif args.action=='intake':answer=intake()
    else:
        if args.checkpoint_manifest is None:parser.error('--checkpoint-manifest required')
        answer=run(args.checkpoint_manifest,args.role)
    print(json.dumps(answer,sort_keys=True))
