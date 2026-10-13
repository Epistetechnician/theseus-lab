"""Hermetic V36 engineering only; no pretrained assets or study outcomes.

State slices: theseus-lab-sandbox-v1, theseus-pretrained-whole-block-transfer-v36,
global-experiment-assurance-lifecycle-v1.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from safetensors.torch import save_file

import numpy as np
import torch
from transformers import (OPTConfig, OPTForCausalLM, GPT2Config, GPT2LMHeadModel,
                          LlamaConfig, LlamaForCausalLM)

import replacement_loss_v36_core as c
import replacement_loss_v36_fixtures as f
import replacement_loss_v36_runner as r
import replacement_loss_v36_closeout as audit

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.use_deterministic_algorithms(True)


def tiny(family):
    torch.manual_seed(81)
    if family=='opt_relu':
        model=OPTForCausalLM(OPTConfig(vocab_size=29,hidden_size=16,word_embed_proj_dim=16,
                 ffn_dim=32,num_hidden_layers=4,num_attention_heads=2,dropout=0.,attention_dropout=0.,pad_token_id=0))
    elif family=='gpt2_gelu':
        model=GPT2LMHeadModel(GPT2Config(vocab_size=29,n_embd=16,n_inner=32,n_layer=4,n_head=2,
                                       n_positions=128,resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,bos_token_id=1,eos_token_id=2))
    else:
        model=LlamaForCausalLM(LlamaConfig(vocab_size=29,hidden_size=16,intermediate_size=32,
               num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128))
    model.config._attn_implementation='eager'
    return model.eval().requires_grad_(False)


class Tokenizer:
    def __call__(self,text,**kwargs):
        return {'input_ids':[1]+[int(x)%25+2 for x in hashlib.sha256(text.encode()).digest()[:8]]}


class Engineering(unittest.TestCase):
    def test_full_mlp_adapter_matches_actual_decoder_all_families(self):
        for family in c.FAMILIES:
            model=tiny(family)
            result=c.forward(model,family,list(range(1,10)),[0,2])
            for site in (0,2):
                module=c.original_mlp(model,family,site).eval()
                actual=module(result['sites'][site]['x'])
                torch.testing.assert_close(actual,result['sites'][site]['y'],atol=1e-7,rtol=1e-6)
            self.assertTrue(np.isfinite(result['logits']).all())

    def test_derivatives_match_independent_autograd_all_families(self):
        for family in c.FAMILIES:
            module=c.original_mlp(tiny(family),family,0).eval()
            x=torch.randn(5,16);directions=torch.randn(3,5,16)
            actual=c.jvp(module,x,directions)
            expected=torch.stack([torch.autograd.functional.jvp(module,x,d)[1] for d in directions],1)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)

    def test_derivative_loss_has_parameter_gradients(self):
        model=c.Replacement(16,'gpt2_gelu',5)
        derivative=c.jvp(model,torch.randn(4,16),torch.randn(2,4,16))
        derivative.square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0)

    def test_original_graph_absent_and_restored_after_exception(self):
        for family in c.FAMILIES:
            model=tiny(family);before=c.array_digest(c.state(model))
            original=c.original_mlp(model,family,0).eval()
            old={id(p) for p in original.parameters()}
            with self.assertRaisesRegex(RuntimeError,'fixture'):
                with c.installed(model,family,{0:c.Replacement(16,family,5)}):
                    self.assertFalse(old & {id(p) for p in model.parameters()})
                    raise RuntimeError('fixture')
            self.assertEqual(before,c.array_digest(c.state(model)))

    def test_input_output_joint_interventions_and_hook_cleanup(self):
        for family in c.FAMILIES:
            model=tiny(family);ids=list(range(1,10));p=c.forward(model,family,ids,[0,2])
            noop=[(s,'input',p['sites'][s]['x']) for s in (0,2)]
            q=c.forward(model,family,ids,[0,2],noop)
            np.testing.assert_array_equal(p['logits'],q['logits'])
            changed=c.forward(model,family,ids,[0,2],[(s,'output',torch.zeros(16)) for s in (0,2)])
            self.assertFalse(np.array_equal(p['logits'],changed['logits']))
            self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))

    def test_probe_supervision_caps_repeatability_and_teacher_preservation(self):
        teacher=c.original_mlp(tiny('opt_relu'),'opt_relu',0).eval()
        before=c.array_digest(c.state(teacher));x=torch.randn(12,16);donor=-x
        for method in c.METHODS:
            a,traces=c.fit(teacher,x,donor,'opt_relu',91,method,updates=3)
            b,second=c.fit(teacher,x,donor,'opt_relu',91,method,updates=3)
            self.assertEqual(c.array_digest(c.state(a)),c.array_digest(c.state(b)))
            np.testing.assert_array_equal(traces,second)
            self.assertTrue(np.all(traces[:,1]*traces[:,2]<=.5*traces[:,0]+1e-8))
            self.assertFalse({id(p) for p in teacher.parameters()} & {id(p) for p in a.parameters()})
        self.assertEqual(before,c.array_digest(c.state(teacher)))

    def test_invalid_supervision_and_unfrozen_teacher_rejected(self):
        teacher=c.Replacement(8,'opt_relu',1);x=torch.zeros(2,8)
        with self.assertRaises(ValueError):c.fit(teacher,x,x,'opt_relu',1,'jvp',updates=1)
        teacher.eval().requires_grad_(False)
        with self.assertRaises(ValueError):c.fit(teacher,x,x,'opt_relu',1,'unknown',updates=1)

    def test_independent_effect_resolution_and_direction(self):
        reference=np.tile([1.,0.],(10,1))
        self.assertTrue(c.effect_metrics(reference,reference)['passed'])
        self.assertFalse(c.effect_metrics(-reference,reference)['passed'])
        self.assertFalse(c.effect_metrics(reference,np.zeros_like(reference))['passed'])

    def test_fixture_partitions_disjoint_and_token_collision_rejected(self):
        seen_text,seen_tokens=set(),set()
        for family in c.FAMILIES:
            for seed in c.SEEDS:
                for partition in ('visible','adapt','donor',*c.STRATA):
                    rows=f.records(family,seed,partition,3)
                    f.encode(Tokenizer(),rows,seen_text,seen_tokens)
        with self.assertRaises(ValueError):f.encode(Tokenizer(),rows,seen_text,seen_tokens)
        class Collision:
            def __call__(self,text,**kwargs):return {'input_ids':[1,2]}
        with self.assertRaises(ValueError):f.encode(Collision(),f.records('opt_relu',5,'adapt',2),set(),set())

    def test_missing_mount_cannot_create_fake_custody(self):
        with tempfile.TemporaryDirectory() as folder:
            volume=Path(folder)/'unmounted';base=volume/'ResearchArtifacts/study'
            with patch.object(r,'PRIMARY_VOLUME',volume),patch.object(r,'BASE',base),patch('os.path.ismount',return_value=False):
                with self.assertRaises(FileNotFoundError):r.intake()
                with self.assertRaises(FileNotFoundError):r.write(base/'receipt.json',{})
            self.assertFalse(volume.exists())

    def test_path_escape_and_repo_assets_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(ValueError):r.contained(root,'../escape')
            with self.assertRaises(ValueError):r.contained(root,str(root/'absolute'))

    def test_history_snapshot_extracts_identity_without_raw_text(self):
        sets={k:set() for k in ('model_ids','text_hashes','token_hashes','prior_digests')}
        r.collect_history({'model_id':'SOURCE/Model','input_text':'fixture','ids':[1,2,3],'sha256':'f'*64},sets)
        self.assertEqual(sets['model_ids'],{'source/model'})
        self.assertNotIn('fixture',sets['text_hashes'])
        self.assertEqual(sets['token_hashes'],{c.digest([1,2,3])})

    def test_terminal_root_reuse_preserves_files(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);(base/'primary').mkdir();marker=base/'primary/marker';marker.write_text('preserve')
            with patch.object(r,'BASE',base),patch.object(r,'checkpoint_manifest',return_value={}),patch.object(r,'runtime',return_value={}):
                with self.assertRaises(FileExistsError):r.run('unused','primary')
            self.assertEqual(marker.read_text(),'preserve')

    def test_crash_receipt_and_no_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);r.write(base/'intake.json',{'exclusion':{'text_hashes':[],'token_hashes':[]}})
            manifest={'checkpoints':[{'family':'opt_relu'}]}
            r.write(base/'models.json',manifest)
            with patch.object(r,'BASE',base),patch.object(r,'mount_guard'),patch.object(r,'verify_source',return_value='frozen'),patch.object(r,'checkpoint_manifest',return_value=manifest),patch.object(r,'runtime',return_value={}),patch.object(r,'load_model',side_effect=RuntimeError('fixture')):
                with self.assertRaises(RuntimeError):r.run(base/'models.json','primary')
                self.assertEqual(r.read(base/'primary/error.json')['outcome_status'],'INCONCLUSIVE')
                with self.assertRaises(FileExistsError):r.run(base/'models.json','primary')

    def test_entire_small_decoder_packet_primary_and_replica(self):
        # This is a deliberately reduced, random-model engineering packet.
        # It does not open the actual study's model/custody/runtime gates.
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            r.write(base/'intake.json',{'exclusion':{'text_hashes':[],'token_hashes':[]}})
            manifest_path=base/'models.json'
            r.write(manifest_path,{'checkpoints':[{'family':family} for family in c.FAMILIES]})
            real_records=f.records;real_fit=c.fit
            short_records=lambda family,seed,partition,count,entropy='preparation':real_records(family,seed,partition,2,entropy)
            short_fit=lambda *args,**kwargs:real_fit(*args,**kwargs,updates=2)
            runtime=lambda:{'prefix':str(base/('env1' if not (base/'primary/result.json').exists() else 'env2')),'site_packages':str(base/('lib1' if not (base/'primary/result.json').exists() else 'lib2')),'python':'fixture'}
            with contextlib.ExitStack() as stack:
                for target,name,value in ((r,'BASE',base),(r,'mount_guard',lambda:None),(r,'verify_source',lambda:'frozen'),
                   (r,'checkpoint_manifest',lambda path:r.read(path)),(r,'runtime',runtime),
                   (r,'load_model',lambda entry:(tiny(entry['family']),Tokenizer())),(r,'reserve_space',lambda:None),
                   (r,'qualify',lambda *args:{'sites':[0,2],'passed':True,'removal':{},'visible_energy':{}}),
                   (c,'SEEDS',(36101,)),(c,'fit',short_fit),(f,'records',short_records)):
                    stack.enter_context(patch.object(target,name,value))
                with contextlib.redirect_stdout(io.StringIO()):
                    primary=r.run(manifest_path,'primary');replica=r.run(manifest_path,'replica')
            self.assertEqual(primary['outcome_status'],replica['outcome_status'])
            self.assertTrue(all(r.read(base/'replica/reconciliation.json')['checks'].values()))
            self.assertTrue((base/'primary/prediction-freeze.json').exists())
            self.assertTrue((base/'primary/candidate-freeze.json').exists())
            self.assertEqual(len(r.read(base/'primary/result.json')['metrics']),3)
            with patch.object(c,'SEEDS',(36101,)):
                audited=audit.verify_run(base/'primary')
            self.assertEqual(audited['native_cells'],48)
            self.assertEqual(audited['gradient_readouts'],24)
            # Result receipts are not their own evidence for controls/verdicts.
            path=base/'primary/result.json';original=path.read_text()
            with patch.object(c,'SEEDS',(36101,)):
                altered=json.loads(original);first=next(iter(altered['metrics'].values()))['jvp']
                first['control_sensitivity']['ordinary']=not first['control_sensitivity']['ordinary']
                path.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError,'control or method verdict'):audit.verify_run(base/'primary')
                altered=json.loads(original);altered['outcome_status']='PASS' if primary['outcome_status']!='PASS' else 'FAIL'
                path.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError,'aggregate verdict'):audit.verify_run(base/'primary')
                path.write_text(original)
                seal_path=base/'primary/prediction-freeze.json';original_seal=seal_path.read_text()
                altered_seal=json.loads(original_seal)
                next(iter(altered_seal['bindings'].values()))['jvp']['block_1']='0'*64
                seal_path.write_text(json.dumps(altered_seal))
                altered=json.loads(original)
                for row in altered['artifacts']:
                    if row['path']=='prediction-freeze.json':row.update(bytes=seal_path.stat().st_size,sha256=r.sha(seal_path))
                path.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError,'prediction binding'):audit.verify_run(base/'primary')
                path.write_text(original);seal_path.write_text(original_seal)
                reconciliation=base/'replica/reconciliation.json';receipt=r.read(reconciliation)
                receipt['checks'].pop('metrics');reconciliation.write_text(json.dumps(receipt))
                with patch.object(r,'BASE',base),patch.object(r,'verify_intake'):
                    with self.assertRaisesRegex(ValueError,'reconciliation keys incomplete'):audit.validate()

    def test_independent_numeric_score_detects_corruption(self):
        reference=np.tile([1.,0.],(8,1))
        self.assertEqual(audit.relative(reference,reference),c.effect_metrics(reference,reference))
        self.assertFalse(audit.relative(-reference,reference)['passed'])

    def test_invalid_intervention_kind_and_site_rejected(self):
        model=tiny('opt_relu')
        for intervention in ((8,'input',torch.zeros(16)),(0,'invalid',torch.zeros(16))):
            with self.assertRaises(ValueError):c.forward(model,'opt_relu',[1,2,3],[0],intervention)

    def test_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'freeze.json'
            r.write(path,{'members':[],'packet_sha256':'altered'})
            with patch.object(r,'FREEZE',path),patch.object(r,'freeze_material',return_value=[]):
                with self.assertRaises(ValueError):r.verify_source()

    def test_manifest_freshness_attestations_and_completeness(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);history={'packet_sha256':'frozen','exclusion':{'model_ids':[],'prior_digests':[]}}
            r.write(base/'intake.json',history);entries=[]
            for i,family in enumerate(c.FAMILIES):
                directory=base/family;directory.mkdir()
                save_file({'weights':torch.tensor([i+1.])},directory/'model.safetensors')
                r.write(directory/'config.json',{'fixture':True})
                entries.append({'family':family,'model_id':'fixture/'+family,'revision':'a'*40,'local_path':family,
                  'base_model':True,'selected_after_source_freeze':True,'not_used_for_protocol_design':True,'rights_to_load_locally':True,
                  'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':r.sha(p)} for p in sorted(directory.iterdir())]})
            payload={'packet_sha256':'frozen','intake_sha256':r.sha(base/'intake.json'),'checkpoints':entries}
            path=base/'models.json'
            with patch.object(r,'BASE',base),patch.object(r,'verify_intake',return_value=history):
                def check(value):
                    path.write_text(json.dumps(value));return r.checkpoint_manifest(path)
                self.assertEqual(len(check(payload)['checkpoints']),3)
                changed=json.loads(json.dumps(payload));changed['checkpoints'][0]['rights_to_load_locally']=False
                with self.assertRaises(ValueError):check(changed)
                changed=json.loads(json.dumps(payload));changed['checkpoints'][0]['files'].append(changed['checkpoints'][0]['files'][0])
                with self.assertRaises(ValueError):check(changed)
                history['exclusion']['model_ids']=[entries[0]['model_id']]
                with self.assertRaises(ValueError):check(payload)
                history['exclusion']['model_ids']=[]
                history['exclusion']['prior_digests']=[entries[0]['files'][1]['sha256']]
                with self.assertRaises(ValueError):check(payload)
                history['exclusion']['prior_digests']=[]
                (base/c.FAMILIES[0]/'unmanifested.txt').write_text('fixture')
                with self.assertRaises(ValueError):check(payload)

    def test_study_runtime_cannot_use_qualification_temp_prefix(self):
        # Correct packages alone do not satisfy the external custody gate.
        with self.assertRaises(ValueError):r.runtime()

    def test_artifact_capacity_preflight(self):
        with patch.object(r.shutil,'disk_usage') as usage:
            usage.return_value.free=r.MAX_BYTES
            with self.assertRaisesRegex(RuntimeError,'free-space reserve'):r.reserve_space()
            usage.return_value.free=r.MAX_BYTES+1024**3
            r.reserve_space()


if __name__=='__main__':
    unittest.main()
