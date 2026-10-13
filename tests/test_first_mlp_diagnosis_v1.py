"""Hermetic contracts for theseus-v39-first-mlp-causal-diagnosis-v1.

Also touches theseus-lab-sandbox-v1 and global-experiment-assurance-lifecycle-v1.
"""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import theseus_first_mlp_diagnosis_v1 as v


class Diagnosis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_selection_is_fixed_sorted_and_data_independent(self):
        a=v.selection('gpt2_gelu-36101');self.assertEqual(a,v.selection('gpt2_gelu-36101'))
        self.assertEqual(a,sorted(set(a)));self.assertEqual(len(a),8)
        self.assertNotEqual(a,v.selection('gpt2_gelu-36102'))
        with self.assertRaises(ValueError):v.selection('case',7)

    def test_token_positions_and_patch_cardinality(self):
        self.assertEqual([v.position(30,k) for k in v.POSITIONS],[0,14,29])
        x=torch.zeros(12);d=torch.ones(12)
        p=v.patched(x,d,4,1.4,('case',3,'final'))
        self.assertEqual(int((p!=0).sum()),4)
        self.assertTrue(torch.all(p[p!=0]==1.4));self.assertTrue(torch.all(x==0))
        self.assertTrue(torch.equal(p,v.patched(x,d,4,1.4,('case',3,'final'))))
        with self.assertRaises(ValueError):v.position(2,'final')

    def test_jacobian_orientation_and_independent_jvp_check(self):
        module=nn.Linear(3,3,bias=False).eval().requires_grad_(False)
        with torch.no_grad():module.weight.copy_(torch.tensor([[1.,2.,3.],[0.,4.,2.],[1.,0.,5.]]))
        x=torch.tensor([.1,.2,.3]);j=v.jacobian(module,x,('test',))
        np.testing.assert_array_equal(j,module.weight.numpy())
        with patch.object(torch.func,'jacrev',return_value=lambda z:module.weight.T):
            with self.assertRaisesRegex(ValueError,'orientation'):v.jacobian(module,x,('test',))

    def test_jacobian_agrees_with_central_finite_differences(self):
        with torch.random.fork_rng():
            torch.manual_seed(401);module=nn.Sequential(nn.Linear(4,7),nn.GELU(),nn.Linear(7,4)).eval().requires_grad_(False)
        x=torch.tensor([.2,-.5,.1,.9]);j=v.jacobian(module,x,('finite_difference',))
        columns=[]
        with torch.no_grad():
            for i in range(4):
                e=torch.zeros(4);e[i]=.001;columns.append(((module(x+e)-module(x-e))/.002).numpy())
        np.testing.assert_allclose(j,np.stack(columns,axis=1),rtol=.005,atol=1e-4)

    def test_finite_metrics_noop_zero_and_unresolved(self):
        b=np.array([[1.,2.],[3.,4.],[5.,6.]])
        self.assertEqual(v.effects(b,b)['median_relative_error'],0.)
        self.assertEqual(v.effects(np.zeros_like(b),b)['median_relative_error'],1.)
        self.assertIsNone(v.effects(b,np.zeros_like(b))['median_relative_error'])
        self.assertEqual(v.normalized([v.sums(b,b)]),0.)
        with self.assertRaises(ValueError):v.sums(b,np.array([np.nan]))

    def test_ordered_vector_accounting_and_cancellation(self):
        g=np.array([[1.,2.]]);h=np.array([[0.,4.]]);s=np.array([[3.,1.]]);a=np.array([[2.,3.]])
        total,parts=v.components(a,g,s,h)
        np.testing.assert_array_equal(sum(parts.values()),a-g)
        self.assertFalse(np.isclose(sum(np.linalg.norm(x) for x in parts.values()),np.linalg.norm(total)))
        with self.assertRaises(ValueError):v.components(a,g,s,np.array([[np.nan,0.]]))

    def test_real_decoder_native_input_and_output_interventions(self):
        from transformers import GPT2Config,GPT2LMHeadModel
        with torch.random.fork_rng():
            torch.manual_seed(402)
            cfg=GPT2Config(n_embd=8,n_layer=4,n_head=2,n_positions=32,vocab_size=16,
                           resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,bos_token_id=0,eos_token_id=0)
            cfg._attn_implementation='eager';model=GPT2LMHeadModel(cfg).eval().requires_grad_(False)
        teacher=v.c.original_mlp(model,'gpt2_gelu',0);student=v.c.Replacement(8,'gpt2_gelu',403).eval().requires_grad_(False)
        ids=[1,2,3,4];donor=[4,3,2,1]
        base=v.c.forward(model,'gpt2_gelu',ids,[0]);d=v.c.forward(model,'gpt2_gelu',donor,[0])
        x=base['sites'][0]['x'];xp=v.patched(x,d['sites'][0]['x'],2,1.4,('toy',))
        oracle=v.c.forward(model,'gpt2_gelu',ids,[0],(0,'input',xp))
        with v.c.installed(model,'gpt2_gelu',{0:student}):
            a=v.c.forward(model,'gpt2_gelu',ids,[0]);p=v.c.forward(model,'gpt2_gelu',ids,[0],(0,'input',xp))
        self.assertTrue(torch.equal(a['sites'][0]['x'],x));self.assertIs(v.c.original_mlp(model,'gpt2_gelu',0),teacher)
        sb=v.c.forward(model,'gpt2_gelu',ids,[0],(0,'output',a['sites'][0]['y']))
        sp=v.c.forward(model,'gpt2_gelu',ids,[0],(0,'output',p['sites'][0]['y']))
        anchor=v.c.forward(model,'gpt2_gelu',ids,[0],(0,'output',base['sites'][0]['y']+p['sites'][0]['y']-a['sites'][0]['y']))
        cast=lambda z:z['logits'].astype(np.float64)
        total,parts=v.components(cast(p)-cast(a),cast(oracle)-cast(base),cast(sp)-cast(sb),cast(anchor)-cast(base))
        np.testing.assert_allclose(sum(parts.values()),total,rtol=0,atol=1e-12)
        with v.c.installed(model,'gpt2_gelu',{0:teacher}):noop=v.c.forward(model,'gpt2_gelu',ids,[0],(0,'input',xp))
        np.testing.assert_array_equal(noop['logits'],oracle['logits'])

    def test_separate_numeric_formulas_agree_and_reject_false_metrics(self):
        import theseus_first_mlp_diagnosis_v1_closeout as audit
        rng=np.random.default_rng(404);data={};index={'cases':{str(i):{} for i in range(3)}}
        for case in index['cases']:
            for method in ('jvp','endpoints'):
                for label in v.POSITIONS:
                    prefix=case+'/'+method+'/'+label+'/'
                    for key,shape in (('value',(8,4)),('jacobian',(8,4,4)),('delta',(64,4))):
                        reference=rng.normal(size=shape).astype(np.float32)
                        data[prefix+'teacher_'+key]=reference;data[prefix+'student_'+key]=reference*1.5
                prefix=case+'/'+method+'/transport/'
                for key in ('actual','guide','student_guide','anchored'):data[prefix+key]=rng.normal(size=(64,6))
        reference=v.summarize(data,index);audit.compare(audit.recompute(data,index),reference)
        reference['cells']['0/jvp/final']['jacobian_nrmse']+=.01
        with self.assertRaises(ValueError):audit.compare(audit.recompute(data,index),reference)

    def test_independent_numeric_comparator_checks_exact_structure(self):
        import theseus_first_mlp_diagnosis_v1_closeout as audit
        for bad in ({'a':1.,'extra':0},{'a':float('nan')},{'a':1.01}):
            with self.assertRaises(ValueError):audit.compare(bad,{'a':1.})


if __name__=='__main__':unittest.main()
