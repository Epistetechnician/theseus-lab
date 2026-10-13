"""Hermetic fitting-adapter contracts for the V39 state slice."""
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
import replacement_loss_v39_runner as v


def masked(rows):
    return np.stack([v.padded(torch.tensor(row,dtype=torch.float32)).numpy() for row in rows])


class Coverage(unittest.TestCase):
    def test_every_adaptation_token_and_both_donor_endpoints(self):
        x=masked([[[1,2],[3,4],[5,6]],[[7,8],[9,10]]])
        d=masked([[[11,12],[13,14]],[[15,16],[17,18],[19,20],[21,22]]])
        before=x.copy();result=v.pool(x,d)
        np.testing.assert_array_equal(result['x'],[[1,2],[3,4],[5,6],[7,8],[9,10]])
        np.testing.assert_array_equal(result['donor'],[[11,12],[11,12],[13,14],[15,16],[21,22]])
        np.testing.assert_array_equal(result['lengths'],[[3,2],[2,4]])
        np.testing.assert_array_equal(x,before)

    def test_invalid_captures_rejected(self):
        good=masked([[[1,2],[3,4]]])
        for field,value in (([0,0,-1],.5),([0,0,-1],0),([0,2,0],2),([0,0,0],np.nan)):
            bad=good.copy();bad[tuple(field)]=value
            with self.assertRaises(ValueError):v.pool(bad,good)
        with self.assertRaises(ValueError):v.pool(good[:,0,:],good[:,0,:])

    def test_marker_preserves_ids_and_requires_fitting_role(self):
        ids=v.FitTokenIDs([2,3,4],'adapt')
        self.assertEqual(json.dumps(ids),'[2, 3, 4]')
        with self.assertRaises(ValueError):v.FitTokenIDs([2,3],'ordinary')
        rows=v.records('gpt2_gelu',1,'ordinary',1)
        self.assertNotIn('fit_partition',rows[0])
        rows=v.records('gpt2_gelu',1,'adapt',1)
        self.assertEqual(rows[0]['fit_partition'],'adapt')

    def test_native_evaluator_passes_through_unchanged(self):
        token_ids=[1,2];sentinel=object()
        with patch.object(v,'ORIGINAL_FORWARD',return_value=sentinel) as original:
            self.assertIs(v.forward('model','family',token_ids,[0],('intervention',),True),sentinel)
            original.assert_called_once_with('model','family',token_ids,[0],('intervention',),True)
        with self.assertRaises(ValueError):v.forward(None,'family',v.FitTokenIDs([1,2],'donor'),[0],('intervention',))

    def test_fitting_capture_masks_are_not_model_features(self):
        native={'sites':{0:{'x':torch.ones(3,8),'y':torch.ones(3,8)}},'nll':0.,'logits':np.zeros(2)}
        with patch.object(v,'ORIGINAL_FORWARD',return_value=native) as original:
            result=v.forward(None,'family',v.FitTokenIDs([1,2,3],'adapt'),[0])
            self.assertEqual(result['sites'][0]['x'].shape,(96,9))
            self.assertTrue(original.call_args.kwargs['capture_all'])
        x=torch.from_numpy(masked([[[1,2],[3,4]]]))
        with patch.object(v,'ORIGINAL_FIT',return_value=('student','trace')) as original:
            self.assertEqual(v.fit('teacher',x,x,'gpt2_gelu',1,'jvp'),('student','trace'))
            self.assertEqual(tuple(original.call_args.args[1].shape),(2,2))
            self.assertEqual(tuple(original.call_args.args[2].shape),(2,2))

    def test_capture_save_binds_actual_pooled_bytes(self):
        from pathlib import Path
        x=masked([[[1,2],[3,4]]]);value={'x':x,'donor':x}
        with patch.object(v,'ORIGINAL_SAVE',side_effect=lambda p,a:v.c.array_digest(a)) as original:
            actual=v.save(Path('jvp-block_1-adaptation-inputs.npz'),value)
            self.assertEqual(actual,v.c.array_digest(v.pool(x,x)))
            self.assertEqual(original.call_args.args[1]['x'].shape,(2,2))

    def test_fixed_scope_and_existing_loss_gates(self):
        self.assertEqual(v.SITES,(0,2));self.assertEqual(v.c.FAMILIES,('gpt2_gelu',))
        self.assertEqual(v.c.UPDATES,1024);self.assertEqual(v.c.METHODS,('jvp','endpoints'))
        self.assertEqual(v.c.SEEDS,(36101,36102,36103));self.assertEqual(v.r.MAX_SECONDS,28800)
        good={'0':{'removal_logit_nrmse':.1,'removal_top1_change':.2},'2':{'removal_logit_nrmse':.1,'removal_top1_change':.2}}
        self.assertTrue(v.previous.qualification(good));good['2']['removal_top1_change']=.199
        self.assertFalse(v.previous.qualification(good))

    def test_held_out_encoding_is_never_tagged(self):
        tokenizer=lambda text,**kwargs:{'input_ids':[1,2,3]}
        ordinary=[{'id':'o','input_text':'synthetic ordinary'}]
        adapt=[{'id':'a','input_text':'synthetic adaptation','fit_partition':'adapt'}]
        plain=v.encode(tokenizer,ordinary,set(),set())[0]
        marked=v.encode(tokenizer,adapt,set(),set())[0]
        self.assertIs(type(plain['ids']),list)
        self.assertIsInstance(marked['ids'],v.FitTokenIDs)
        self.assertEqual(marked['ids'],plain['ids'])

    def test_real_small_jvp_fit_with_independent_parameters(self):
        teacher=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.GELU(),torch.nn.Linear(8,4)).eval().requires_grad_(False)
        gen=torch.Generator().manual_seed(391)
        source=[torch.randn(n,4,generator=gen) for n in (2,4)]
        target=[torch.randn(n,4,generator=gen) for n in (3,2)]
        x=torch.stack([v.padded(t) for t in source]);d=torch.stack([v.padded(t) for t in target])
        student,trace=v.fit(teacher,x,d,'gpt2_gelu',391,'jvp',updates=2)
        self.assertEqual(trace.shape,(2,3));self.assertEqual(student.width,4)
        self.assertTrue(np.isfinite(trace).all())
        self.assertFalse({id(p) for p in student.parameters()} & {id(p) for p in teacher.parameters()})


if __name__=='__main__':unittest.main()
