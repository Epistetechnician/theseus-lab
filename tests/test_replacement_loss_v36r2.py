"""V36R2 catalog contract; no network/model evaluation.

State slices: theseus-lab-sandbox-v1, theseus-pretrained-whole-block-transfer-v36r2,
theseus-public-checkpoint-acquisition-v36r2, global-experiment-assurance-lifecycle-v1.
"""
import unittest
from unittest.mock import patch

import replacement_loss_v36r2_runner as current


class Catalog(unittest.TestCase):
    def test_server_filter_removed_and_other_contract_fields_preserved(self):
        with patch.object(current.OriginalHfApi,'list_models',return_value=[]) as method:
            current.LocalParameterCatalog(token=False).list_models(search='gpt2',num_parameters='max:250M',
                         limit=200,sort='created_at',token=False)
            self.assertNotIn('num_parameters',method.call_args.kwargs)
            self.assertEqual(method.call_args.kwargs['limit'],200)
            self.assertEqual(method.call_args.kwargs['search'],'gpt2')
            self.assertIs(method.call_args.kwargs['token'],False)

    def test_unexpected_parameter_contract_rejected(self):
        with patch.object(current.OriginalHfApi,'list_models') as method:
            with self.assertRaises(ValueError):
                current.LocalParameterCatalog(token=False).list_models(num_parameters='max:1B')
            method.assert_not_called()


if __name__=='__main__':unittest.main()
