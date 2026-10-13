"""V36R2 prospective catalog compatibility continuation; no loss/gate changes.

State slices: theseus-lab-sandbox-v1,
theseus-pretrained-whole-block-transfer-v36r2,
theseus-public-checkpoint-acquisition-v36r2,
global-experiment-assurance-lifecycle-v1.
"""
import argparse
import json
import sys

import huggingface_hub
from huggingface_hub import HfApi as OriginalHfApi
import replacement_loss_v36r1_runner as previous

c,r=previous.c,previous.r
c.PROTOCOL='theseus-pretrained-whole-block-transfer-v36r2'
c.SLICE=c.PROTOCOL
r.BASE=r.ARTIFACTS/'theseus-lab-sandbox-v1-pretrained-whole-block-transfer-v36r2'
r.FREEZE=r.REPO/'docs/replacement-loss-v36r2-freeze.json'
r.MEMBERS=(*r.MEMBERS,'docs/replacement-loss-v36r1-freeze.json',
           'docs/replacement-loss-v36r1-results.json','docs/replacement-loss-v36r1-catalog-contract-audit.json',
           'replacement_loss_v36r2_runner.py','tests/test_replacement_loss_v36r2.py',
           'docs/replacement-loss-v36r2-catalog-continuation.md')
previous.RUNTIMES=r.RUNTIME_VOLUME/'ResearchRuntimes/theseus-lab-v36r2/toolchains'


class LocalParameterCatalog(OriginalHfApi):
    """Preserve the local 250M check while dropping unreliable server filtering."""
    def list_models(self,**kwargs):
        if kwargs.pop('num_parameters',None)!='max:250M':
            raise ValueError('unexpected catalog parameter contract')
        return super().list_models(**kwargs)


huggingface_hub.HfApi=LocalParameterCatalog


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','status','intake','bootstrap','runtime-check','acquire','run','closeout'))
    parser.add_argument('--execute-public-acquisition',action='store_true')
    parser.add_argument('--role',choices=('primary','replica'),default='primary')
    args=parser.parse_args()
    try:
        if args.action=='verify':answer={'packet_sha256':r.verify_source()}
        elif args.action=='status':answer=r.status()
        elif args.action=='intake':answer=previous.intake()
        elif args.action=='bootstrap':answer=previous.bootstrap()
        elif args.action=='runtime-check':answer=r.runtime()
        elif args.action=='acquire':answer=previous.acquire(args.execute_public_acquisition)
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
