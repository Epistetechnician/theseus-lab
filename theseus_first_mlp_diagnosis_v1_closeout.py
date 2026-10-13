"""Separate NumPy formulas for the first-MLP diagnostic, with no model calls.

State slices: theseus-lab-sandbox-v1, theseus-v39-first-mlp-causal-diagnosis-v1,
global-experiment-assurance-lifecycle-v1. Same-owner numeric audit only.
"""
import math
import numpy as np


def rms_ratio(prediction,reference):
    p,g=np.asarray(prediction,dtype=np.float64),np.asarray(reference,dtype=np.float64)
    numerator=float(np.sum(np.square(p-g)))
    denominator=max(float(np.sum(np.square(g))),g.size*1e-8)
    return float(np.sqrt(numerator/denominator))


def effect_statistics(prediction,reference):
    p,g=np.asarray(prediction,dtype=np.float64),np.asarray(reference,dtype=np.float64)
    norm=np.sqrt(np.sum(g*g,axis=1));mask=norm>=1e-4
    if not np.any(mask):return {'resolved_fraction':0.,'median_relative_error':None,'p90_relative_error':None,'median_cosine':None}
    p,g=p[mask],g[mask];norm=norm[mask]
    error=np.sqrt(np.sum(np.square(p-g),axis=1))/np.maximum(norm,1e-6)
    cosine=np.sum(p*g,axis=1)/np.maximum(np.sqrt(np.sum(p*p,axis=1))*norm,1e-12)
    return {'resolved_fraction':float(mask.mean()),'median_relative_error':float(np.median(error)),
            'p90_relative_error':float(np.quantile(error,.9)),'median_cosine':float(np.median(cosine))}


def recompute(data,index):
    cells={};transport={}
    for case in index['cases']:
        for method in ('jvp','endpoints'):
            for label in ('first','middle_prefix','final'):
                key=case+'/'+method+'/'+label;stem=key+'/'
                ref=data[stem+'teacher_delta']
                cells[key]={'value_nrmse':rms_ratio(data[stem+'student_value'],data[stem+'teacher_value']),
                    'jacobian_nrmse':rms_ratio(data[stem+'student_jacobian'],data[stem+'teacher_jacobian']),
                    'finite_effects':effect_statistics(data[stem+'student_delta'],ref),
                    'analytic_zero_delta_control':effect_statistics(np.zeros_like(ref),ref),
                    'sampled_prompts':8,'finite_patches':64}
            stem=case+'/'+method+'/transport/'
            a,g,s,h=[np.asarray(data[stem+k],dtype=np.float64) for k in ('actual','guide','student_guide','anchored')]
            error=a-g;parts={'prefix_context':a-s,'factual_anchor':s-h,'local_delta_transport':h-g}
            if np.max(np.abs(parts['prefix_context']+parts['factual_anchor']+parts['local_delta_transport']-error))>1e-12:
                raise ValueError('independent vector accounting failed')
            denominator=float(np.sum(error*error))
            transport[case+'/'+method]={'actual_effects':effect_statistics(a,g),
                'student_in_guide_context':effect_statistics(s,g),'anchored_student_delta':effect_statistics(h,g),
                'component_relative_rms':{name:float(np.sqrt(np.sum(term*term)/max(denominator,1e-12))) for name,term in parts.items()},
                'error_projection_shares':{name:float(np.sum(term*error))/denominator if denominator>=1e-12 else None for name,term in parts.items()},
                'projection_shares_are_ordered_not_unique_causal_attribution':True}
    selected=[cells[case+'/jvp/final'] for case in index['cases']]
    rule=all(row['jacobian_nrmse']>.2 and row['finite_effects']['resolved_fraction']>=.9
             and row['finite_effects']['median_relative_error'] is not None and row['finite_effects']['median_relative_error']>.2 for row in selected)
    return {'cells':cells,'transport':transport,'outcome_status':'LOCAL_DERIVATIVE_MISMATCH_SUPPORTED' if rule else 'MIXED_OR_LOCAL_FIT_REQUIRES_FURTHER_DIAGNOSIS'}


def compare(actual,expected,path='metrics'):
    """Tight numeric tolerance; structure, categorical decisions and counts exact."""
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('numeric audit keys differ: '+path)
        for key in expected:compare(actual[key],expected[key],path+'/'+key)
    elif type(expected) is float:
        if not isinstance(actual,(float,int)) or not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('numeric audit value differs: '+path)
    elif actual!=expected:raise ValueError('numeric audit exact value differs: '+path)
