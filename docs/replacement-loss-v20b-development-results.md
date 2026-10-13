# Replacement-loss V20B development results

State slices: `theseus-lab-sandbox-v1`, `theseus-replacement-loss-v20b-low-rank-causal-residual-continuation-v1`, and `global-experiment-assurance-lifecycle-v1`.

- Outcome: `NO_QUALIFIED_DEVELOPMENT_CANDIDATE`.
- Assurance: `REPRODUCED_LOCAL`; reviewer: `NOT_SUBMITTED`; blind status: `NOT_BLIND_SAME_OWNER_REPLAY`.
- Packet SHA-256: `8431bade93a85ce559e34dac3d3d324da78663b96a21540ffbcf10ea54319196`; result SHA-256: `f24b90241e4e06aa4bf54e92ec396afe8779b8f1a32d72ee54cde782fc3b44d1`.
- Claim ceiling: `LOCAL_LOW_RANK_CAUSAL_RESIDUAL_QUALIFICATION_ON_ONE_REUSED_OPT_CHECKPOINT`.
- Custody: all prompt-bearing and row-level artifacts remain in external owner custody.

## Nine predeclared development cells

Each cell must pass the effect ratio <=0.90, pointwise value ratio <=1.25, continuation CE increase <=0.05, and normalized next-logit error <=0.05 gates.

| Seed index | Stratum | Effect ratio | Value ratio | CE increase | Next-logit error | Cell passes |
|---:|---|---:|---:|---:|---:|:---:|
| 0 | ordinary | 1.0639 | 1.9502 | -0.0570 | 0.0104 | False |
| 0 | compositional | 1.0755 | 1.7505 | 0.0020 | 0.0089 | False |
| 0 | adversarial | 0.2679 | 1.7910 | -0.0385 | 0.0184 | False |
| 1 | ordinary | 1.0703 | 2.1506 | -0.0754 | 0.0102 | False |
| 1 | compositional | 1.0714 | 1.8496 | 0.0064 | 0.0127 | False |
| 1 | adversarial | 0.2700 | 1.9707 | -0.0397 | 0.0103 | False |
| 2 | ordinary | 1.0697 | 2.0049 | -0.0642 | 0.0136 | False |
| 2 | compositional | 1.0699 | 1.7157 | 0.0220 | 0.0097 | False |
| 2 | adversarial | 0.2790 | 1.8041 | 0.0005 | 0.0068 | False |

Equal-cell mean effect ratio: `0.804188`; worst cell: `1.075524`.

Behavior and mechanism diverged: all 9 continuation-CE and next-logit behavior gates pass, but only 3/9 causal-effect gates pass. The six ordinary/compositional cells have effect ratios from `1.0639` to `1.0755`.

Next gate: new method packet required; do not create confirmation fixtures.
