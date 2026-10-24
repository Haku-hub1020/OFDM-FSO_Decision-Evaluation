"""08 -- NO TRAINING. Repeats the censoring-ceiling, equivalence and constant-policy checks (scripts 02/05) on the saved ITU-R P.1814-1 mu=0 outputs
(RF and LR per-config predictions + decision detail only). Read-only inputs; output to stdout and final_candidate/outputs/08_itu_log.txt."""
import numpy as np, pandas as pd
IT="/home/claude/phase5_sensitivity/itu_variant_data/"
FO=[(w,m) for w in (850,1250,1550) for m in (4,16)]
def rule_b(pred):
    ber={c:10.0**v for c,v in pred.items()}; passing=[c for c in FO if ber[c]<=1e-9]
    if not passing: return min(FO,key=lambda c:ber[c]),{min(FO,key=lambda c:ber[c])}
    rb={c:(5.0 if c[1]==4 else 10.0) for c in passing}; best=max(rb.values()); tied=[c for c in passing if rb[c]==best]
    if len(tied)==1: return tied[0],{tied[0]}
    mg={c:-9.0-pred[c] for c in tied}; bm=max(mg.values()); tm=[c for c in tied if abs(mg[c]-bm)<1e-9]
    return [c for c in FO if c in tm][0],set(tm)
pc=pd.read_csv(IT+"phase3_per_config_predictions_ITU18141mu0.csv"); rows=[]
for env,g in pc.groupby("env_id"):
    cfg=[(int(w),int(m)) for w,m in zip(g.wavelength_nm,g.modulation_M)]
    tgt=dict(zip(cfg,g.log10_BER_target.values)); tc=[c for c,s in zip(cfg,g.selected_by_rule_B) if s][0]
    oc,ts=rule_b(tgt); r=dict(env_id=env,regime=g.weather_label.iloc[0],true=tc,oracle_ok=(oc==tc),const16=(tc==(1550,16)))
    for mdl in ["RandomForest","LinearRegression"]:
        ch,_=rule_b(dict(zip(cfg,g[f"pred_{mdl}"].values))); r[mdl+"_ok"]=(ch==tc); r[mdl+"_eq"]=(ch in ts)
    rows.append(r)
d=pd.DataFrame(rows); print("envs",len(d))
t=d.groupby("regime")[["oracle_ok","const16","RandomForest_ok","LinearRegression_ok","RandomForest_eq","LinearRegression_eq"]].mean().mul(100).round(1)
print(t.to_string()); print("MACRO",t.mean().round(1).to_dict()); print("MICRO",d[["oracle_ok","const16","RandomForest_ok","LinearRegression_ok","RandomForest_eq","LinearRegression_eq"]].mean().mul(100).round(1).to_dict())
