"""10 -- NO TRAINING, read-only. Final submission audit: (A) replay-rule implementation check, (B) equivalence-set audit,
(C) direct-classifier audit, (D) headline-number table. Inputs: baseline parquet/detail, outputs/v7_*, saved expA/B results."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
BD="/home/claude/phase5_sensitivity/baseline_data/"; O="/home/claude/final_candidate/outputs/"
FO=[(w,m) for w in (850,1250,1550) for m in (4,16)]
def rule_b(pred):
    ber={c:10.0**v for c,v in pred.items()}; passing=[c for c in FO if ber[c]<=1e-9]
    if not passing: c=min(FO,key=lambda c:ber[c]); return c,{c}
    rb={c:(5.0 if c[1]==4 else 10.0) for c in passing}; best=max(rb.values()); tied=[c for c in passing if rb[c]==best]
    if len(tied)==1: return tied[0],{tied[0]}
    mg={c:-9.0-pred[c] for c in tied}; bm=max(mg.values()); tm=[c for c in tied if abs(mg[c]-bm)<1e-9]
    return [c for c in FO if c in tm][0],set(tm)
df=pd.read_parquet(BD+"dataset_primary_R1250_0.86.parquet")
for c in ["env_id","weather_label"]: df[c]=df[c].astype(object)
rows=[]
for env,g in df.groupby("env_id"):
    cfg=[(int(w),int(m)) for w,m in zip(g.wavelength_nm,g.modulation_M)]
    true=[c for c,s in zip(cfg,g.selected_by_rule_B) if s][0]
    raw_ch,raw_ts=rule_b(dict(zip(cfg,g.log10_BER_raw.values))); tg_ch,tg_ts=rule_b(dict(zip(cfg,g.log10_BER_target.values)))
    rows.append(dict(env_id=env,regime=g.weather_label.iloc[0],true=true,case=g.rule_B_case.iloc[0],raw_ok=(raw_ch==true),replay=tg_ch,replay_ok=(tg_ch==true),
                     tied=tg_ts,tied_n=len(tg_ts),true_in_tied=(true in tg_ts)))
E=pd.DataFrame(rows); mac=lambda s:round(100*s.groupby(E.regime).mean().mean(),1); mic=lambda s:round(100*s.mean(),1)
print("== A. implementation check (all",len(E),"envs)")
print("rule_b on RAW target reproduces stored ground-truth label:",E.raw_ok.mean())
print("rule_b on CENSORED target (replay) vs label: macro",mac(E.replay_ok),"micro",mic(E.replay_ok))
mm=E[~E.replay_ok]; print("replay mismatches:",len(mm),"| of which tied set >1 (tie at floor):",int((mm.tied_n>1).sum()),"| replay choice counts:",mm.replay.value_counts().to_dict())
print("mismatch share by rule_B_case:\n",(1-E.groupby("case").replay_ok.mean()).round(3).to_string())
print("replay_ok by regime:\n",(100*E.groupby("regime").replay_ok.mean()).round(1).to_string())
print("\n== B. equivalence-set audit")
print("true choice NOT in censored tied set:",int((~E.true_in_tied).sum()),"of",len(E))
print("tied-set size>1 share by regime (%):\n",(100*E.assign(g=E.tied_n>1).groupby("regime").g.mean()).round(1).to_string())
print("tied-set size distribution overall:",E.tied_n.value_counts().sort_index().to_dict())
# constants under strict and equivalence
res={}
for c in FO:
    st=(E.true==c); eq=E.tied.map(lambda s:c in s)
    res[c]=dict(strict_macro=mac(st),strict_micro=mic(st),equiv_macro=mac(eq),equiv_micro=mic(eq))
print("constant policies:\n",pd.DataFrame(res).T.to_string())
con=["Dense Fog (0.5-1km)","Rain: Heavy (>7.6 mm/h)","Very Dense Fog (<0.5km)"]
print("\n== C/D. all selectors")
v=pd.read_csv(O+"v7_rerun_decisions_all_models.csv"); v["strict_ok"]=v.strict_ok.astype(bool); v["equiv_ok"]=v.equiv_ok.astype(bool)
tab=[]
for m,g in v.groupby("model"):
    by=g.groupby("regime")[["strict_ok","equiv_ok"]].mean()*100
    tab.append(dict(model=m,strict_macro=by.strict_ok.mean(),strict_micro=100*g.strict_ok.mean(),equiv_macro=by.equiv_ok.mean(),equiv_micro=100*g.equiv_ok.mean(),
                    strict_contested3=by.loc[con].strict_ok.mean(),equiv_contested3=by.loc[con].equiv_ok.mean()))
c16=(E.true==(1550,16)); by=100*c16.groupby(E.regime).mean()
tab.append(dict(model="CONST(1550,16)",strict_macro=by.mean(),strict_micro=100*c16.mean(),equiv_macro=np.nan,equiv_micro=np.nan,strict_contested3=by.loc[con].mean(),equiv_contested3=np.nan))
by=100*E.replay_ok.groupby(E.regime).mean(); tab.append(dict(model="REPLAY(censored)",strict_macro=by.mean(),strict_micro=100*E.replay_ok.mean(),equiv_macro=np.nan,equiv_micro=np.nan,strict_contested3=by.loc[con].mean(),equiv_contested3=np.nan))
print(pd.DataFrame(tab).set_index("model").round(1).to_string())
print("\nper-regime strict, direct classifiers minus constant (pp):")
pr=(100*v.groupby(["model","regime"]).strict_ok.mean()).unstack(0); pr["CONST"]=100*c16.groupby(E.regime).mean()
for m in ["DirectLogReg","DirectRF","DirectXGB"]: pr[m+"-CONST"]=pr[m]-pr["CONST"]
print(pr[["CONST","DirectLogReg-CONST","DirectRF-CONST","DirectXGB-CONST"]].round(1).to_string())
print("\n== C. classifier label audit")
lab=E.groupby(["regime","true"]).size().unstack(fill_value=0); print(lab.to_string())
for r in sorted(E.regime.unique()):
    tr=set(E[E.regime!=r].true); te=set(E[E.regime==r].true); print(f"held-out {r[:22]:22s} classes in test not in train: {sorted(te-tr)}")
# infeasible rate max
d=pd.read_csv(BD+"lowco_full_decision_detail.csv"); d["inf"]=~d.ml_choice_actually_feasible
ir=(100*d.groupby(["model","regime"]).inf.mean()).unstack(0); print("\ninfeasible-selection rate (%), max per model:\n",ir.max().round(1).to_string()); print("overall max cell:",ir.stack().idxmax(),round(ir.stack().max(),1))
print("true-infeasible share by regime (%):\n",(100*(~d[d.model=='LinearRegression'].true_feasible).groupby(d[d.model=='LinearRegression'].regime).mean()).round(1).to_string())
print("\nexpA:\n",pd.read_csv(BD+"expA_results.csv").round(3).to_string())
