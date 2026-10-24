"""02 -- NO TRAINING. (a) censored-target oracle ceiling, (b) RF/LR behaviour where all six predictions tie at the -20 floor,
(c) margin-stratified check incl. ITU mu=0 variant, (d) rank-corr NaN decomposition."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/final_candidate/analysis")
BD="/home/claude/phase5_sensitivity/baseline_data/"; P3="/mnt/attach/outputs/phase3/data/"; IT="/home/claude/phase5_sensitivity/itu_variant_data/"
FO=[(w,m) for w in (850,1250,1550) for m in (4,16)]
def rule_b(pred):
    ber={c:10.0**v for c,v in pred.items()}
    passing=[c for c in FO if ber[c]<=1e-9]
    if not passing: return min(FO,key=lambda c:ber[c])
    rb={c:(5.0 if c[1]==4 else 10.0) for c in passing}; best=max(rb.values()); tied=[c for c in passing if rb[c]==best]
    if len(tied)==1: return tied[0]
    mg={c:-9.0-pred[c] for c in tied}; bm=max(mg.values()); tm=[c for c in tied if abs(mg[c]-bm)<1e-9]
    return [c for c in FO if c in tm][0]
def tied_set(pred):  # configs Rule B cannot distinguish (same tier, equal margin)
    ber={c:10.0**v for c,v in pred.items()}; passing=[c for c in FO if ber[c]<=1e-9]
    if not passing: return {min(FO,key=lambda c:ber[c])}
    rb={c:(5.0 if c[1]==4 else 10.0) for c in passing}; best=max(rb.values()); tied=[c for c in passing if rb[c]==best]
    mg={c:-9.0-pred[c] for c in tied}; bm=max(mg.values()); return {c for c in tied if abs(mg[c]-bm)<1e-9}
pc=pd.read_csv(P3+"phase3_per_config_predictions.csv"); det=pd.read_csv(BD+"lowco_full_decision_detail.csv")
rows=[]
for env,g in pc.groupby("env_id"):
    cfg=[(int(w),int(m)) for w,m in zip(g.wavelength_nm,g.modulation_M)]
    tgt=dict(zip(cfg,g.log10_BER_target.values)); raw=dict(zip(cfg,g.log10_BER_raw.values))
    tc=[c for c,s in zip(cfg,g.selected_by_rule_B) if s][0]
    orc=rule_b(tgt); ts=tied_set(tgt)
    r=dict(env_id=env,regime=g.weather_label.iloc[0],true=tc,oracle_cens=orc,oracle_ok=(orc==tc),tied_n=len(ts),true_in_tied=(tc in ts))
    for mdl in ["RandomForest","LinearRegression"]:
        pv=dict(zip(cfg,g[f"pred_{mdl}"].values)); ch=rule_b(pv)
        r[mdl]=ch; r[mdl+"_ok"]=(ch==tc); r[mdl+"_in_tied"]=(ch in ts); r[mdl+"_alltie"]=(np.ptp(list(pv.values()))<1e-9)
    rows.append(r)
d=pd.DataFrame(rows); d.to_csv("/home/claude/final_candidate/outputs/v6_censoring_oracle_per_env.csv",index=False)
print("(a) Rule B applied to the TRUE CENSORED target (a perfect predictor of log10_BER_target):")
print(" overall decision accuracy vs true decision:",round(d.oracle_ok.mean(),4))
print(d.groupby("regime").agg(n=("env_id","size"),oracle_acc=("oracle_ok","mean"),RF=("RandomForest_ok","mean"),LR=("LinearRegression_ok","mean"),
   RF_in_tied=("RandomForest_in_tied","mean"),LR_in_tied=("LinearRegression_in_tied","mean"),true_in_tied=("true_in_tied","mean"),RF_alltie=("RandomForest_alltie","mean")).round(3).to_string())
print("macro oracle acc:",round(d.groupby("regime").oracle_ok.mean().mean(),4),"| macro RF acc",round(d.groupby("regime").RandomForest_ok.mean().mean(),4),
      "| macro LR acc",round(d.groupby("regime").LinearRegression_ok.mean().mean(),4))
print("macro 'chosen config in censored-oracle tied set': RF",round(d.groupby("regime").RandomForest_in_tied.mean().mean(),4),"LR",round(d.groupby("regime").LinearRegression_in_tied.mean().mean(),4))
cl=d[d.regime.str.startswith("Clear")]
print("(b) Clear regime: true config counts",cl.true.value_counts().to_dict(),"| RF choice counts",cl.RandomForest.value_counts().to_dict(),"| LR choice counts",cl.LinearRegression.value_counts().to_dict())
# (d) NaN decomposition of rank corr
rr=[]
for env,g in pc.groupby("env_id"):
    tv=g.log10_BER_target.values
    for mdl in ["RandomForest","LinearRegression"]:
        pv=g[f"pred_{mdl}"].values; rr.append(dict(env_id=env,regime=g.weather_label.iloc[0],model=mdl,true_const=tv.std()<1e-9,pred_const=pv.std()<1e-9))
rr=pd.DataFrame(rr)
print("(d) rank-corr undefined: share true-target constant:",rr.groupby("model").true_const.mean().round(3).to_dict(),"share pred constant:",rr.groupby("model").pred_const.mean().round(3).to_dict())
print(rr.groupby(["regime","model"]).agg(true_const=("true_const","mean"),pred_const=("pred_const","mean")).round(3).unstack().to_string())
# (c) margin strata, baseline vs ITU
for nm,detf,mgf in [("baseline",BD+"lowco_full_decision_detail.csv",P3+"phase3_margin_analysis.csv"),("ITU mu=0",IT+"lowco_full_decision_detail_ITU18141mu0.csv",IT+"phase3_margin_analysis_ITU18141mu0.csv")]:
    x=pd.read_csv(detf).merge(pd.read_csv(mgf),on="env_id"); x=x[x.contest_status=="contested"].copy()
    x["b"]=pd.cut(x.tier_contest_margin,bins=[-0.01,1,5,1e9],labels=["small","medium","large"])
    t=(x.groupby(["model","b"],observed=True).decision_match.mean().unstack()*100).round(1); t["monotone_small<=med<=large"]=(t["small"]<=t["medium"])&(t["medium"]<=t["large"])
    print(f"(c) margin-stratified acc, {nm}:\n",t.to_string())
