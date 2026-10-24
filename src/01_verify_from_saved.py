"""01_verify_from_saved.py -- NO TRAINING. Re-derives Phase 5 headline numbers from saved artifacts.
Reads (read-only): baseline_data/*, /mnt/attach/outputs/phase3/data/*. Writes only to final_candidate/outputs."""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, "/home/claude/phase5_sensitivity")
BD="/home/claude/phase5_sensitivity/baseline_data/"; P3="/mnt/attach/outputs/phase3/data/"; OUT="/home/claude/final_candidate/outputs/"
FIXED_ORDER=[(w,m) for w in (850,1250,1550) for m in (4,16)]
def rule_b(pred):  # identical logic to phase2b_config_selection.ml_rule_b_decision
    ber={c:10.0**v for c,v in pred.items()}
    passing=[c for c in FIXED_ORDER if ber[c]<=1e-9]
    if not passing: return min(FIXED_ORDER,key=lambda c:ber[c])
    rb={c:(5.0 if c[1]==4 else 10.0) for c in passing}; best=max(rb.values()); tied=[c for c in passing if rb[c]==best]
    if len(tied)==1: return tied[0]
    mg={c:-9.0-pred[c] for c in tied}; bm=max(mg.values()); tm=[c for c in tied if abs(mg[c]-bm)<1e-9]
    return [c for c in FIXED_ORDER if c in tm][0]
det=pd.read_csv(BD+"lowco_full_decision_detail.csv"); tab=pd.read_csv(BD+"lowco_full_decision_table.csv")
# V1: aggregate detail -> table
g=det.groupby(["regime","model"]).agg(n=("env_id","size"),acc=("decision_match","mean"),wl=("wavelength_match","mean"),modm=("modulation_match","mean"),
   inf=("ml_choice_actually_feasible",lambda s:1-s.mean()),reg=("bitrate_regret_Gbps","mean")).reset_index()
m=tab.merge(g,on=["regime","model"])
print("V1 table-vs-detail max abs diff: acc",(m.decision_accuracy-m.acc).abs().max(),"wl",(m.wavelength_accuracy-m.wl).abs().max(),
      "mod",(m.modulation_accuracy-m.modm).abs().max(),"infeas",(m.infeasible_rate-m.inf).abs().max(),"regret",(m.mean_bitrate_regret-m.reg).abs().max(),"n",(m.n_test_env-m.n).abs().max())
macro=tab.groupby("model").decision_accuracy.mean(); micro=det.groupby("model").decision_match.mean()
print("V1 macro:\n",(macro*100).round(1).to_string()); print("V1 micro:\n",(micro*100).round(1).to_string())
print("V1 RF VDF infeasible rate:", tab[(tab.model=="RandomForest")&(tab.regime.str.startswith("Very Dense"))].infeasible_rate.values)
# V4: regression R2 vs decision accuracy (expB)
eb=pd.read_csv(BD+"expB_results.csv"); rows=[]
for _,r in eb.iterrows():
    for mdl in ["LinearRegression","RandomForest","XGBoost","SVR","MLP"]:
        d=tab[(tab.regime==r.regime)&(tab.model==mdl)].decision_accuracy.iloc[0]; rows.append((r.regime,mdl,r[f"{mdl}_r2"],d))
rd=pd.DataFrame(rows,columns=["regime","model","r2","dec"]); rho,p=spearmanr(rd.r2,rd.dec); print(f"V4 rho(R2,decision acc) n={len(rd)}: {rho:.3f} p={p:.3f}")
print("V4 expB mean R2:", {k:round(eb[f"{k}_r2"].mean(),2) for k in ["RandomForest","LinearRegression","XGBoost","SVR","MLP"]})
rd.to_csv(OUT+"v4_r2_vs_decision_40obs.csv",index=False)
# V2: decisions re-derived from saved per-config predictions vs saved decision detail (RF, LR)
pc=pd.read_csv(P3+"phase3_per_config_predictions.csv"); print("V2 pred rows",len(pc),"envs",pc.env_id.nunique())
res=[]
for env,gp in pc.groupby("env_id"):
    cfg=[(int(w),int(mm)) for w,mm in zip(gp.wavelength_nm,gp.modulation_M)]
    t=gp.loc[gp.selected_by_rule_B,["wavelength_nm","modulation_M"]].iloc[0]; tc=(int(t.wavelength_nm),int(t.modulation_M))
    for mdl,col in [("RandomForest","pred_RandomForest"),("LinearRegression","pred_LinearRegression")]:
        ch=rule_b(dict(zip(cfg,gp[col].values)))
        ber_true=dict(zip(cfg,10.0**gp.log10_BER_raw.clip(upper=0).values))
        res.append(dict(env_id=env,model=mdl,regime=gp.weather_label.iloc[0],true_cfg=tc,rederived_cfg=ch))
re=pd.DataFrame(res)
dd=det[det.model.isin(["RandomForest","LinearRegression"])][["env_id","model","true_config","ml_config","decision_match"]]
mm=re.merge(dd,on=["env_id","model"]); mm["rederived_str"]=mm.rederived_cfg.astype(str)
mm["same_as_saved"]=(mm.rederived_str==mm.ml_config); mm["true_same"]=(mm.true_cfg.astype(str)==mm.true_config)
print("V2 rows compared",len(mm),"re-derived ML choice == saved Phase 2B.1 choice:",mm.same_as_saved.mean(),"true config agree:",mm.true_same.mean())
print(mm.groupby("model").same_as_saved.agg(["mean","size"]).to_string())
print(mm[~mm.same_as_saved].groupby(["model","regime"]).size().to_string())
mm.to_csv(OUT+"v2_rederived_vs_saved_decisions.csv",index=False)
# V3: rank correlation recompute
rk=pd.read_csv(P3+"phase3_rank_correlation.csv"); print("V3 saved rank-corr means:\n",rk.groupby("model").rank_corr.mean().round(3).to_string())
print(rk[rk.weather_label.str.startswith("Very Dense")].groupby("model").rank_corr.mean().round(3).to_string())
rr=[]
for env,gp in pc.groupby("env_id"):
    tv=gp.log10_BER_target.values
    for mdl in ["RandomForest","LinearRegression"]:
        pv=gp[f"pred_{mdl}"].values
        rr.append(dict(env_id=env,model=mdl,rank_corr_new=np.nan if (tv.std()<1e-9 or pv.std()<1e-9) else spearmanr(tv,pv)[0]))
rr=pd.DataFrame(rr).merge(rk,on=["env_id","model"]); 
print("V3 recompute vs saved: max abs diff",(rr.rank_corr_new-rr.rank_corr).abs().max(),"NaN agree",(rr.rank_corr_new.isna()==rr.rank_corr.isna()).mean())
print("V3 share NaN (degenerate):", rr.groupby("model").rank_corr.apply(lambda s:s.isna().mean()).round(3).to_string())
# V5: margin-stratified accuracy
mg=pd.read_csv(P3+"phase3_margin_analysis.csv"); x=det.merge(mg,on="env_id"); x=x[x.contest_status=="contested"].copy()
x["bucket"]=pd.cut(x.tier_contest_margin,bins=[-0.01,1,5,1e9],labels=["small","medium","large"])
print("V5 margin-stratified decision accuracy (%):\n",(x.groupby(["model","bucket"],observed=True).decision_match.mean().unstack()*100).round(1).to_string())
print("V5 share of envs: ",mg.contest_status.value_counts(normalize=True).round(3).to_dict())
