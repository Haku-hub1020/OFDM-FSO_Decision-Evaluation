import pandas as pd, numpy as np
from scipy.stats import spearmanr
O="/home/claude/final_candidate/outputs/"; BD="/home/claude/phase5_sensitivity/baseline_data/"
r=pd.read_csv(O+"v7_rerun_decisions_all_models.csv"); det=pd.read_csv(BD+"lowco_full_decision_detail.csv")
# reproduction check vs saved Phase 2B.1 decisions
reg=r[r.kind=="regression"].merge(det[["env_id","model","ml_config"]],on=["env_id","model"])
print("REPRO strict ML choice == saved Phase 2B.1 choice, by model:\n",reg.assign(same=reg.ml_str==reg.ml_config).groupby("model").same.agg(["mean","size"]).to_string())
print("REPRO mismatches by model x regime:\n",reg[reg.ml_str!=reg.ml_config].groupby(["model","regime"]).size().to_string())
rg=r.groupby(["model","regime"]).agg(strict=("strict_ok","mean"),equiv=("equiv_ok","mean"),n=("env_id","size")).reset_index()
macro=rg.groupby("model")[["strict","equiv"]].mean().mul(100).round(1)
micro=r.groupby("model")[["strict_ok","equiv_ok"]].mean().mul(100).round(1)
print("MACRO % (strict, equiv):\n",macro.to_string()); print("MICRO %:\n",micro.to_string())
print("per-regime strict:\n",(rg.pivot(index="regime",columns="model",values="strict")*100).round(1).to_string())
print("per-regime equiv:\n",(rg.pivot(index="regime",columns="model",values="equiv")*100).round(1).to_string())
rg.to_csv(O+"v7_per_regime_summary.csv",index=False)
# rho between regression R2 (expB) and equivalence-decision accuracy (40 obs; regression models only)
eb=pd.read_csv(BD+"expB_results.csv"); rows=[]
for _,x in eb.iterrows():
    for m in ["LinearRegression","RandomForest","XGBoost","SVR","MLP"]:
        e=rg[(rg.regime==x.regime)&(rg.model==m)]; rows.append((x[f"{m}_r2"],e.strict.iloc[0],e.equiv.iloc[0]))
a=pd.DataFrame(rows,columns=["r2","strict","equiv"])
print("rho(R2, strict) =",spearmanr(a.r2,a.strict)); print("rho(R2, equiv) =",spearmanr(a.r2,a.equiv))
# restrict to the three non-degenerate regimes (oracle strict >0)
nd=["Dense Fog (0.5-1km)","Rain: Heavy (>7.6 mm/h)","Very Dense Fog (<0.5km)"]
print("Non-degenerate regimes (strict %):\n",(rg[rg.regime.isin(nd)].pivot(index="regime",columns="model",values="strict")*100).round(1).to_string())
print("Non-degenerate regimes (equiv %):\n",(rg[rg.regime.isin(nd)].pivot(index="regime",columns="model",values="equiv")*100).round(1).to_string())
# bootstrap CI (cluster over envs within regime) for macro strict & equiv per model
rng=np.random.default_rng(1); B=1000; out=[]
for m,gm in r.groupby("model"):
    byreg={k:v for k,v in gm.groupby("regime")}
    for metric in ["strict_ok","equiv_ok"]:
        vals=[]
        for _ in range(B):
            vals.append(np.mean([ v[metric].values[rng.integers(0,len(v),len(v))].mean() for v in byreg.values()]))
        out.append((m,metric,np.mean([v[metric].mean() for v in byreg.values()])*100,np.percentile(vals,2.5)*100,np.percentile(vals,97.5)*100))
ci=pd.DataFrame(out,columns=["model","metric","macro","lo","hi"]).round(1); print("Macro with env-level bootstrap 95% CI (regime set fixed; 8 regimes only):\n",ci.to_string())
ci.to_csv(O+"v7_macro_ci.csv",index=False)
