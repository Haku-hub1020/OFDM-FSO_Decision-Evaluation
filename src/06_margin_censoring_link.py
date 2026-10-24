import pandas as pd, numpy as np
BD="/home/claude/phase5_sensitivity/baseline_data/"; P3="/mnt/attach/outputs/phase3/data/"
df=pd.read_parquet(BD+"dataset_primary_R1250_0.86.parquet"); mg=pd.read_csv(P3+"phase3_margin_analysis.csv")
mg=mg[mg.contest_status=="contested"].copy(); mg["b"]=pd.cut(mg.tier_contest_margin,bins=[-0.01,1,5,1e9],labels=["small","medium","large"])
# for each contested env: winning tier = 16-QAM if any 16-QAM config feasible else 4-QAM; are the top-2 configs of that tier both below the -20 floor in raw?
res=[]
for env,g in df[df.env_id.isin(mg.env_id)].groupby("env_id"):
    f=g[g.feasible_1e9]; tier=16 if (f.modulation_M==16).any() else 4
    t=f[f.modulation_M==tier].sort_values("log10_BER_raw")[["log10_BER_raw"]].values.ravel()[:2]  # two lowest BER = two largest margins
    res.append((env,(t<-20).all()))
r=pd.DataFrame(res,columns=["env_id","top2_both_censored"]).merge(mg[["env_id","b"]],on="env_id")
print(r.groupby("b",observed=True).top2_both_censored.agg(["mean","size"]).round(3).to_string())
