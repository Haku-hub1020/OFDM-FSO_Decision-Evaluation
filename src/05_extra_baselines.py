import pandas as pd, numpy as np
from scipy.stats import spearmanr
O="/home/claude/final_candidate/outputs/"; BD="/home/claude/phase5_sensitivity/baseline_data/"
df=pd.read_parquet(BD+"dataset_primary_R1250_0.86.parquet")
env=df[df.selected_by_rule_B].groupby("env_id").agg(regime=("weather_label","first"),w=("wavelength_nm","first"),m=("modulation_M","first"),case=("rule_B_case","first")).reset_index()
env["true"]=list(zip(env.w.astype(int),env.m.astype(int)))
print("true-config distribution by regime:\n",pd.crosstab(env.regime,env.true.astype(str)).to_string())
print("rule_B_case by regime:\n",pd.crosstab(env.regime,env.case).to_string())
for c in [(1550,16),(1550,4)]:
    a=env.groupby("regime").apply(lambda g:(g.true==c).mean(),include_groups=False); print(f"CONSTANT-{c} baseline strict acc by regime (%):\n",(a*100).round(1).to_string(),"\n macro",round(a.mean()*100,1),"micro",round((env.true==c).mean()*100,1))
# rho restricted to non-degenerate regimes
rg=pd.read_csv(O+"v7_per_regime_summary.csv"); eb=pd.read_csv(BD+"expB_results.csv")
nd=["Dense Fog (0.5-1km)","Rain: Heavy (>7.6 mm/h)","Very Dense Fog (<0.5km)"]
rows=[]
for _,x in eb.iterrows():
    for m in ["LinearRegression","RandomForest","XGBoost","SVR","MLP"]:
        e=rg[(rg.regime==x.regime)&(rg.model==m)].iloc[0]; rows.append((x.regime,m,x[f"{m}_r2"],e.strict,e.equiv))
a=pd.DataFrame(rows,columns=["regime","model","r2","strict","equiv"]); b=a[a.regime.isin(nd)]
print("non-degenerate 3 regimes x 5 models (n=15): rho(R2,strict)=",spearmanr(b.r2,b.strict),"rho(R2,equiv)=",spearmanr(b.r2,b.equiv))
print("R2 in degenerate regimes (n=5 regimes x5):\n",a[~a.regime.isin(nd)].pivot(index="regime",columns="model",values="r2").round(3).to_string())
print(a[a.regime.isin(nd)].pivot(index="regime",columns="model",values="r2").round(2).to_string())
