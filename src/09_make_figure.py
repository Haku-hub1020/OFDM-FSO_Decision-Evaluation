"""09 -- NO TRAINING. Fig. 1: per-regime strict decision accuracy of regression selectors vs trivial baseline and censoring oracle ceiling.
Inputs (read-only): outputs/v7_per_regime_summary.csv, outputs/v6_censoring_oracle_per_env.csv, baseline_data/lowco_full_decision_detail.csv"""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
O="/home/claude/final_candidate/outputs/"; BD="/home/claude/phase5_sensitivity/baseline_data/"
s=pd.read_csv(O+"v7_per_regime_summary.csv"); orc=pd.read_csv(O+"v6_censoring_oracle_per_env.csv").groupby("regime").oracle_ok.mean()*100
det=pd.read_csv(BD+"lowco_full_decision_detail.csv"); t=det[det.model=="LinearRegression"]; const=(t.true_config=="(1550, 16)").groupby(t.regime).mean()*100
order=["Clear (V>50km)","Visibility Regime 1 (6-50km)","Visibility Regime 2 (1-6km)","Rain: Light (<2.5 mm/h)","Rain: Moderate (2.5-7.6 mm/h)","Dense Fog (0.5-1km)","Rain: Heavy (>7.6 mm/h)","Very Dense Fog (<0.5km)"]
short=["Clear","Vis. 1","Vis. 2","Rain\nlight","Rain\nmoderate","Dense\nfog","Rain\nheavy","Very dense\nfog"]
get=lambda m:[100*s[(s.model==m)&(s.regime==r)].strict.iloc[0] for r in order]
series=[("Clipped-truth replay (diagnostic, not a bound)",[orc[r] for r in order],"#999999","//"),("Constant (1550 nm, 16-QAM)",[const[r] for r in order],"#444444",""),
        ("Linear Regression",get("LinearRegression"),"#0072B2","\\\\"),("XGBoost",get("XGBoost"),"#009E73","xx"),("Random Forest",get("RandomForest"),"#D55E00","..")]
fig,ax=plt.subplots(figsize=(7.2,3.3),dpi=200); n=len(series); w=0.16; x=np.arange(len(order))
for i,(nm,v,c,h) in enumerate(series):
    ax.bar(x+(i-(n-1)/2)*w,v,w*0.9,label=nm,color=c if h=="" else "white",edgecolor=c,hatch=h,linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(short,fontsize=7); ax.set_ylabel("Strict decision accuracy (%)",fontsize=8); ax.set_ylim(0,105)
ax.tick_params(axis="y",labelsize=7); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",lw=0.3,alpha=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=6.5,ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.2))
fig.tight_layout(); fig.savefig("/home/claude/final_candidate/manuscript/fig1_regime_accuracy.png"); print("saved")
