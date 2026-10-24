"""11 -- NO TRAINING, read-only supplement to 10: (a) replay with an alternative tie-break order (illustration of tie-break dependence),
(b) training-majority label per LOWCO fold, (c) paired env-level bootstrap for selected differences vs the constant policy (descriptive; regime set fixed)."""
import numpy as np, pandas as pd
BD="/home/claude/phase5_sensitivity/baseline_data/"; O="/home/claude/final_candidate/outputs/"
FO=[(w,m) for w in (850,1250,1550) for m in (4,16)]; FO_REV=[(w,m) for w in (1550,1250,850) for m in (4,16)]
def rule_b(pred,order):
    ber={c:10.0**v for c,v in pred.items()}; passing=[c for c in order if ber[c]<=1e-9]
    if not passing: return min(order,key=lambda c:ber[c])
    rb={c:(5.0 if c[1]==4 else 10.0) for c in passing}; best=max(rb.values()); tied=[c for c in passing if rb[c]==best]
    if len(tied)==1: return tied[0]
    mg={c:-9.0-pred[c] for c in tied}; bm=max(mg.values()); tm=[c for c in tied if abs(mg[c]-bm)<1e-9]
    return [c for c in order if c in tm][0]
df=pd.read_parquet(BD+"dataset_primary_R1250_0.86.parquet"); rows=[]
for env,g in df.groupby("env_id"):
    cfg=[(int(w),int(m)) for w,m in zip(g.wavelength_nm,g.modulation_M)]; true=[c for c,s in zip(cfg,g.selected_by_rule_B) if s][0]
    t=dict(zip(cfg,g.log10_BER_target.values)); rows.append(dict(env_id=env,regime=str(g.weather_label.iloc[0]),true=true,fwd=(rule_b(t,FO)==true),rev=(rule_b(t,FO_REV)==true)))
E=pd.DataFrame(rows)
print("(a) clipped-truth replay, fixed order 850-first (as in Rule B): macro",round(100*E.groupby('regime').fwd.mean().mean(),1),"micro",round(100*E.fwd.mean(),1))
print("    same replay, tie-break reversed (1550-first):             macro",round(100*E.groupby('regime').rev.mean().mean(),1),"micro",round(100*E.rev.mean(),1))
print("(b) training-majority label in each LOWCO fold:")
for r in sorted(E.regime.unique()):
    vc=E[E.regime!=r].true.value_counts(); print("   held-out",r[:24].ljust(24),"->",vc.index[0],int(vc.iloc[0]),"vs next",vc.index[1],int(vc.iloc[1]))
v=pd.read_csv(O+"v7_rerun_decisions_all_models.csv"); v["strict_ok"]=v.strict_ok.astype(bool)
const=(E.set_index("env_id").true==(1550,16)); rng=np.random.default_rng(0)
print("(c) paired env-level bootstrap (2000 resamples) of strict-accuracy difference vs constant (1550,16), within regime:")
for model,reg in [("LinearRegression","Rain: Heavy (>7.6 mm/h)"),("LinearRegression","Very Dense Fog (<0.5km)"),("LinearRegression","Dense Fog (0.5-1km)"),
                  ("DirectXGB","Very Dense Fog (<0.5km)"),("DirectRF","Very Dense Fog (<0.5km)"),("DirectRF","Rain: Heavy (>7.6 mm/h)"),("XGBoost","Rain: Heavy (>7.6 mm/h)")]:
    g=v[(v.model==model)&(v.regime==reg)].set_index("env_id"); d=(g.strict_ok.astype(float)-const.loc[g.index].astype(float)).values
    bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)]; print(f"   {model:17s} {reg[:22]:22s} n={len(d):5d} diff={100*d.mean():6.1f} pp  95% CI [{100*np.percentile(bs,2.5):6.1f}, {100*np.percentile(bs,97.5):6.1f}]")
