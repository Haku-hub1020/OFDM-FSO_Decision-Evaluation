import sys, time, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
SEED=20260906; FEATURES=["V_km_observed","R_mmh_observed","L_m","wavelength_nm","modulation_M"]; TARGET="log10_BER_target"
FO=[(w,m) for w in (850,1250,1550) for m in (4,16)]; CAP=15000
OUT="/home/claude/final_candidate/outputs/"
df=pd.read_parquet("/home/claude/phase5_sensitivity/baseline_data/dataset_primary_R1250_0.86.parquet")
for c in ["env_id","weather_label","stratum","scenario"]: df[c]=df[c].astype(object)
REG=sorted(df.weather_label.unique().tolist())
def rule_b(pred):
    ber={c:10.0**v for c,v in pred.items()}; passing=[c for c in FO if ber[c]<=1e-9]
    if not passing: return min(FO,key=lambda c:ber[c]),{min(FO,key=lambda c:ber[c])}
    rb={c:(5.0 if c[1]==4 else 10.0) for c in passing}; best=max(rb.values()); tied=[c for c in passing if rb[c]==best]
    if len(tied)==1: return tied[0],set(tied)
    mg={c:-9.0-pred[c] for c in tied}; bm=max(mg.values()); tm=[c for c in tied if abs(mg[c]-bm)<1e-9]
    return [c for c in FO if c in tm][0],set(tm)
def models():
    return {"LinearRegression":(LinearRegression(),True),
     "RandomForest":(RandomForestRegressor(n_estimators=200,max_depth=14,min_samples_leaf=2,random_state=SEED,n_jobs=-1),False),
     "XGBoost":(xgb.XGBRegressor(n_estimators=300,max_depth=6,learning_rate=0.1,subsample=0.8,random_state=SEED,n_jobs=-1,verbosity=0),False),
     "SVR":(SVR(C=10.0,epsilon=0.1,kernel="rbf"),True),
     "MLP":(MLPRegressor(hidden_layer_sizes=(64,32),max_iter=400,random_state=SEED,early_stopping=True,n_iter_no_change=10),True)}
envtab=df.groupby("env_id",sort=True).agg(regime=("weather_label","first"),V=("V_km_observed","first"),R=("R_mmh_observed","first"),L=("L_m","first")).reset_index()
tc=df[df.selected_by_rule_B][["env_id","wavelength_nm","modulation_M"]]; envtab=envtab.merge(tc,on="env_id"); envtab["label"]=list(zip(envtab.wavelength_nm.astype(int),envtab.modulation_M.astype(int)))
# censored-oracle tied set per env
orc={}
for env,g in df.groupby("env_id"):
    cfg=[(int(w),int(m)) for w,m in zip(g.wavelength_nm,g.modulation_M)]; ch,ts=rule_b(dict(zip(cfg,g[TARGET].values))); orc[env]=(ch,ts)
rows=[]; preds=[]
for regime in REG:
    tr=df[df.weather_label!=regime]; te=df[df.weather_label==regime]
    for name,(mdl,scale) in models().items():
        t0=time.time(); X=tr[FEATURES].values.astype(float); y=tr[TARGET].values
        if name in("SVR","MLP") and len(X)>CAP:
            idx=np.random.default_rng(0).choice(len(X),CAP,replace=False); X,y=X[idx],y[idx]
        sc=None
        if scale: sc=StandardScaler().fit(X); X=sc.transform(X)
        mdl.fit(X,y); Xt=te[FEATURES].values.astype(float); Xt=sc.transform(Xt) if scale else Xt
        p=mdl.predict(Xt); t=te[["env_id","wavelength_nm","modulation_M"]].copy(); t["pred"]=p; t["model"]=name; t["regime"]=regime; preds.append(t)
        for env,g in t.groupby("env_id"):
            cfg=[(int(w),int(m)) for w,m in zip(g.wavelength_nm,g.modulation_M)]; ch,_=rule_b(dict(zip(cfg,g.pred.values)))
            rows.append(dict(model=name,regime=regime,env_id=env,ml=ch,kind="regression"))
        print(f"{regime:32s} {name:16s} {time.time()-t0:6.1f}s",flush=True)
    # direct classifiers
    trE=envtab[envtab.regime!=regime]; teE=envtab[envtab.regime==regime]; Xe=trE[["V","R","L"]].values; ye=trE.label.map(lambda c:FO.index(c)).values
    sc=StandardScaler().fit(Xe); Xs=sc.transform(Xe); Xts=sc.transform(teE[["V","R","L"]].values)
    clfs={"DirectLogReg":(LogisticRegression(max_iter=2000),True),
      "DirectRF":(RandomForestClassifier(n_estimators=200,max_depth=14,min_samples_leaf=2,random_state=SEED,n_jobs=-1),False),
      "DirectXGB":(xgb.XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,subsample=0.8,random_state=SEED,n_jobs=-1,verbosity=0),False)}
    for name,(c,scl) in clfs.items():
        t0=time.time(); le=LabelEncoder().fit(ye)
        c.fit(Xs if scl else Xe, le.transform(ye)); pr=le.inverse_transform(c.predict(Xts if scl else teE[["V","R","L"]].values))
        for env,k in zip(teE.env_id,pr): rows.append(dict(model=name,regime=regime,env_id=env,ml=FO[int(k)],kind="direct"))
        print(f"{regime:32s} {name:16s} {time.time()-t0:6.1f}s",flush=True)
res=pd.DataFrame(rows); res["true"]=res.env_id.map(envtab.set_index("env_id").label)
res["strict_ok"]=[a==b for a,b in zip(res.ml,res.true)]; res["equiv_ok"]=[m in orc[e][1] for m,e in zip(res.ml,res.env_id)]
res["ml_str"]=res.ml.astype(str); res["true_str"]=res.true.astype(str)
res.drop(columns=["ml","true"]).to_csv(OUT+"v7_rerun_decisions_all_models.csv",index=False)
pd.concat(preds).to_parquet(OUT+"v7_rerun_per_config_predictions_all_models.parquet",index=False)
print("DONE")
