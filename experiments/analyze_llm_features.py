"""
analyze_llm_features.py

GPU-free diagnostic for the existing LLM dataset.
No existing dataset/experiment output is modified.

Outputs: results/llm_diagnostic/
All buggy-label analyses use TRAIN ONLY.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
LLM = ROOT / "data" / "llm_experiment_dataset"
JIT = ROOT / "results" / "chronological_splits"
OUT = ROOT / "results" / "llm_diagnostic"
PLOTS = OUT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

ID = "commit_id"
TARGET = "buggy"
CATS = ["intent","change","risk","complexity","scope","test","security"]
NUMS = [
    "intent_confidence","intent_margin","change_confidence","change_margin",
    "risk_confidence","risk_margin","complexity_confidence","complexity_margin",
    "scope_confidence","scope_margin","test_confidence","test_margin",
    "security_confidence","security_margin"
]
EXPECTED = {"train":41998, "validation":8998, "test":9000}

def norm_target(s):
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        if x.isna().any() or not set(x.dropna().unique()).issubset({0,1}):
            raise ValueError("Invalid buggy labels")
        return x.astype(int)
    m={"0":0,"false":0,"non-buggy":0,"non_buggy":0,
       "1":1,"true":1,"buggy":1}
    x=s.astype(str).str.strip().str.lower().map(m)
    if x.isna().any(): raise ValueError("Unknown buggy labels")
    return x.astype(int)

def language(s):
    p=s.astype(str).str.lower()
    return np.select([p.str.startswith("apache/"),p.str.startswith("cpp/"),
                      p.str.startswith("python/")],["Java","C++","Python"],"Unknown")

def load():
    data={}
    for split in EXPECTED:
        a=pd.read_csv(LLM/f"{split}.csv",low_memory=False)
        b=pd.read_csv(JIT/f"{split}.csv",usecols=lambda c:c in
                      [ID,TARGET,"project","author_date"],low_memory=False)
        need=[ID]+CATS+NUMS
        missing=[c for c in need if c not in a.columns]
        if missing: raise ValueError(f"{split}: missing LLM columns {missing}")
        if len(a)!=EXPECTED[split] or len(b)!=EXPECTED[split]:
            raise ValueError(f"{split}: unexpected row count")
        if a[ID].duplicated().any() or b[ID].duplicated().any():
            raise ValueError(f"{split}: duplicate commit_id")
        # Some versions of the LLM split files already contain the SZZ `buggy` label.
        # The canonical chronological split also contains `buggy`, so merging them
        # directly would create `buggy_x` / `buggy_y` and make `d["buggy"]` fail.
        # Use the canonical chronological label as the authoritative target, while
        # checking that an existing LLM-side label (if present) agrees with it.
        if TARGET in a.columns:
            llm_labels = norm_target(a[TARGET])
            canonical_labels = norm_target(b[TARGET])
            if not np.array_equal(
                llm_labels.to_numpy(), canonical_labels.to_numpy()
            ):
                # Compare by commit_id rather than row position.
                chk = a[[ID, TARGET]].copy()
                chk[TARGET] = llm_labels.to_numpy()
                chk = chk.merge(
                    b[[ID, TARGET]], on=ID, how="inner",
                    suffixes=("_llm", "_canonical"), validate="one_to_one"
                )
                if not np.array_equal(
                    chk[f"{TARGET}_llm"].to_numpy(),
                    norm_target(chk[f"{TARGET}_canonical"]).to_numpy()
                ):
                    raise ValueError(
                        f"{split}: LLM buggy labels do not match canonical labels"
                    )
            a = a.drop(columns=[TARGET])

        d=a.merge(b,on=ID,how="inner",validate="one_to_one")
        if len(d)!=len(a): raise ValueError(f"{split}: alignment failure")
        d[TARGET]=norm_target(d[TARGET])
        for c in NUMS: d[c]=pd.to_numeric(d[c],errors="coerce")
        d["language"]=language(d["project"])
        d["date"]=pd.to_datetime(d["author_date"],errors="coerce",utc=True)
        d["year"]=d["date"].dt.year
        data[split]=d
    ids={k:set(v[ID]) for k,v in data.items()}
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        if ids[a]&ids[b]: raise ValueError(f"Overlap: {a}/{b}")
    return data

def save_category_dist(data):
    rows=[]
    for sp,d in data.items():
        for f in CATS:
            x=d[f].fillna("<MISSING>").value_counts()
            for cat,n in x.items():
                rows.append([sp,f,str(cat),int(n),100*n/len(d)])
    pd.DataFrame(rows,columns=["split","feature","category","count","percentage"]).to_csv(
        OUT/"category_distributions.csv",index=False)

def save_num_stats(data):
    rows=[]
    for sp,d in data.items():
        for f in NUMS:
            x=d[f]
            rows.append([sp,f,x.notna().sum(),x.isna().sum(),x.mean(),x.median(),
                         x.std(),x.min(),x.quantile(.25),x.quantile(.75),x.max()])
    pd.DataFrame(rows,columns=["split","feature","count","missing","mean","median",
                               "std","min","p25","p75","max"]).to_csv(
        OUT/"confidence_margin_statistics.csv",index=False)

def save_bug_rates(train):
    rows=[]
    for f in CATS:
        for cat,g in train.groupby(f,dropna=False):
            rows.append([f,"<MISSING>" if pd.isna(cat) else cat,len(g),
                         int(g[TARGET].sum()),g[TARGET].mean()])
    pd.DataFrame(rows,columns=["feature","category","count","buggy_count","buggy_rate"]).to_csv(
        OUT/"train_category_bug_rates.csv",index=False)

    rows=[]
    for f in NUMS:
        x=train[[f,TARGET]].dropna().copy()
        if x[f].nunique()<2: continue
        try: x["bin"]=pd.qcut(x[f],5,duplicates="drop")
        except ValueError: x["bin"]=pd.cut(x[f],5)
        for b,g in x.groupby("bin",observed=True):
            rows.append([f,str(b),len(g),int(g[TARGET].sum()),g[TARGET].mean()])
    pd.DataFrame(rows,columns=["feature","bin","count","buggy_count","buggy_rate"]).to_csv(
        OUT/"train_numerical_bug_rates.csv",index=False)

def save_language(data):
    rows=[]
    for sp,d in data.items():
        for lang,g in d.groupby("language"):
            rows.append([sp,lang,len(g),100*len(g)/len(d),int(g[TARGET].sum()),g[TARGET].mean()])
    pd.DataFrame(rows,columns=["split","language","count","percentage","buggy_count","buggy_rate"]).to_csv(
        OUT/"language_distributions.csv",index=False)

    rows=[]
    for sp,d in data.items():
        for f in CATS:
            z=d.groupby(["language",f],dropna=False).size().reset_index(name="count")
            totals=d.groupby("language").size()
            for _,r in z.iterrows():
                cat="<MISSING>" if pd.isna(r[f]) else r[f]
                rows.append([sp,f,r["language"],cat,int(r["count"]),
                             100*r["count"]/totals[r["language"]]])
    pd.DataFrame(rows,columns=["split","feature","language","category","count",
                               "percentage_within_language"]).to_csv(
        OUT/"language_category_distributions.csv",index=False)

def save_temporal(data):
    rows=[]
    for sp,d in data.items():
        for year,g in d.groupby("year",dropna=False):
            row={"split":sp,"year":int(year) if pd.notna(year) else -1,
                 "count":len(g),"buggy_rate":g[TARGET].mean()}
            for f in NUMS: row[f"{f}_mean"]=g[f].mean()
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/"temporal_distributions.csv",index=False)

def cramers_v(a,b):
    t=pd.crosstab(a.fillna("<MISSING>"),b.fillna("<MISSING>")).to_numpy(float)
    n=t.sum()
    if n==0: return 0
    rs=t.sum(1,keepdims=True); cs=t.sum(0,keepdims=True)
    e=rs@cs/n
    chi=((t-e)**2/np.where(e>0,e,1)).sum()
    r,k=t.shape; phi=chi/n
    pc=max(0,phi-((k-1)*(r-1))/max(n-1,1))
    rc=r-(r-1)**2/max(n-1,1); kc=k-(k-1)**2/max(n-1,1)
    den=min(kc-1,rc-1)
    return float(np.sqrt(pc/den)) if den>0 else 0.0

def save_associations(train):
    rows=[]
    for i,a in enumerate(CATS):
        for b in CATS[i+1:]:
            rows.append([a,b,cramers_v(train[a],train[b])])
    pd.DataFrame(rows,columns=["feature_1","feature_2","cramers_v"]).sort_values(
        "cramers_v",ascending=False).to_csv(OUT/"categorical_associations.csv",index=False)
    train[NUMS].corr(method="spearman").to_csv(OUT/"numerical_correlations.csv")

def save_univariate(train):
    rows=[]; y=train[TARGET].to_numpy()
    for f in CATS:
        x=train[f].fillna("<MISSING>").astype(str)
        for cat in sorted(x.unique()):
            z=(x==cat).astype(int).to_numpy()
            if len(np.unique(z))<2: continue
            auc=roc_auc_score(y,z)
            rows.append([f,"categorical_indicator",cat,auc,max(auc,1-auc)])
    for f in NUMS:
        x=pd.to_numeric(train[f],errors="coerce"); ok=x.notna()
        if ok.sum()<2 or x[ok].nunique()<2: continue
        auc=roc_auc_score(y[ok],x[ok].to_numpy())
        rows.append([f,"numerical","",auc,max(auc,1-auc)])
    pd.DataFrame(rows,columns=["feature","type","category_or_value","roc_auc","symmetric_auc"]).sort_values(
        "symmetric_auc",ascending=False).to_csv(OUT/"univariate_signal.csv",index=False)

def plots(data):
    # Category distributions
    fig,axs=plt.subplots(7,1,figsize=(11,22))
    for ax,f in zip(axs,CATS):
        x=data["train"][f].fillna("<MISSING>").value_counts()
        ax.bar(x.index.astype(str),100*x.values/len(data["train"]))
        ax.set_title(f"Train distribution: {f}"); ax.set_ylabel("%"); ax.tick_params(axis="x",rotation=35)
    plt.tight_layout(); plt.savefig(PLOTS/"category_distributions.png",dpi=180); plt.close()

    # Confidence/margins
    sel=NUMS
    fig,axs=plt.subplots(7,2,figsize=(12,22))
    for ax,f in zip(axs.ravel(),sel):
        for sp in ["train","validation","test"]:
            ax.hist(data[sp][f].dropna(),bins=30,alpha=.3,density=True,label=sp)
        ax.set_title(f)
    axs[0,0].legend(); plt.tight_layout(); plt.savefig(PLOTS/"confidence_distributions.png",dpi=180); plt.close()

    # Train buggy rate by category
    fig,axs=plt.subplots(7,1,figsize=(11,22))
    for ax,f in zip(axs,CATS):
        x=data["train"].groupby(f,dropna=False)[TARGET].mean().sort_values()
        ax.bar(x.index.astype(str),x.values); ax.set_title(f"Train buggy rate: {f}"); ax.tick_params(axis="x",rotation=35)
    plt.tight_layout(); plt.savefig(PLOTS/"buggy_rate_by_category.png",dpi=180); plt.close()

    # Temporal buggy prevalence
    fig,ax=plt.subplots(figsize=(11,5))
    for sp in data:
        x=data[sp].dropna(subset=["year"]).groupby("year")[TARGET].mean()
        ax.plot(x.index,x.values,marker="o",label=sp)
    ax.set_title("Buggy prevalence over time"); ax.set_xlabel("Year"); ax.set_ylabel("Buggy rate"); ax.legend()
    plt.tight_layout(); plt.savefig(PLOTS/"temporal_buggy_prevalence.png",dpi=180); plt.close()

def summary(data):
    train=data["train"]
    u=pd.read_csv(OUT/"univariate_signal.csv")
    c=pd.read_csv(OUT/"categorical_associations.csv")
    lines=["LLM FEATURE DIAGNOSTIC SUMMARY","="*78,"",
           "All buggy-label association analyses use TRAIN ONLY.",""]
    for sp,d in data.items():
        lines.append(f"{sp}: rows={len(d)}, buggy={int(d[TARGET].sum())}, buggy_rate={d[TARGET].mean():.4f}")
    lines += ["","TOP UNIVARIATE SIGNAL (TRAIN ONLY)","-"*78]
    for _,r in u.head(20).iterrows():
        lines.append(f"{r.feature:28s} {str(r.category_or_value):18s} AUC={r.roc_auc:.4f} symmetric_AUC={r.symmetric_auc:.4f}")
    lines += ["","MOST ASSOCIATED CATEGORICAL PAIRS (TRAIN ONLY)","-"*78]
    for _,r in c.head(15).iterrows():
        lines.append(f"{r.feature_1:20s} x {r.feature_2:20s} Cramer's V={r.cramers_v:.4f}")
    lines += ["","INTERPRETATION GUIDE","-"*78,
              "• Large category concentration = limited variation.",
              "• Large buggy-rate differences = possible univariate signal.",
              "• High Cramer's V = possible redundancy between semantic dimensions.",
              "• Train/validation/test distribution changes = possible temporal/population drift.",
              "• This diagnostic does not tune or evaluate a final model on the test set."]
    (OUT/"diagnostic_summary.txt").write_text("\n".join(lines),encoding="utf-8")
    (OUT/"diagnostic_summary.json").write_text(json.dumps({
        "train_rows":len(train),"train_buggy_rate":float(train[TARGET].mean()),
        "top_univariate":u.head(20).to_dict("records"),
        "top_categorical_associations":c.head(15).to_dict("records")
    },indent=2),encoding="utf-8")

def main():
    print("="*78); print("LLM FEATURE DIAGNOSTIC — GPU FREE"); print("="*78)
    print("Loading existing LLM + canonical split data...")
    data=load()
    save_category_dist(data)
    save_num_stats(data)
    save_bug_rates(data["train"])
    save_language(data)
    save_temporal(data)
    save_associations(data["train"])
    save_univariate(data["train"])
    plots(data)
    summary(data)
    print("\nDIAGNOSTIC COMPLETE")
    print(f"Output: {OUT}")
    print("\nNo existing dataset or experiment output was modified.")

if __name__=="__main__":
    main()
