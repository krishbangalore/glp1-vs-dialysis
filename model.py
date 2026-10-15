"""
Part 2, step 4 - Which nephrologists start prescribing GLP-1s?

Question: among nephrologists NOT prescribing a GLP-1 in year Y, who starts in year Y+1?
Features come only from year Y (no peeking at the future).

Temporal validation (how a real model would be used):
  TRAIN on the 2022 -> 2023 cohort  (who started in 2023)
  TEST  on the 2023 -> 2024 cohort  (who started in 2024, the year after the FLOW news)

Models: logistic regression (interpretable) and random forest (nonlinear).
"Prescribes" = 11+ Medicare claims in a year (CMS hides smaller counts).

Outputs (in ./output):
  model_metrics.csv, lr_odds_ratios.csv, rf_importance.csv,
  model_performance.png, model_drivers.png, features_<year>.csv
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "output")
SEED = 42

# --- drug groups (matched as substrings of the lowercase generic name) -------------
GLP1 = ["semaglutide", "tirzepatide", "dulaglutide", "liraglutide", "exenatide", "lixisenatide"]
GROUPS = {
    "sglt2": ["gliflozin"],
    "finerenone": ["finerenone"],
    "insulin": ["insulin"],
    "metformin": ["metformin"],
    "dpp4": ["gliptin"],
    "sulfonylurea": ["glipizide", "glimepiride", "glyburide"],
    # drugs used almost only for dialysis patients: phosphate binders, cinacalcet, active vitamin D
    "dialysis_drugs": ["sevelamer", "calcium acetate", "lanthanum", "ferric citrate",
                       "sucroferric", "cinacalcet", "calcitriol", "paricalcitol", "doxercalciferol"],
}
DIABETES = ["insulin", "metformin", "dpp4", "sulfonylurea", "sglt2"]

REGION = {  # US Census regions
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_REGION = {s: r for r, states in REGION.items() for s in states.split()}

# Note: "share of patients under 65" was tested and dropped - it is 93% correlated with
# average patient age, which made the two coefficients misleading.
LABELS = {  # readable names for charts and tables
    "prior_glp1_peer_rate": "% of other nephrologists in state on GLP-1s",
    "diabetes_claim_share": "Share of claims that are diabetes drugs",
    "dialysis_claim_share": "Share of claims that are dialysis drugs",
    "uses_sglt2": "Already prescribes SGLT2s",
    "uses_finerenone": "Already prescribes finerenone (Kerendia)",
    "uses_insulin": "Prescribes insulin",
    "uses_metformin": "Prescribes metformin",
    "uses_dpp4": "Prescribes DPP-4 inhibitors",
    "uses_sulfonylurea": "Prescribes sulfonylureas",
    "brand_claim_share": "Share of claims that are brand-name drugs",
    "n_distinct_drugs": "Number of distinct drugs prescribed",
    "log_patients": "Number of Medicare patients (log)",
    "claims_per_patient": "Prescriptions per patient",
    "avg_patient_age": "Patients' average age",
    "risk_score": "Patients' average risk score (sickness)",
    "dual_share": "Share of low-income (dual-eligible) patients",
    "black_share": "Share of Black patients",
    "female_share": "Share of female patients",
    "mapd_share": "Share of claims via Medicare Advantage",
    "rural": "Practices in a rural area",
    "is_DO": "DO (vs MD) degree",
    "region_Northeast": "Region: Northeast",
    "region_South": "Region: South",
    "region_West": "Region: West",
}


def classify(generic):
    g = str(generic).lower()
    if any(k in g for k in GLP1):
        return "glp1"
    for grp, keys in GROUPS.items():
        if any(k in g for k in keys):
            return grp
    return None


def drug_features(year):
    d = pd.read_csv(os.path.join(RAW, f"partd_nephrology_{year}.csv"), dtype={"Prscrbr_NPI": str})
    d["grp"] = d["Gnrc_Name"].map(classify)
    d["is_brand"] = d["Brnd_Name"].str.lower() != d["Gnrc_Name"].str.lower()
    npi = d.groupby("Prscrbr_NPI")
    f = pd.DataFrame({
        "total_claims_drugfile": npi["Tot_Clms"].sum(),
        "n_distinct_drugs": npi["Gnrc_Name"].nunique(),
        "brand_claims": d[d["is_brand"]].groupby("Prscrbr_NPI")["Tot_Clms"].sum(),
    }).fillna(0)
    grp_claims = d.dropna(subset=["grp"]).pivot_table(
        index="Prscrbr_NPI", columns="grp", values="Tot_Clms", aggfunc="sum", fill_value=0)
    f = f.join(grp_claims).fillna(0)
    for g in ["glp1"] + list(GROUPS):
        if g not in f:
            f[g] = 0
    f["brand_claim_share"] = f["brand_claims"] / f["total_claims_drugfile"]
    f["diabetes_claim_share"] = f[DIABETES].sum(axis=1) / f["total_claims_drugfile"]
    f["dialysis_claim_share"] = f["dialysis_drugs"] / f["total_claims_drugfile"]
    for g in ["glp1", "sglt2", "finerenone", "insulin", "metformin", "dpp4", "sulfonylurea"]:
        f[f"uses_{g}"] = (f[g] > 0).astype(int)
    return f


def provider_features(year):
    p = pd.read_csv(os.path.join(RAW, f"providers_nephrology_{year}.csv"), dtype={"Prscrbr_NPI": str})
    p = p.drop_duplicates("Prscrbr_NPI").set_index("Prscrbr_NPI")
    benes = p["Tot_Benes"].replace(0, np.nan)
    f = pd.DataFrame(index=p.index)
    f["state"] = p["Prscrbr_State_Abrvtn"]
    f["log_patients"] = np.log1p(p["Tot_Benes"])
    f["claims_per_patient"] = p["Tot_Clms"] / benes
    f["avg_patient_age"] = p["Bene_Avg_Age"]
    f["under65_share"] = p["Bene_Age_LT_65_Cnt"] / benes
    f["risk_score"] = p["Bene_Avg_Risk_Scre"]
    f["dual_share"] = p["Bene_Dual_Cnt"] / benes
    f["black_share"] = p["Bene_Race_Black_Cnt"] / benes
    f["female_share"] = p["Bene_Feml_Cnt"] / benes
    f["mapd_share"] = p["MAPD_Tot_Clms"] / p["Tot_Clms"]
    ruca = p["Prscrbr_RUCA"]
    f["rural"] = ((ruca >= 4) & (ruca < 99)).astype(int)        # RUCA 4+ = outside metro areas
    f["is_DO"] = p["Prscrbr_Crdntls"].fillna("").str.upper().str.contains(r"\bD\.?O\b").astype(int)
    f["region"] = f["state"].map(STATE_REGION).fillna("Other")   # PR, territories -> Other
    return f


def build(year):
    """Cohort: nephrologists NOT on GLP-1s in `year`; label = started in year+1."""
    prov, drugs = provider_features(year), drug_features(year)
    nxt_prov = provider_features(year + 1)
    nxt_drugs = drug_features(year + 1)
    df = prov.join(drugs, how="left")
    df = df[df.index.isin(nxt_prov.index)]                   # still practicing next year
    for c in drugs.columns:
        df[c] = df[c].fillna(0)

    # peer effect: share of OTHER nephrologists in the same state on GLP-1s this year
    st = df.groupby("state")["uses_glp1"].agg(["sum", "count"])
    s, n = df["state"].map(st["sum"]), df["state"].map(st["count"])
    df["prior_glp1_peer_rate"] = ((s - df["uses_glp1"]) / (n - 1).replace(0, np.nan)).fillna(0) * 100

    df = df[df["uses_glp1"] == 0].copy()                      # not yet prescribing
    df["started_glp1"] = df.index.isin(nxt_drugs.index[nxt_drugs["uses_glp1"] == 1]).astype(int)
    df = pd.get_dummies(df, columns=["region"], prefix="region", dtype=int)
    for r in ["region_Northeast", "region_South", "region_West"]:
        if r not in df:
            df[r] = 0
    return df


FEATURES = list(LABELS)


def prep(df, medians):
    X = df[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
    return X.fillna(medians)


def top_decile(y, p):
    k = max(1, int(len(p) * 0.10))
    idx = np.argsort(-p)[:k]
    precision = y[idx].mean()
    return precision, precision / y.mean(), y[idx].sum() / y.sum()


def main():
    os.makedirs(OUT, exist_ok=True)
    train, test = build(2022), build(2023)
    train.to_csv(os.path.join(OUT, "features_2022_to_2023.csv"))
    test.to_csv(os.path.join(OUT, "features_2023_to_2024.csv"))
    medians = train[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan).median()
    Xtr, Xte = prep(train, medians), prep(test, medians)
    ytr, yte = train["started_glp1"].values, test["started_glp1"].values
    print(f"Train (2022->2023): {len(ytr):,} nephrologists, {ytr.sum()} started GLP-1s ({ytr.mean():.1%})")
    print(f"Test  (2023->2024): {len(yte):,} nephrologists, {yte.sum()} started GLP-1s ({yte.mean():.1%})\n")

    models = {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000)),
        "Random forest": RandomForestClassifier(
            n_estimators=600, min_samples_leaf=15, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    }
    # simple benchmark: rank doctors by how diabetes-heavy their prescribing is
    preds = {"Benchmark: diabetes-drug share only": Xte["diabetes_claim_share"].values}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        preds[name] = m.predict_proba(Xte)[:, 1]

    rows = []
    for name, p in preds.items():
        prec, lift, capture = top_decile(yte, p)
        rows.append({"model": name, "ROC_AUC": roc_auc_score(yte, p),
                     "PR_AUC": average_precision_score(yte, p), "PR_AUC_random": yte.mean(),
                     "top10pct_precision": prec, "top10pct_lift": lift, "top10pct_capture": capture})
    metrics = pd.DataFrame(rows).round(3)
    metrics.to_csv(os.path.join(OUT, "model_metrics.csv"), index=False)

    # logistic regression: odds ratio per 1 standard deviation, with bootstrap 95% CIs
    lr = models["Logistic regression"]
    coefs = lr[-1].coef_[0]
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(200):
        i = rng.integers(0, len(ytr), len(ytr))
        b = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000))
        b.fit(Xtr.iloc[i], ytr[i])
        boots.append(b[-1].coef_[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    ors = pd.DataFrame({"feature": FEATURES, "label": [LABELS[f] for f in FEATURES],
                        "odds_ratio_per_SD": np.exp(coefs), "CI_low": np.exp(lo), "CI_high": np.exp(hi)})
    ors["significant"] = (ors["CI_low"] > 1) | (ors["CI_high"] < 1)
    ors = ors.sort_values("odds_ratio_per_SD", ascending=False).round(3)
    ors.to_csv(os.path.join(OUT, "lr_odds_ratios.csv"), index=False)

    # random forest: permutation importance on the TEST year (drop in PR-AUC when shuffled)
    rf = models["Random forest"]
    pi = permutation_importance(rf, Xte, yte, scoring="average_precision",
                                n_repeats=20, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATURES, "label": [LABELS[f] for f in FEATURES],
                        "importance": pi.importances_mean, "std": pi.importances_std})
    imp = imp.sort_values("importance", ascending=False).round(4)
    imp.to_csv(os.path.join(OUT, "rf_importance.csv"), index=False)

    # adoption rate by decile of the RF score (does the score sort doctors well?)
    dec = pd.qcut(pd.Series(preds["Random forest"]).rank(method="first"), 10, labels=range(1, 11))
    by_dec = pd.Series(yte).groupby(dec, observed=True).mean() * 100

    plot(yte, preds, ors, imp, by_dec)

    pd.set_option("display.width", 160)
    print("Performance on the 2024 test year:\n")
    print(metrics.to_string(index=False))
    print("\nLogistic regression - odds ratio per 1 SD (significant = 95% CI excludes 1):\n")
    print(ors[["label", "odds_ratio_per_SD", "CI_low", "CI_high", "significant"]].to_string(index=False))
    print("\nRandom forest - top 10 features by permutation importance:\n")
    print(imp.head(10)[["label", "importance", "std"]].to_string(index=False))
    print("\n2024 adoption rate by RF score decile (1 = lowest score):")
    print(by_dec.round(1).to_string())
    print("\nSaved: output/model_metrics.csv, lr_odds_ratios.csv, rf_importance.csv, "
          "model_performance.png, model_drivers.png")


def plot(y, preds, ors, imp, by_dec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Logistic regression": "#2471a3", "Random forest": "#27ae60",
              "Benchmark: diabetes-drug share only": "#95a5a6"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} (AUC {roc_auc_score(y, p):.2f})")
    ax.plot([0, 1], [0, 1], ls=":", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (2024 test year)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    for name, p in preds.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=colors[name], lw=2,
                label=f"{name} (AP {average_precision_score(y, p):.2f})")
    ax.axhline(y.mean(), ls=":", color="grey", label=f"Random guessing ({y.mean():.2f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (2024 test year)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.bar(by_dec.index.astype(int), by_dec.values, color="#27ae60")
    ax.axhline(y.mean() * 100, ls=":", color="grey")
    ax.text(1, y.mean() * 100 + 0.3, f"average {y.mean():.1%}", fontsize=8, color="dimgray")
    ax.set(xlabel="Random forest score decile (10 = most likely)", ylabel="% who started GLP-1s in 2024",
           title="Actual 2024 adoption by predicted risk")
    ax.set_xticks(range(1, 11))
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "model_performance.png"), dpi=180)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    o = ors.sort_values("odds_ratio_per_SD")
    col = np.where(~o["significant"], "#bdc3c7", np.where(o["odds_ratio_per_SD"] > 1, "#27ae60", "#c0392b"))
    ax.errorbar(o["odds_ratio_per_SD"], range(len(o)),
                xerr=[o["odds_ratio_per_SD"] - o["CI_low"], o["CI_high"] - o["odds_ratio_per_SD"]],
                fmt="none", ecolor="#7f8c8d", lw=1)
    ax.scatter(o["odds_ratio_per_SD"], range(len(o)), c=col, s=40, zorder=3)
    ax.axvline(1, color="black", lw=0.8)
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels(o["label"], fontsize=8.5)
    ax.set(xlabel="Odds ratio per 1 standard deviation (log scale)",
           title="Logistic regression: what raises or lowers the odds of starting")
    ax = axes[1]
    t = imp.head(12).iloc[::-1]
    ax.barh(t["label"], t["importance"], xerr=t["std"], color="#27ae60", ecolor="#7f8c8d")
    ax.set(xlabel="Drop in PR-AUC when the feature is shuffled",
           title="Random forest: most important features")
    ax.tick_params(axis="y", labelsize=8.5)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "model_drivers.png"), dpi=180)


if __name__ == "__main__":
    main()
