"""
Part 2, step 2 - How fast did nephrologists adopt kidney-protective drugs?

For each year 2019-2024: what share of Medicare nephrologists wrote 11+
prescriptions of a GLP-1, an SGLT2 inhibitor, or finerenone (Kerendia)?

Context: Novo stopped the FLOW kidney trial early for efficacy in Oct 2023 and
published results in May 2024, so 2024 is the first "after" year in this data.

Outputs (in ./output):
  adoption_by_year.csv      share of nephrologists prescribing each drug class
  glp1_by_drug.csv          GLP-1 claims by individual drug and year
  glp1_by_state_2024.csv    GLP-1 adoption rate by state in 2024
  adoption_chart.png
"""

import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "output")

# Matched against the lowercase generic name; substring match also catches combos
# like "Insulin Degludec/Liraglutide" or "Empagliflozin/Metformin".
CLASSES = {
    "GLP-1": ["semaglutide", "tirzepatide", "dulaglutide", "liraglutide",
              "exenatide", "lixisenatide"],      # tirzepatide = dual GIP/GLP-1, grouped here
    "SGLT2": ["gliflozin"],
    "Finerenone": ["finerenone"],
}
GLP1_DRUGS = CLASSES["GLP-1"]


def drug_class(generic):
    g = str(generic).lower()
    for cls, keys in CLASSES.items():
        if any(k in g for k in keys):
            return cls
    return None


def load():
    files = sorted(glob.glob(os.path.join(RAW, "partd_nephrology_*.csv")))
    if not files:
        raise SystemExit("No data yet - run: python pull_prescriptions.py")
    frames = []
    for f in files:
        year = int(f.rsplit("_", 1)[1].split(".")[0])
        df = pd.read_csv(f, dtype={"Prscrbr_NPI": str})
        df["year"] = year
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["drug_class"] = df["Gnrc_Name"].map(drug_class)
    return df


def adoption_table(df):
    rows = []
    for year, g in df.groupby("year"):
        n_docs = g["Prscrbr_NPI"].nunique()
        row = {"year": year, "nephrologists": n_docs}
        for cls in CLASSES:
            users = g.loc[g["drug_class"] == cls, "Prscrbr_NPI"].nunique()
            row[f"{cls}_prescribers"] = users
            row[f"{cls}_pct"] = round(100 * users / n_docs, 2)
            row[f"{cls}_claims"] = int(g.loc[g["drug_class"] == cls, "Tot_Clms"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def glp1_by_drug(df):
    g = df[df["drug_class"] == "GLP-1"].copy()
    g["drug"] = g["Gnrc_Name"].str.lower().map(
        lambda s: next(d for d in GLP1_DRUGS if d in s))
    return g.pivot_table(index="year", columns="drug", values="Tot_Clms",
                         aggfunc="sum", fill_value=0).astype(int)


def glp1_by_state(df, year=2024):
    g = df[df["year"] == year]
    docs = g.groupby("Prscrbr_State_Abrvtn")["Prscrbr_NPI"].nunique()
    users = g[g["drug_class"] == "GLP-1"].groupby("Prscrbr_State_Abrvtn")["Prscrbr_NPI"].nunique()
    t = pd.DataFrame({"nephrologists": docs, "glp1_prescribers": users}).fillna(0)
    t["glp1_pct"] = (100 * t["glp1_prescribers"] / t["nephrologists"]).round(1)
    return t[t["nephrologists"] >= 20].sort_values("glp1_pct")   # skip tiny states


def plot(adopt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"GLP-1": "#2471a3", "SGLT2": "#27ae60", "Finerenone": "#8e44ad"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cls in CLASSES:
        ax.plot(adopt["year"], adopt[f"{cls}_pct"], marker="o", lw=2.4,
                color=colors[cls], label=cls)
        last = adopt.iloc[-1]
        ax.annotate(f"{last[f'{cls}_pct']:.1f}%", (last["year"], last[f"{cls}_pct"]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=colors[cls], fontweight="bold")
    ax.axvspan(2023.75, 2024.5, color="grey", alpha=0.12)
    ax.text(2023.8, ax.get_ylim()[0] + 0.5, "FLOW trial stopped\nfor efficacy (Oct 2023)",
            fontsize=8.5, color="dimgray", va="bottom")
    ax.set_xticks(adopt["year"])
    ax.set_xlim(adopt["year"].min() - 0.3, adopt["year"].max() + 0.5)
    ax.set_ylabel("% of Medicare nephrologists writing 11+ prescriptions")
    ax.set_title("Nephrologists' adoption of kidney-protective drugs, 2019-2024",
                 fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "adoption_chart.png"), dpi=200)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    adopt = adoption_table(df)
    by_drug = glp1_by_drug(df)
    by_state = glp1_by_state(df, year=df["year"].max())
    adopt.to_csv(os.path.join(OUT, "adoption_by_year.csv"), index=False)
    by_drug.to_csv(os.path.join(OUT, "glp1_by_drug.csv"))
    by_state.to_csv(os.path.join(OUT, f"glp1_by_state_{df['year'].max()}.csv"))
    plot(adopt)

    pd.set_option("display.width", 140)
    cols = ["year", "nephrologists"] + [f"{c}_pct" for c in CLASSES]
    print("\n% of nephrologists prescribing each class (11+ claims):\n")
    print(adopt[cols].to_string(index=False))
    print("\nGLP-1 claims by drug:\n")
    print(by_drug.to_string())
    print(f"\nLowest / highest GLP-1 adoption by state ({df['year'].max()}, states with 20+ nephrologists):")
    print(by_state.head(5).to_string())
    print(by_state.tail(5).to_string())
    print("\nSaved charts and tables to output/")


if __name__ == "__main__":
    main()
