"""
Part 3 - What is the GLP-1 adoption gap worth? (cost-effectiveness simulation)

Model: a Markov cohort model with three health states, run in 1-year cycles:
    CKD (not yet on dialysis)  ->  Dialysis  ->  Dead
                 \\-------------------------->  Dead
Two arms of 1,000 patients each: standard care vs. standard care + semaglutide.

Monte Carlo (probabilistic sensitivity analysis): every uncertain input is drawn
10,000 times from a distribution centred on its published value, and the model is
re-run for each draw. That turns single-number answers into ranges.

EVERY INPUT IS FROM A PUBLISHED SOURCE (see INPUTS below). No made-up numbers.
Patient counts for the scale-up come from the real CMS Part D files in data/raw/.

What the model leaves out (stated so nobody over-reads it):
  - Savings from fewer heart attacks/strokes (FLOW showed fewer), so savings are understated.
  - Kidney transplants (treated as dialysis-level cost), side-effect costs, other CKD care costs.
  - It assumes the trial's effect continues beyond the trial's 3.4 years (we also report a
    trial-length-only horizon).

Outputs (./output): sim_results.csv, sim_summary.txt, sim_chart.png
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "output")
SEED, N_DRAWS, COHORT = 42, 10_000, 1_000

# ------------------------------------------------------------------------------------
# INPUTS - each with its source
# ------------------------------------------------------------------------------------
INPUTS = {
    # FLOW trial: Perkovic et al., N Engl J Med 2024;391:109-121 (placebo arm, n=1,766)
    "placebo_patient_years": 410 / 0.075,   # 410 primary events at 7.5 per 100 patient-years
    "placebo_dialysis_starts": 100,         # kidney-replacement therapy initiated, placebo arm
    "placebo_deaths": 279,                  # death from any cause, placebo arm
    "hr_dialysis": (0.84, 0.63, 1.12),      # semaglutide vs placebo, KRT initiation (HR, 95% CI)
    "hr_death": (0.80, 0.67, 0.95),         # semaglutide vs placebo, all-cause death (HR, 95% CI)
    # USRDS 2024 Annual Data Report, ESRD ch. 6: 5-year survival after starting
    # hemodialysis = 40.7% (2018 cohort)  -> annual death rate = -ln(0.407)/5
    "dialysis_5yr_survival": 0.407,
    # USRDS 2024 ADR, ESRD ch. 9: Medicare FFS spending per person per year,
    # in-center hemodialysis, 2022 dollars
    "dialysis_cost_per_year": 99_369,
    # CMS negotiated Medicare price for Ozempic (effective 2027): $274 per month
    "drug_cost_per_year_negotiated": 274 * 12,
    # Ozempic list price at the time of the negotiation announcement: $959 per month
    "drug_cost_per_year_list": 959 * 12,
    "discount_rate": 0.03,                  # standard in US cost-effectiveness analysis (costs AND life-years)
}
HORIZONS = {"10-year": 10, "Trial length (3.4 years)": 3.4}


def rate_to_prob(rate, years=1.0):
    return 1 - np.exp(-rate * years)


def draw_params(rng, n):
    """Draw n sets of inputs. Event rates ~ Gamma(events, patient-years); hazard ratios
    ~ log-normal using the published 95% CI; dialysis survival ~ Beta around 40.7%."""
    py = INPUTS["placebo_patient_years"]

    def hr(hr_tuple):
        est, lo, hi = hr_tuple
        se = (np.log(hi) - np.log(lo)) / (2 * 1.96)
        return np.exp(rng.normal(np.log(est), se, n))

    surv = rng.beta(407, 593, n)            # centred on 40.7%, ~1,000-patient precision
    return {
        "dialysis_rate": rng.gamma(INPUTS["placebo_dialysis_starts"], 1 / py, n),
        "death_rate_ckd": rng.gamma(INPUTS["placebo_deaths"], 1 / py, n),
        "hr_dialysis": hr(INPUTS["hr_dialysis"]),
        "hr_death": hr(INPUTS["hr_death"]),
        "death_rate_dialysis": -np.log(surv) / 5,
    }


def run_cohort(p, treat, horizon, drug_cost):
    """Run both arms for all parameter draws at once (vectorised). Returns per-arm totals."""
    n = len(p["dialysis_rate"])
    ckd = np.full(n, float(COHORT))
    dial = np.zeros(n)
    totals = {k: np.zeros(n) for k in ["dialysis_starts", "dialysis_years", "life_years",
                                       "drug_years", "dialysis_cost", "drug_cost"]}
    r_dial = p["dialysis_rate"] * (p["hr_dialysis"] if treat else 1)
    r_death = p["death_rate_ckd"] * (p["hr_death"] if treat else 1)
    t = 0.0
    while t < horizon - 1e-9:
        step = min(1.0, horizon - t)
        disc = 1 / (1 + INPUTS["discount_rate"]) ** t
        # competing risks within the cycle (CKD -> dialysis or death)
        total_rate = r_dial + r_death
        leave = ckd * rate_to_prob(total_rate, step)
        to_dial = leave * r_dial / total_rate
        die_dial = dial * rate_to_prob(p["death_rate_dialysis"], step)
        # half-cycle correction: people who move count as half a cycle in the old state
        ckd_time = (ckd - leave / 2) * step
        dial_time = (dial - die_dial / 2 + to_dial / 2) * step
        totals["dialysis_starts"] += to_dial
        totals["dialysis_years"] += dial_time
        totals["life_years"] += (ckd_time + dial_time) * disc      # discounted, like costs
        totals["drug_years"] += ckd_time if treat else 0
        totals["dialysis_cost"] += dial_time * INPUTS["dialysis_cost_per_year"] * disc
        totals["drug_cost"] += (ckd_time * drug_cost * disc) if treat else 0
        ckd, dial = ckd - leave, dial + to_dial - die_dial
        t += step
    return totals


def simulate(horizon, drug_cost, rng):
    p = draw_params(rng, N_DRAWS)
    base, sema = run_cohort(p, False, horizon, drug_cost), run_cohort(p, True, horizon, drug_cost)
    d = pd.DataFrame({
        "dialysis_starts_avoided": base["dialysis_starts"] - sema["dialysis_starts"],
        "dialysis_years_avoided": base["dialysis_years"] - sema["dialysis_years"],
        "life_years_gained": sema["life_years"] - base["life_years"],
        "drug_cost": sema["drug_cost"],
        "dialysis_cost_saved": base["dialysis_cost"] - sema["dialysis_cost"],
    })
    d["net_cost"] = d["drug_cost"] - d["dialysis_cost_saved"]
    d["cost_per_life_year"] = d["net_cost"] / d["life_years_gained"]
    # price at which dialysis savings exactly pay for the drug
    d["break_even_drug_price_per_year"] = drug_cost * d["dialysis_cost_saved"] / d["drug_cost"]
    return d


def patient_count():
    """Diabetic patients managed by nephrologists who prescribed no GLP-1 in 2024 (real CMS data).
    A doctor's count = the largest patient count on any of their diabetes-drug lines (a lower
    bound, since patients overlap across drugs). CMS hides counts under 11, so a doctor whose
    counts are all hidden contributes 1-10: we report that range."""
    glp1 = ["semaglutide", "tirzepatide", "dulaglutide", "liraglutide", "exenatide", "lixisenatide"]
    dm = ["metformin", "insulin", "gliflozin", "gliptin", "glipizide", "glimepiride", "glyburide"]
    d = pd.read_csv(os.path.join(RAW, "partd_nephrology_2024.csv"), dtype={"Prscrbr_NPI": str})
    g = d["Gnrc_Name"].str.lower()
    users = set(d.loc[g.map(lambda s: any(k in s for k in glp1)), "Prscrbr_NPI"])
    non = d[~d["Prscrbr_NPI"].isin(users) & g.map(lambda s: any(k in s for k in dm))]
    per_doc = non.groupby("Prscrbr_NPI")["Tot_Benes"].max()
    known = per_doc.dropna()
    hidden = per_doc.isna().sum()
    return {"doctors": len(per_doc), "known": int(known.sum()),
            "low": int(known.sum() + hidden * 1), "high": int(known.sum() + hidden * 10)}


def summarise(d):
    q = d.quantile([0.05, 0.5, 0.95]).T
    q.columns = ["p5", "median", "p95"]
    q["mean"] = d.mean()
    return q


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, lines, chart_data = [], [], {}
    for price_name, price in [("Negotiated Medicare price ($274/mo)", INPUTS["drug_cost_per_year_negotiated"]),
                              ("List price ($959/mo)", INPUTS["drug_cost_per_year_list"])]:
        for hname, h in HORIZONS.items():
            d = simulate(h, price, rng)
            s = summarise(d)
            if hname == "10-year":
                chart_data[price_name] = d
            for metric, r in s.iterrows():
                rows.append({"drug_price": price_name, "horizon": hname, "metric": metric,
                             **r.round(2).to_dict()})
            p_any_benefit = (d["dialysis_starts_avoided"] > 0).mean()
            p_saves = (d["net_cost"] < 0).mean()
            p_100k = (d["cost_per_life_year"] < 100_000).mean()
            lines += [f"\n=== {price_name} | {hname} | per 1,000 patients ===",
                      f"Dialysis starts avoided:   {s.loc['dialysis_starts_avoided','median']:.1f}  "
                      f"(90% range {s.loc['dialysis_starts_avoided','p5']:.1f} to {s.loc['dialysis_starts_avoided','p95']:.1f})",
                      f"Dialysis-years avoided:    {s.loc['dialysis_years_avoided','median']:.1f}  "
                      f"(90% range {s.loc['dialysis_years_avoided','p5']:.1f} to {s.loc['dialysis_years_avoided','p95']:.1f})",
                      f"Life-years gained:         {s.loc['life_years_gained','median']:.1f}  "
                      f"(90% range {s.loc['life_years_gained','p5']:.1f} to {s.loc['life_years_gained','p95']:.1f})",
                      f"Drug cost:                 ${s.loc['drug_cost','median']/1e6:.1f}M",
                      f"Dialysis cost saved:       ${s.loc['dialysis_cost_saved','median']/1e6:.2f}M  "
                      f"(90% range ${s.loc['dialysis_cost_saved','p5']/1e6:.2f}M to ${s.loc['dialysis_cost_saved','p95']/1e6:.2f}M)",
                      f"Net cost to Medicare:      ${s.loc['net_cost','median']/1e6:.1f}M",
                      f"Net cost per life-year:    ${s.loc['cost_per_life_year','median']:,.0f}  "
                      f"(90% range ${s.loc['cost_per_life_year','p5']:,.0f} to ${s.loc['cost_per_life_year','p95']:,.0f})",
                      f"Break-even drug price:     ${s.loc['break_even_drug_price_per_year','median']:,.0f}/yr "
                      f"(${s.loc['break_even_drug_price_per_year','median']/12:,.0f}/mo)",
                      f"P(drug avoids any dialysis starts) = {p_any_benefit:.0%};  "
                      f"P(pays for itself on dialysis alone) = {p_saves:.0%};  "
                      f"P(under $100k per life-year) = {p_100k:.0%}"]

    pc = patient_count()
    neg = chart_data["Negotiated Medicare price ($274/mo)"]
    scale = lambda col, n: neg[col].median() * n / COHORT
    lines += ["\n=== Scale-up: the 2024 adoption gap (real CMS Part D data) ===",
              f"{pc['doctors']:,} nephrologists prescribed diabetes drugs but no GLP-1s in 2024.",
              f"Their diabetic Medicare patients: {pc['low']:,} to {pc['high']:,} "
              f"({pc['known']:,} directly counted; the rest hidden by CMS small-count rules).",
              f"If those patients were treated (negotiated price, 10 years), the median estimate is:",
              f"  dialysis starts avoided: {scale('dialysis_starts_avoided', pc['low']):,.0f} to "
              f"{scale('dialysis_starts_avoided', pc['high']):,.0f}",
              f"  life-years gained:       {scale('life_years_gained', pc['low']):,.0f} to "
              f"{scale('life_years_gained', pc['high']):,.0f}",
              f"  dialysis spending saved: ${scale('dialysis_cost_saved', pc['low'])/1e6:,.0f}M to "
              f"${scale('dialysis_cost_saved', pc['high'])/1e6:,.0f}M",
              "Caveat: some of these patients may already get a GLP-1 from another doctor, and not all",
              "match the FLOW trial population (type 2 diabetes + CKD), so treat this as an upper-end",
              "illustration of the gap, not a forecast."]

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sim_results.csv"), index=False)
    text = "\n".join(lines)
    with open(os.path.join(OUT, "sim_summary.txt"), "w") as f:
        f.write("INPUTS (all published):\n")
        for k, v in INPUTS.items():
            f.write(f"  {k}: {v}\n")
        f.write(text + "\n")
    plot(chart_data)
    print(text)
    print("\nSaved: output/sim_results.csv, output/sim_summary.txt, output/sim_chart.png")


def plot(chart_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    neg = chart_data["Negotiated Medicare price ($274/mo)"]
    lst = chart_data["List price ($959/mo)"]

    ax = axes[0]
    ax.hist(neg["dialysis_years_avoided"], bins=60, color="#2471a3", alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(neg["dialysis_years_avoided"].median(), color="#c0392b", ls="--",
               label=f"median {neg['dialysis_years_avoided'].median():.0f}")
    ax.set(xlabel="Dialysis-years avoided per 1,000 patients (10 years)", ylabel="Simulation runs",
           title="Uncertainty in dialysis avoided\n(trial CI crosses 'no effect')")
    ax.legend(frameon=False)

    ax = axes[1]
    for d, name, c in [(neg, "Negotiated $274/mo", "#27ae60"), (lst, "List $959/mo", "#c0392b")]:
        s = d.sample(2000, random_state=SEED)
        ax.scatter(s["life_years_gained"], s["net_cost"] / 1e6, s=4, alpha=0.35, color=c, label=name)
    xs = np.linspace(0, max(neg["life_years_gained"].quantile(0.99), 1), 50)
    ax.plot(xs, xs * 100_000 / 1e6, color="black", ls=":", lw=1, label="$100k per life-year line")
    ax.axhline(0, color="black", lw=0.6)
    ax.set(xlabel="Life-years gained per 1,000 patients", ylabel="Net cost to Medicare ($M)",
           title="Cost vs. benefit, 10,000 simulation runs")
    ax.legend(frameon=False, fontsize=8, markerscale=4)

    ax = axes[2]
    vals = [neg["drug_cost"].median() / 1e6, neg["dialysis_cost_saved"].median() / 1e6,
            lst["drug_cost"].median() / 1e6]
    bars = ax.bar(["Drug cost\n(negotiated)", "Dialysis\nspending saved", "Drug cost\n(list price)"],
                  vals, color=["#27ae60", "#2471a3", "#c0392b"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"${v:.1f}M", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="$M per 1,000 patients over 10 years (discounted)",
           title="Dialysis savings vs. what the drug costs")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sim_chart.png"), dpi=180)


if __name__ == "__main__":
    main()
