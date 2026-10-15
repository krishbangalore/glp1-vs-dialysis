# Ozempic vs. Dialysis

What do GLP-1 drugs mean for kidney failure? A three-part analysis on public data:
how the market reacted, how nephrologists responded, and what the gap is worth.

## Findings

**1. Market signal.** When Novo Nordisk stopped its FLOW kidney trial early for efficacy
(announced after the close on Oct 10, 2023), DaVita fell 17.1% and Fresenius 18.0% beyond
what the market explains on the next trading day; Novo rose 5.9% (all |t| > 3).
The result holds with a second price source (Tiingo) and Fama-French benchmarks.

![Event study](output/car_chart.png)

**2. Prescribing gap.** By 2024, 71.8% of Medicare nephrologists prescribed SGLT2 inhibitors
but only 12.8% prescribed GLP-1s. A model trained on 2023 adopters predicted 2024 adopters
with AUC 0.79 (3.9x lift in the top decile). Dialysis-focused practices were about 40% less
likely to start per standard deviation of dialysis-drug share.

![Adoption](output/adoption_chart.png)
![Model](output/model_performance.png)

**3. Value.** A Markov cohort model with Monte Carlo uncertainty (10,000 draws) puts
semaglutide at about $55,000 per discounted life-year over 10 years at Medicare's negotiated
price ($274/month), versus about $230,000 at list price. Dialysis savings cover roughly a
fifth of the drug's cost.

![Simulation](output/sim_chart.png)

## Repository

| Script | Part | What it does |
| --- | --- | --- |
| `event_study.py` | 1 | Market-model event study (downloads prices via `yfinance`) |
| `robustness.py` | 1 | Tiingo price cross-check; Fama-French market and 3-factor benchmarks |
| `pull_prescriptions.py` | 2 | Downloads CMS Part D prescriptions by nephrologists, 2019-2024 |
| `adoption_summary.py` | 2 | Adoption trends by drug class, drug and state |
| `pull_providers.py` | 2 | Downloads CMS per-prescriber summary files, 2022-2024 |
| `model.py` | 2 | Logistic regression and random forest, validated forward in time |
| `simulate.py` | 3 | Cost-effectiveness simulation from published inputs |

Results are in `output/`. Raw data is not committed; the scripts re-download it.

## Reproduce

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python event_study.py
export TIINGO_API_KEY="your_key"   # free at tiingo.com
python robustness.py
python pull_prescriptions.py && python adoption_summary.py
python pull_providers.py && python model.py
python simulate.py
```

## Data sources

- Centers for Medicare & Medicaid Services, *Medicare Part D Prescribers - by Provider and Drug* and *by Provider*, 2019-2024, data.cms.gov.
- Daily adjusted prices: Yahoo Finance (`yfinance`), cross-checked with Tiingo.
- Kenneth French Data Library, Fama-French 3 factors (daily).
- Perkovic et al., "Effects of Semaglutide on Chronic Kidney Disease in Patients with Type 2 Diabetes," *NEJM* 2024 (FLOW).
- United States Renal Data System, *2024 Annual Data Report*, chapters 6 and 9.
- CMS negotiated Medicare prices for 2027 (announced Nov 2025).

## Limitations

- CMS hides doctor-drug pairs with fewer than 11 claims, so adoption is understated (equally across drug classes).
- Medicare only; GLP-1s may be prescribed by a patient's other doctors.
- The FLOW dialysis hazard ratio's 95% CI crosses 1; the simulation samples that uncertainty.
- The 10-year horizon extrapolates beyond the 3.4-year trial (trial-length result: ~$173,000 per life-year).
- The cost model omits cardiovascular savings (understating value) and other costs of longer survival (overstating it).
