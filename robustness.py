"""
Part 1 robustness check - do the event-study results hold up with
  (a) a second, independent price source (Tiingo instead of Yahoo Finance), and
  (b) academic benchmarks from the Kenneth French Data Library
      (the Fama-French market return, and the Fama-French 3-factor model)?

Setup:  export TIINGO_API_KEY="your_key"   (free account at tiingo.com)
Run:    python robustness.py

Outputs (./output):
  prices_tiingo.csv            Tiingo adjusted closes (cached)
  ff_factors_daily.csv         Fama-French daily factors (cached)
  price_source_comparison.csv  Yahoo vs Tiingo daily returns, per ticker
  robustness_results.csv       CARs and t-stats under each specification
"""

import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

import event_study as es

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
START, END = "2022-08-01", "2024-02-29"
SYMBOLS = list(es.TICKERS) + [es.MARKET]
FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_daily_CSV.zip")
WINDOWS = {"day 0": (0, 0), "days 0 to +1": (0, 1)}   # the clean, headline windows


# ----------------------------------------------------------------------------- data
def tiingo_prices():
    cache = os.path.join(OUT, "prices_tiingo.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    key = os.environ.get("TIINGO_API_KEY")
    if not key:
        sys.exit('Set your key first:  export TIINGO_API_KEY="your_key"')
    cols = {}
    for s in SYMBOLS:
        r = requests.get(f"https://api.tiingo.com/tiingo/daily/{s.lower()}/prices",
                         params={"startDate": START, "endDate": END, "token": key}, timeout=60)
        if r.status_code != 200:
            sys.exit(f"Tiingo error for {s}: HTTP {r.status_code} - {r.text[:200]}")
        data = r.json()
        if not data:
            sys.exit(f"Tiingo returned no data for {s}")
        ser = pd.Series({pd.Timestamp(d["date"][:10]): d["adjClose"] for d in data})
        cols[s] = ser
        print(f"  Tiingo {s}: {len(ser)} days")
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "Date"
    df.to_csv(cache)
    return df


def french_factors():
    """Fama-French daily factors (Mkt-RF, SMB, HML, RF), converted from % to decimals."""
    cache = os.path.join(OUT, "ff_factors_daily.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    r = requests.get(FF_URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and parts[0].isdigit() and len(parts[0]) == 8:
            rows.append(parts[:5])
    ff = pd.DataFrame(rows, columns=["Date", "Mkt-RF", "SMB", "HML", "RF"])
    ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m%d")
    ff = ff.set_index("Date").astype(float) / 100
    ff = ff.loc[START:END]
    ff.to_csv(cache)
    return ff


# --------------------------------------------------------------------- event study
def car_table(stock_rets, factors, label):
    """Generic event study: regress each stock's (excess) return on the factor columns
    over the estimation window, then sum abnormal returns over each event window."""
    dates = stock_rets.index
    ev = dates.searchsorted(pd.Timestamp(es.EVENT_DATE))
    rel = pd.Series(np.arange(len(dates)) - ev, index=dates)
    est = (rel >= es.EST_START) & (rel <= es.EST_END)
    rows = []
    for t in es.TICKERS:
        y = stock_rets[t]
        X = factors.reindex(dates)
        ok = est & y.notna() & X.notna().all(axis=1)
        A = np.column_stack([np.ones(ok.sum()), X[ok].values])
        coef, *_ = np.linalg.lstsq(A, y[ok].values, rcond=None)
        resid = y[ok].values - A @ coef
        sd = resid.std(ddof=A.shape[1])
        ar = y - (coef[0] + X.values @ coef[1:])
        ar_rel = pd.Series(ar.values, index=rel.values)
        for w, (a, b) in WINDOWS.items():
            if t in es.CONFOUND_DAY and b >= es.CONFOUND_DAY[t]:
                continue
            win = ar_rel.loc[a:b].dropna()
            car = win.sum()
            rows.append({"specification": label, "ticker": t, "window": w,
                         "CAR_pct": round(car * 100, 2),
                         "t_stat": round(car / (sd * np.sqrt(len(win))), 2)})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    yahoo = pd.read_csv(os.path.join(OUT, "prices.csv"), index_col=0, parse_dates=True)
    print("Downloading Tiingo prices and Fama-French factors (cached after the first run)...")
    tiingo = tiingo_prices()
    ff = french_factors()

    # (1) compare the two price sources day by day
    ry, rt = yahoo[SYMBOLS].pct_change(), tiingo[SYMBOLS].pct_change()
    common = ry.index.intersection(rt.index)
    comp = []
    for s in SYMBOLS:
        a, b = ry.loc[common, s], rt.loc[common, s]
        ok = a.notna() & b.notna()
        diff = (a[ok] - b[ok]).abs() * 100
        comp.append({"ticker": s, "days_compared": int(ok.sum()),
                     "return_correlation": round(a[ok].corr(b[ok]), 5),
                     "median_abs_diff_pct_points": round(diff.median(), 4),
                     "max_abs_diff_pct_points": round(diff.max(), 3),
                     "event_day_return_yahoo_pct": round(a.get(pd.Timestamp(es.EVENT_DATE)) * 100, 2),
                     "event_day_return_tiingo_pct": round(b.get(pd.Timestamp(es.EVENT_DATE)) * 100, 2)})
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(OUT, "price_source_comparison.csv"), index=False)

    # (2) event study under four specifications
    ry = ry.dropna(how="all")
    rt = rt.dropna(how="all")
    ff_mkt = (ff["Mkt-RF"] + ff["RF"]).rename("Mkt")
    rows = []
    rows += car_table(ry, ry[[es.MARKET]], "Yahoo prices, market model (SPY) [original]")
    rows += car_table(rt, rt[[es.MARKET]], "Tiingo prices, market model (SPY)")
    rows += car_table(ry, ff_mkt.to_frame(), "Yahoo prices, market model (Fama-French market)")
    excess = ry[list(es.TICKERS)].sub(ff["RF"], axis=0)
    rows += car_table(excess, ff[["Mkt-RF", "SMB", "HML"]], "Yahoo prices, Fama-French 3-factor")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "robustness_results.csv"), index=False)

    pd.set_option("display.width", 160)
    print("\n(1) Yahoo vs Tiingo daily returns:\n")
    print(comp.to_string(index=False))
    print("\n(2) Cumulative abnormal return (%) under each specification:\n")
    print(res.pivot_table(index=["specification"], columns=["ticker", "window"],
                          values="CAR_pct").to_string())
    print("\n    t-stats:\n")
    print(res.pivot_table(index=["specification"], columns=["ticker", "window"],
                          values="t_stat").to_string())
    print("\nSaved: output/price_source_comparison.csv, output/robustness_results.csv")


if __name__ == "__main__":
    main()
