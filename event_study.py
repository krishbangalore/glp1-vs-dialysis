"""
Part 1 - Event study: how did dialysis stocks react when Novo Nordisk stopped
its FLOW kidney trial early for efficacy?

Novo announced after the market close on Tue Oct 10, 2023, so the first trading
day that could react is Wed Oct 11, 2023 (day 0).

Method (standard market-model event study):
  1. Estimation window: trading days -250 to -30 before the event. For each stock,
     fit  stock_return = alpha + beta * market_return  (market = S&P 500 via SPY).
  2. Abnormal return (AR) on each event-window day = actual return - predicted return.
     "How much did the stock move beyond what the market alone explains?"
  3. Cumulative abnormal return (CAR) = sum of ARs over a window.
  4. t-stat for a CAR over L days = CAR / (residual std dev * sqrt(L)).
     |t| above ~2 means the move is very unlikely to be normal noise.

Outputs (in ./output):
  event_study_results.csv   CARs and t-stats per stock and window
  daily_abnormal_returns.csv
  car_chart.png             cumulative abnormal return from day -10 to +60
"""

import os
import sys

import numpy as np
import pandas as pd

EVENT_DATE = "2023-10-11"
TICKERS = {
    "DVA": "DaVita (dialysis clinics)",
    "FMS": "Fresenius Medical Care (dialysis clinics + machines)",
    "OM": "Outset Medical (home dialysis machines)",
    "NVO": "Novo Nordisk (makes Ozempic) - contrast",
}
MARKET = "SPY"
EST_START, EST_END = -250, -30           # estimation window (trading days)
WINDOWS = {"day 0": (0, 0), "days 0 to +1": (0, 1), "days -1 to +5": (-1, 5),
           "days 0 to +30": (0, 30), "days 0 to +60": (0, 60)}
PLOT_RANGE = (-10, 60)
# First trading day contaminated by unrelated company news (excluded from that point on)
CONFOUND_DAY = {"OM": 2}   # Outset preannounced weak Q3 results after close Oct 12, 2023

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def load_prices():
    """Download adjusted daily closes (cached to output/prices.csv after first run)."""
    cache = os.path.join(OUT, "prices.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    import yfinance as yf
    syms = list(TICKERS) + [MARKET]
    df = yf.download(syms, start="2022-08-01", end="2024-03-01",
                     auto_adjust=True, progress=False)["Close"]
    df = df.dropna(how="all")
    missing = [s for s in syms if s not in df or df[s].isna().all()]
    if missing:
        sys.exit(f"No price data for {missing} - check your internet connection and retry.")
    df.to_csv(cache)
    return df


def run(prices):
    rets = prices.pct_change().dropna(how="all")
    dates = rets.index
    event_idx = dates.searchsorted(pd.Timestamp(EVENT_DATE))
    if dates[event_idx] != pd.Timestamp(EVENT_DATE):
        sys.exit(f"{EVENT_DATE} is not a trading day in the data")
    rel = np.arange(len(dates)) - event_idx      # trading-day offset from event

    results, daily = [], {}
    mkt = rets[MARKET]
    for t, label in TICKERS.items():
        est = (rel >= EST_START) & (rel <= EST_END)
        y, x = rets[t][est], mkt[est]
        ok = y.notna() & x.notna()
        beta, alpha = np.polyfit(x[ok], y[ok], 1)
        resid_sd = (y[ok] - (alpha + beta * x[ok])).std(ddof=2)

        ar = rets[t] - (alpha + beta * mkt)
        ar_series = pd.Series(ar.values, index=rel)
        daily[t] = ar_series.loc[PLOT_RANGE[0]:PLOT_RANGE[1]]

        for wname, (a, b) in WINDOWS.items():
            if t in CONFOUND_DAY and b >= CONFOUND_DAY[t]:
                results.append({"ticker": t, "company": label, "window": wname,
                                "CAR_pct": np.nan, "t_stat": np.nan, "beta": round(beta, 2),
                                "n_days": 0})
                continue                         # window overlaps unrelated company news
            window = ar_series.loc[a:b].dropna()
            car = window.sum()
            t_stat = car / (resid_sd * np.sqrt(len(window)))
            results.append({"ticker": t, "company": label, "window": wname,
                            "CAR_pct": round(car * 100, 2), "t_stat": round(t_stat, 2),
                            "beta": round(beta, 2), "n_days": len(window)})
    return pd.DataFrame(results), pd.DataFrame(daily)


def plot(daily):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    car = daily.fillna(0).cumsum() * 100
    car = car - car.loc[-1]                       # anchor at 0 the day before the event
    # Outset issued its own earnings warning after the close on Oct 12, 2023 (day +2),
    # so its returns after day +1 reflect company news, not the GLP-1 event.
    if "OM" in car:
        car.loc[CONFOUND_DAY["OM"]:, "OM"] = np.nan
    colors = {"DVA": "#c0392b", "FMS": "#e67e22", "OM": "#8e44ad", "NVO": "#2471a3"}
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for t in car:
        ax.plot(car.index, car[t], label=f"{t} - {TICKERS[t].split(' (')[0]}",
                color=colors.get(t), lw=2.2 if t != "NVO" else 1.6,
                ls="-" if t != "NVO" else "--")
    ax.axvline(0, color="grey", lw=1)
    ax.axhline(0, color="black", lw=0.6)
    ax.annotate("Novo stops FLOW trial early\n(announced after close, Oct 10 2023)",
                xy=(0, 0), xytext=(1.5, car.max().max() * 0.9), fontsize=9, color="dimgray")
    if "OM" in car:
        ax.annotate("Outset cut off after day +1:\nits own earnings warning (Oct 12)",
                    xy=(1, car.loc[1, "OM"]), xytext=(4, car.loc[1, "OM"] - 2),
                    fontsize=8, color="#8e44ad")
    ax.set_xlabel("Trading days relative to Oct 11, 2023")
    ax.set_ylabel("Cumulative abnormal return (%)\n(move beyond what the S&P 500 explains)")
    ax.set_title("Dialysis stocks vs. the Ozempic kidney trial news", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUT, "car_chart.png")
    fig.savefig(path, dpi=200)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    prices = load_prices()
    results, daily = run(prices)
    results.to_csv(os.path.join(OUT, "event_study_results.csv"), index=False)
    daily.to_csv(os.path.join(OUT, "daily_abnormal_returns.csv"))
    chart = plot(daily)

    pd.set_option("display.width", 120)
    print("\nCumulative abnormal returns (CAR) - |t| > 2 is statistically significant\n")
    print(results.pivot(index="window", columns="ticker", values="CAR_pct")
          .loc[list(WINDOWS)].to_string())
    print("\nt-stats:")
    print(results.pivot(index="window", columns="ticker", values="t_stat")
          .loc[list(WINDOWS)].to_string())
    print("\n(blank = excluded: window overlaps Outset's own Oct 12 earnings warning)")
    print(f"\nSaved: output/event_study_results.csv, output/car_chart.png")


if __name__ == "__main__":
    main()
