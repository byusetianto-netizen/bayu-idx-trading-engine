from pathlib import Path
import numpy as np
import pandas as pd

PRICE = Path("data/idx_stock_prices.csv")
PHASE = Path("data/broker/flow_phase_features.csv")
OUT = Path("data/broker/v20_validation_report.csv")

def future_return(df, n):
    return df.groupby("ticker")["close"].shift(-n) / df["close"] - 1

def bootstrap_ci(x, n=2000, seed=42):
    x = pd.Series(x).dropna().to_numpy()
    if len(x) < 20:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))

def main():
    p = pd.read_csv(PRICE)
    p.columns = [c.lower() for c in p.columns]
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["ticker","date"])
    p["fwd20"] = future_return(p, 20)
    p["fwd60"] = future_return(p, 60)

    ph = pd.read_csv(PHASE)
    ph["date"] = pd.to_datetime(ph["date"])
    df = p.merge(ph, on=["ticker","date"], how="inner")

    rows = []
    for phase, g in df.groupby("flow_phase"):
        for horizon in [20, 60]:
            x = g[f"fwd{horizon}"].dropna()
            lo, hi = bootstrap_ci(x)
            rows.append({
                "phase": phase,
                "horizon_sessions": horizon,
                "n": len(x),
                "avg_return": x.mean(),
                "median_return": x.median(),
                "win_rate": (x > 0).mean(),
                "ci95_low": lo,
                "ci95_high": hi,
            })

    # Time split report, not parameter optimization.
    df["year"] = df["date"].dt.year
    for year, gy in df.groupby("year"):
        for phase, g in gy.groupby("flow_phase"):
            x = g["fwd20"].dropna()
            if len(x) >= 20:
                rows.append({
                    "phase": phase,
                    "horizon_sessions": 20,
                    "year": year,
                    "n": len(x),
                    "avg_return": x.mean(),
                    "median_return": x.median(),
                    "win_rate": (x > 0).mean(),
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

if __name__ == "__main__":
    main()
