from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd
import yfinance as yf


# ------------------ config ------------------

DATA_DIR = Path("data")
RAW, OUT = DATA_DIR / "raw", DATA_DIR / "processed"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2000-01-01")
END = pd.Timestamp("2025-12-31")
EFFECTIVE_END = min(END, pd.Timestamp.today().normalize())
TOL = 0.05  # split confirmation tolerance


# ------------------ helpers ------------------

def norm_ticker(x) -> str:
    return str(x).strip().upper()


def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def batch(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def quality(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name} ===")
    print("rows:", len(df), "| tickers:", df["ticker"].nunique() if "ticker" in df else "n/a")
    if "date" in df:
        print("range:", df["date"].min(), "->", df["date"].max())
    if {"open", "close"}.issubset(df.columns):
        print("missing open:", df["open"].isna().mean(), "| missing close:", df["close"].isna().mean())
    if {"ticker", "date"}.issubset(df.columns):
        print("dup(ticker,date):", df.duplicated(["ticker", "date"]).sum())


# ------------------ loaders/transformers ------------------


def load_price_dir(dirpath: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(dirpath.glob("*.csv")):
        d = clean_cols(pd.read_csv(f))
        if "date" not in d.columns:
            continue

        if "ticker" not in d.columns:
            d["ticker"] = f.stem

        d["ticker"] = d["ticker"].map(norm_ticker)
        d["date"] = to_date(d["date"])

        for c in ("open", "close"):
            d[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan

        d = (
            d.dropna(subset=["ticker", "date"])
            .loc[lambda x: (x["date"] >= START) & (x["date"] <= END), ["ticker", "date", "open", "close"]]
            .drop_duplicates(["ticker", "date"])
        )
        frames.append(d)

    if not frames:
        raise RuntimeError(f"No usable CSVs in {dirpath}")

    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values(["ticker", "date"]).copy()
    x["date"] = x["date"].dt.to_period("W-FRI").dt.to_timestamp()
    w = x.groupby(["ticker", "date"], as_index=False).agg(open=("open", "first"), close=("close", "last"))
    w[["open", "close"]] = w[["open", "close"]].apply(pd.to_numeric, errors="coerce").round(5)
    return w.dropna(subset=["open", "close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def load_splits(path: Path) -> pd.DataFrame:
    sp = clean_cols(pd.read_csv(path))
    sp = sp.rename(
        columns={
            "symbol": "ticker",
            "stock splits": "split_factor",
        }
    )
    sp["ticker"] = sp["ticker"].map(norm_ticker)
    sp["date"] = to_date(sp["date"]).dt.normalize()
    sp["split_factor"] = pd.to_numeric(sp["split_factor"], errors="coerce")

    sp = sp.dropna(subset=["ticker", "date", "split_factor"])
    sp = sp[(sp["date"] >= START) & (sp["date"] <= END) & (sp["split_factor"] != 0)]
    return sp[["ticker", "date", "split_factor"]]


def adjust_splits_if_needed(prices: pd.DataFrame, splits: pd.DataFrame, tol: float = TOL):
    out, applied = [], 0

    for t, dft in prices.groupby("ticker", sort=False):
        spt = splits[splits["ticker"] == t].sort_values("date")
        if spt.empty or len(dft) < 2:
            out.append(dft)
            continue

        dft = dft.sort_values("date").copy()
        dft["prev_close"] = dft["close"].shift(1)

        chk = (
            spt.merge(dft[["date", "close", "prev_close"]], on="date", how="left")
            .dropna(subset=["close", "prev_close", "split_factor"])
            .copy()
        )
        if chk.empty:
            out.append(dft.drop(columns="prev_close"))
            continue

        ratio = chk["prev_close"] / chk["close"]
        ok = ((ratio - chk["split_factor"]).abs() / chk["split_factor"].abs()) <= tol

        confirmed = (
            chk.loc[ok, ["date", "split_factor"]]
            .groupby("date", as_index=False)["split_factor"]
            .prod()
            .sort_values("date")
        )
        if confirmed.empty:
            out.append(dft.drop(columns="prev_close"))
            continue

        dft2 = dft.drop(columns="prev_close")
        for dt, f in confirmed.sort_values("date", ascending=False).itertuples(index=False):
            mask = dft2["date"] < dt
            dft2.loc[mask, ["open", "close"]] = dft2.loc[mask, ["open", "close"]] / float(f)

        applied += len(confirmed)
        out.append(dft2)

    prices_adj = pd.concat(out, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices_adj, applied


def tickers_needing_yf(prices: pd.DataFrame, end: pd.Timestamp) -> list[str]:
    missing = set(prices.loc[prices[["open", "close"]].isna().any(axis=1), "ticker"].unique())
    last = prices.groupby("ticker")["date"].max()
    short = set(last[last < end].index)
    return sorted(missing | short)


def yf_fetch_open_close(tickers: list[str], start=START, end=EFFECTIVE_END, retries: int = 4) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", "date", "yf_open", "yf_close"])

    def is_rl(e: Exception) -> bool:
        s = repr(e).lower()
        return "ratelimit" in s or "too many requests" in s or "rate limited" in s

    data = None
    for k in range(retries):
        try:
            data = yf.download(
                tickers=tickers,
                start=start.strftime("%Y-%m-%d"),
                end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),  # end exclusive
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            break
        except Exception as e:
            time.sleep((60 * (k + 1)) if is_rl(e) else (1.0 * (2**k)))

    if data is None or data.empty:
        return pd.DataFrame(columns=["ticker", "date", "yf_open", "yf_close"])

    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        for sym in sorted(set(data.columns.get_level_values(0))):
            sub = data[sym][["Open", "Close"]].rename(columns={"Open": "yf_open", "Close": "yf_close"}).reset_index()
            sub["ticker"] = sym
            frames.append(sub)
    else:
        sub = data[["Open", "Close"]].rename(columns={"Open": "yf_open", "Close": "yf_close"}).reset_index()
        sub["ticker"] = tickers[0]
        frames.append(sub)

    y = pd.concat(frames, ignore_index=True).rename(columns={"Date": "date"})
    y["date"] = to_date(y["date"])
    y["ticker"] = y["ticker"].map(norm_ticker)
    y["yf_open"] = pd.to_numeric(y["yf_open"], errors="coerce").round(5)
    y["yf_close"] = pd.to_numeric(y["yf_close"], errors="coerce").round(5)

    return y.dropna(subset=["ticker", "date", "yf_open", "yf_close"])[["ticker", "date", "yf_open", "yf_close"]]


def yf_fetch_batched(tickers: list[str], start=START, end=EFFECTIVE_END, batch_size: int = 150) -> pd.DataFrame:
    parts = [yf_fetch_open_close(b, start, end) for b in batch(tickers, batch_size)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ticker", "date", "yf_open", "yf_close"])


def merge_prices_with_yf(base: pd.DataFrame, yfl: pd.DataFrame) -> pd.DataFrame:
    yfl = yfl.rename(columns={"yf_open": "open_yf", "yf_close": "close_yf"})
    m = base.merge(yfl, on=["ticker", "date"], how="outer")

    m["open"] = pd.to_numeric(m.get("open"), errors="coerce").combine_first(m.get("open_yf"))
    m["close"] = pd.to_numeric(m.get("close"), errors="coerce").combine_first(m.get("close_yf"))

    m = m.drop(columns=[c for c in ("open_yf", "close_yf") if c in m.columns])
    m = m.dropna(subset=["ticker", "date"]).drop_duplicates(["ticker", "date"])
    m = m[(m["date"] >= START) & (m["date"] <= END)].dropna(subset=["open", "close"])

    m["open"] = m["open"].round(5)
    m["close"] = m["close"].round(5)
    return m.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_meta(meta_file: Path, addr_file: Path) -> pd.DataFrame:
    meta = clean_cols(pd.read_csv(meta_file)).rename(columns={"symbol": "ticker"})
    meta["ticker"] = meta["ticker"].map(norm_ticker)
    meta = meta.drop_duplicates("ticker")

    addr = clean_cols(pd.read_csv(addr_file))
    if "ticker" not in addr.columns:
        raise RuntimeError("Address file must have column 'ticker'.")
    if "address" not in addr.columns:
        addr["address"] = np.nan

    addr["ticker"] = addr["ticker"].map(norm_ticker)
    addr = addr[(addr["ticker"].notna()) & (addr["ticker"] != "") & (addr["ticker"] != "TICKER")]
    addr = addr.drop_duplicates("ticker")[["ticker", "address"]]

    return meta.merge(addr, on="ticker", how="left").reset_index(drop=True)


def yf_symbol_crypto(t: str) -> str:
    t = norm_ticker(t)
    return t if "-" in t else f"{t}-USD"


# ------------------ run ------------------

df_nasdaq_meta = load_meta(RAW / "nasdaq_screener.csv", RAW / "nasdaq_company_addresses.csv")

# NASDAQ
nasdaq = load_price_dir(RAW / "nasdaq-daily")
splits = load_splits(RAW / "splits_2000_2025.csv")
nasdaq, n_adj = adjust_splits_if_needed(nasdaq, splits, tol=TOL)
quality(nasdaq, "NASDAQ daily (after split-check, before yfinance)")

need = tickers_needing_yf(nasdaq, EFFECTIVE_END)
yf_n = yf_fetch_batched(need, START, EFFECTIVE_END)
df_nasdaq_daily = merge_prices_with_yf(nasdaq, yf_n)[["ticker", "date", "open", "close"]]
df_nasdaq_weekly = daily_to_weekly(df_nasdaq_daily)

# Crypto
crypto = load_price_dir(RAW / "crypto-daily")
quality(crypto, "Crypto daily (before yfinance)")

need_c = tickers_needing_yf(crypto, EFFECTIVE_END)
sym_map = {t: yf_symbol_crypto(t) for t in need_c}
inv_map = {norm_ticker(v): norm_ticker(k) for k, v in sym_map.items()}

yf_c_parts = []
for b in batch(need_c, 150):
    y = yf_fetch_open_close([sym_map[t] for t in b], START, EFFECTIVE_END)
    y["ticker"] = y["ticker"].map(lambda s: inv_map.get(norm_ticker(s), norm_ticker(s)))
    yf_c_parts.append(y)

yf_c = pd.concat(yf_c_parts, ignore_index=True) if yf_c_parts else pd.DataFrame(
    columns=["ticker", "date", "yf_open", "yf_close"]
)
df_crypto_daily = merge_prices_with_yf(crypto, yf_c)[["ticker", "date", "open", "close"]]
df_crypto_weekly = daily_to_weekly(df_crypto_daily)

# FX (EURUSD) from yfinance
eurusd_raw = yf_fetch_open_close(["EURUSD=X"], START, EFFECTIVE_END)
df_eurusd_daily = (
    eurusd_raw.rename(columns={"yf_open": "open", "yf_close": "close"})
    .assign(ticker="EURUSD")[["ticker", "date", "open", "close"]]
    .sort_values(["ticker", "date"])
    .reset_index(drop=True)
)
df_eurusd_weekly = daily_to_weekly(df_eurusd_daily)

# quality + save (required names)
quality(df_nasdaq_daily, "df_nasdaq_daily (final)")
quality(df_nasdaq_weekly, "df_nasdaq_weekly (final)")
quality(df_crypto_daily, "df_crypto_daily (final)")
quality(df_crypto_weekly, "df_crypto_weekly (final)")
quality(df_eurusd_daily, "df_eurusd_daily (final)")
quality(df_eurusd_weekly, "df_eurusd_weekly (final)")
quality(df_nasdaq_meta, "df_nasdaq_meta (final)")

df_nasdaq_daily.to_csv(OUT / "df_nasdaq_daily.csv", index=False)
df_nasdaq_weekly.to_csv(OUT / "df_nasdaq_weekly.csv", index=False)
df_crypto_daily.to_csv(OUT / "df_crypto_daily.csv", index=False)
df_crypto_weekly.to_csv(OUT / "df_crypto_weekly.csv", index=False)
df_eurusd_daily.to_csv(OUT / "df_eurusd_daily.csv", index=False)
df_eurusd_weekly.to_csv(OUT / "df_eurusd_weekly.csv", index=False)
df_nasdaq_meta.to_csv(OUT / "df_nasdaq_meta.csv", index=False)

print("\nSaved to:", OUT.resolve())
print("Splits adjusted (unadjusted jumps detected):", n_adj)
print("EFFECTIVE_END used for yfinance:", EFFECTIVE_END.date())