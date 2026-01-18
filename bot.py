# bot.py
# MEXC 17:00 Lagos candle screener (FINAL-ONLY output)
# HIT definition:
#   Within the 17:00–18:00 Lagos 1h candle, (HIGH / OPEN - 1) >= TARGET_PCT
# Optional auto-buy.

import os, time, re, sys, warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ccxt
import pytz

warnings.filterwarnings("ignore")

LOCAL_TZ = "Africa/Lagos"
TZ_LAGOS = pytz.timezone(LOCAL_TZ)

MEXC_KEY    = os.getenv("MEXC_KEY", "")
MEXC_SECRET = os.getenv("MEXC_SECRET", "")

def env_float(name, default):
    v = os.getenv(name, "")
    try:
        return float(v) if v != "" else float(default)
    except:
        return float(default)

def env_int(name, default):
    v = os.getenv(name, "")
    try:
        return int(v) if v != "" else int(default)
    except:
        return int(default)

def env_str(name, default):
    v = os.getenv(name, "")
    return v if v != "" else str(default)

HISTORY_DAYS   = env_int("HISTORY_DAYS", 30)
MIN_DAYS_OLD   = env_int("MIN_DAYS_OLD", 3)

VOL_MIN        = env_float("VOL_MIN", 10_000_000)
VOL_MAX        = env_float("VOL_MAX", 600_000_000)

TARGET_PCT     = env_float("TARGET_PCT", 3.0)
ANCHOR_HOUR    = env_int("ANCHOR_HOUR", 10)

TOP_N          = env_int("TOP_N", 20)

DO_BUY         = env_int("DO_BUY", 1)
BUY_RANKS      = env_str("BUY_RANKS", "1")
BUY_TIME       = env_str("BUY_TIME", "14:59:58")
BUY_PCT_BALANCE = env_float("BUY_PCT_BALANCE", 50.0)
TP_PCT          = env_float("TP_PCT", 50.0)

MAX_RETRIES    = 4

LEVERAGED_PATTERNS = [
    r".*\d+[LS]$",
    r".*(UP|DOWN)$",
    r".*(BULL|BEAR)$",
    r".*(3L|3S|5L|5S|10L|10S|20L|20S|50L|50S|100L|100S)$",
]

def log(msg=""):
    print(msg)
    sys.stdout.flush()

def is_leveraged_token(base):
    if not base:
        return False
    b = str(base).upper().replace("-", "").replace("_", "")
    for pat in LEVERAGED_PATTERNS:
        if re.match(pat, b):
            return True
    return False

def safe_quote_volume(tkr):
    if not tkr:
        return None
    for k in ["quoteVolume","quoteVolume24h","quote_vol","volValue","quoteVol","turnover"]:
        v = tkr.get(k)
        if v is not None:
            try:
                return float(v)
            except:
                pass
    info = tkr.get("info", {}) or {}
    for k in ["quoteVolume","turnover","volumeUsdt","volValue"]:
        if k in info:
            try:
                return float(info[k])
            except:
                pass
    return None

def make_exchange(public_only=True):
    params = {
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {"defaultType": "spot", "adjustForTimeDifference": True},
    }
    if not public_only and MEXC_KEY and MEXC_SECRET:
        params["apiKey"] = MEXC_KEY
        params["secret"] = MEXC_SECRET
    return ccxt.mexc(params)

def fetch_ohlcv_1h(ex, symbol, since_ms, limit=1000):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = ex.fetch_ohlcv(symbol, timeframe="1h", since=since_ms, limit=limit)
            if not data:
                return None
            df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            return df
        except Exception:
            time.sleep(0.7 * attempt)
    return None

def build_daily_17h_metrics(df_1h: pd.DataFrame, anchor_hour: int) -> pd.DataFrame:
    if df_1h is None or df_1h.empty:
        return pd.DataFrame()

    d = df_1h.copy().sort_values("ts")
    d["ts_lagos"] = d["ts"].dt.tz_convert(TZ_LAGOS)
    d["day_lagos"] = d["ts_lagos"].dt.normalize()
    d["hour_lagos"] = d["ts_lagos"].dt.hour

    d["quote_vol"] = d["volume"] * d["close"]
    daily = d.groupby("day_lagos").agg(daily_quote_vol=("quote_vol","sum"))

    cA = d[d["hour_lagos"] == anchor_hour].groupby("day_lagos").agg(
        openA=("open","first"),
        highA=("high","max"),
        lowA=("low","min"),
        closeA=("close","last"),
    )

    out = daily.join(cA, how="inner").dropna(subset=["daily_quote_vol","openA","highA"])
    out["ret_open_to_high"] = (out["highA"] / out["openA"]) - 1.0
    out["ret_open_to_low"]  = (out["lowA"] / out["openA"]) - 1.0
    return out

def laplace_smooth(hits: int, n: int) -> float:
    return (hits + 1.0) / (n + 2.0)

def pick_symbols_in_24h_volume_range(ex, quote="USDT", vol_min=0, vol_max=1e18):
    mk = ex.load_markets()
    tix = ex.fetch_tickers()

    symbols, vol_map = [], {}
    for s, m in mk.items():
        try:
            if not m.get("spot"):
                continue
            if str(m.get("quote")) != quote:
                continue
            if m.get("active") is False:
                continue
            if is_leveraged_token(m.get("base")):
                continue

            qv = safe_quote_volume(tix.get(s, {}))
            if qv is None:
                continue
            if vol_min <= qv <= vol_max:
                symbols.append(s)
                vol_map[s] = float(qv)
        except:
            continue

    return sorted(symbols), vol_map

def parse_ranks(s):
    out = []
    for part in (s or "").replace(" ", "").split(","):
        if not part:
            continue
        try:
            out.append(int(part))
        except:
            pass
    return out

def wait_until_lagos_time(ex, buy_time_str):
    if buy_time_str.strip().lower() == "now":
        return
    try:
        hh, mm, ss = [int(x) for x in buy_time_str.split(":")]
    except:
        log("⚠️ Bad BUY_TIME format; using NOW.")
        return

    now_lagos = datetime.now(TZ_LAGOS)
    target_dt = now_lagos.replace(hour=hh, minute=mm, second=ss, microsecond=0)
    if target_dt <= now_lagos:
        target_dt += timedelta(days=1)

    target_utc = target_dt.astimezone(pytz.UTC)
    target_ms = int(target_utc.timestamp() * 1000)

    while True:
        try:
            now_ms = ex.fetch_time()
        except Exception:
            now_ms = int(time.time() * 1000)

        remaining = (target_ms - now_ms) / 1000.0
        if remaining <= 0:
            break
        time.sleep(1.0)

def make_tp_price(ex, symbol, entry_price, tp_pct):
    raw = float(entry_price) * (1.0 + tp_pct / 100.0)
    return float(ex.price_to_precision(symbol, raw))

def make_amount(ex, symbol, base_amount):
    return float(ex.amount_to_precision(symbol, float(base_amount)))

def run_screener():
    target_frac = TARGET_PCT / 100.0

    log("=" * 80)
    log("MEXC 17:00 Lagos Candle Screener (FINAL-ONLY output)")
    log(f"HIT: (HIGH/OPEN - 1) >= {TARGET_PCT:.1f}% for the {ANCHOR_HOUR:02d}:00 candle")
    log(f"History days: {HISTORY_DAYS} | Min valid days: {MIN_DAYS_OLD}")
    log(f"Volume filter: {VOL_MIN:,.0f} to {VOL_MAX:,.0f} USDT")
    log("=" * 80)

    ex = make_exchange(public_only=True)

    symbols, vol24_map = pick_symbols_in_24h_volume_range(ex, quote="USDT", vol_min=VOL_MIN, vol_max=VOL_MAX)
    if not symbols:
        log("No symbols matched the 24h volume filter.")
        return None, None, ex

    now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
    since_utc = now_utc - timedelta(days=HISTORY_DAYS + 2)
    since_ms = int(since_utc.timestamp() * 1000)

    limit = 1000 if HISTORY_DAYS <= 40 else 2000

    rows = []
    skip_no_ohlcv = 0
    skip_few_hours = 0
    skip_few_days = 0
    skip_median_vol = 0
    errors = 0

    start = time.time()

    for sym in symbols:
        df = fetch_ohlcv_1h(ex, sym, since_ms, limit=limit)
        if df is None or df.empty:
            skip_no_ohlcv += 1
            continue

        if len(df) < 24 * MIN_DAYS_OLD:
            skip_few_hours += 1
            continue

        try:
            daily = build_daily_17h_metrics(df, ANCHOR_HOUR).sort_index().tail(HISTORY_DAYS)
            if len(daily) < MIN_DAYS_OLD:
                skip_few_days += 1
                continue

            med_daily_vol = float(daily["daily_quote_vol"].median())
            if not (VOL_MIN <= med_daily_vol <= VOL_MAX):
                skip_median_vol += 1
                continue

            hits = int((daily["ret_open_to_high"] >= target_frac).sum())
            n = int(len(daily))

            p_smooth = laplace_smooth(hits, n)
            avg_reach = float(daily["ret_open_to_high"].mean())
            best_reach = float(daily["ret_open_to_high"].max())
            worst_low  = float(daily["ret_open_to_low"].min())

            score = (p_smooth * 1.0) + (np.tanh(avg_reach) * 0.25) + (np.tanh(best_reach) * 0.10)

            rows.append({
                "Pair": sym,
                "HistDays": n,
                "Hits": hits,
                "P_hit_smooth": p_smooth,
                "AvgOpenToHigh": avg_reach,
                "BestOpenToHigh": best_reach,
                "WorstOpenToLow": worst_low,
                "Vol24h_quote": float(vol24_map.get(sym, np.nan)),
                "MedianDailyQuoteVol": med_daily_vol,
                "Score": score,
            })

        except Exception:
            errors += 1

    elapsed = time.time() - start

    log("\nDONE.")
    log(f"Total symbols in volume range: {len(symbols)}")
    log(f"Kept: {len(rows)} | Errors: {errors}")
    log(f"Skip reasons -> noOHLCV:{skip_no_ohlcv} | fewHours:{skip_few_hours} | fewDays17h:{skip_few_days} | medianVol:{skip_median_vol}")
    log(f"Elapsed: {elapsed/60:.1f} min\n")

    if not rows:
        log("No symbols matched after filtering.")
        return None, None, ex

    result = pd.DataFrame(rows).sort_values(
        ["Score","P_hit_smooth","Hits","HistDays","Vol24h_quote"],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)

    result.insert(0, "Rank", np.arange(1, len(result) + 1))

    fname = f"./mexc_17h_reach_{int(TARGET_PCT)}pct_{HISTORY_DAYS}d_min{MIN_DAYS_OLD}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    result.to_csv(fname, index=False)

    log("Top results:")
    log(result.head(TOP_N).to_string(index=False))
    log(f"\nSaved CSV: {fname}")
    return result, fname, ex

def auto_buy_from_results(results_df: pd.DataFrame):
    if results_df is None or results_df.empty:
        log("❌ No results to buy from.")
        return

    ranks = parse_ranks(BUY_RANKS)
    if not ranks:
        log("❌ BUY_RANKS invalid.")
        return

    chosen = results_df[results_df["Rank"].isin(ranks)].copy().sort_values("Rank")
    if chosen.empty:
        log("❌ None of those ranks exist in results.")
        return

    if not MEXC_KEY or not MEXC_SECRET:
        log("❌ Missing MEXC_KEY / MEXC_SECRET (GitHub Secrets).")
        return

    trading = make_exchange(public_only=False)

    bal = trading.fetch_balance()
    usdt_free = float(bal.get("free", {}).get("USDT", 0.0))
    log(f"💼 USDT free balance: {usdt_free:.4f}")

    total_spend = usdt_free * (BUY_PCT_BALANCE / 100.0)
    if total_spend <= 0:
        log("❌ Total spend <= 0.")
        return

    per_coin = total_spend / len(chosen)
    log(f"💰 Total spend ≈ {total_spend:.4f} | per coin ≈ {per_coin:.4f}")

    wait_until_lagos_time(trading, BUY_TIME)

    log("🚀 Starting auto-buy...")
    for _, row in chosen.iterrows():
        symbol = row["Pair"]
        try:
            ticker = trading.fetch_ticker(symbol)
            last = float(ticker["last"])
            base_amt = per_coin / last
            base_amt = make_amount(trading, symbol, base_amt)

            log(f"💸 Market BUY {symbol}: amount≈{base_amt} base @ last≈{last}")
            order = trading.create_market_buy_order(symbol, base_amt)

            filled = last
            try:
                full = trading.fetch_order(order["id"], symbol)
                if full.get("average"):
                    filled = float(full["average"])
                elif full.get("price"):
                    filled = float(full["price"])
            except:
                pass

            log(f"✅ Bought {symbol} around {filled:.8f}")

            if TP_PCT > 0:
                tp_price = make_tp_price(trading, symbol, filled, TP_PCT)
                log(f"📌 TP LIMIT SELL {symbol}: amount={base_amt} @ {tp_price} (+{TP_PCT}%)")
                trading.create_limit_sell_order(symbol, base_amt, tp_price)
                log("✅ TP order placed.")

        except Exception as e:
            log(f"❌ Error buying {symbol}: {e}")

    log("✅ Auto-buy finished.")

def main():
    now_lagos = datetime.now(TZ_LAGOS)
    log(f"Starting bot. Lagos time now: {now_lagos.strftime('%Y-%m-%d %H:%M:%S')}")
    results, fname, ex_public = run_screener()

    if results is None or results.empty:
        log("No screener results. Exiting.")
        return

    if DO_BUY == 1:
        log("DO_BUY=1 → Trading enabled.")
        auto_buy_from_results(results)
    else:
        log("DO_BUY=0 → Screener only (no trades).")

if __name__ == "__main__":
    main()
