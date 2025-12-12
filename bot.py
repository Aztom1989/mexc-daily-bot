# bot.py
# Headless version of your screener + auto-buy (for GitHub Actions).
# No widgets, no Colab UI.

import os, time, math, re, sys, warnings
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import ccxt
import pytz

warnings.filterwarnings("ignore")

# -------- BASIC SETTINGS --------
LOCAL_TZ = "Africa/Lagos"

# MEXC keys come from GitHub Secrets (MEXC_KEY, MEXC_SECRET)
# You can also hard-code here for local testing, but NOT recommended on GitHub.
DEFAULT_MEXC_KEY    = os.getenv("MEXC_KEY", "")
DEFAULT_MEXC_SECRET = os.getenv("MEXC_SECRET", "")

SUPPORTED_EXCHANGES = ["mexc", "kucoin", "gate", "bybit", "okx", "binance"]

DEFAULTS = {
    "vol_min":  10_000_000,
    "vol_max":  900_000_000,
    "history_days": 7,
    "target_str": "50",       # default 50%
    "anchor_hour": 17,
    "fees_slip": 0.0015,
    "tp_pct": 30.0,           # TP 30%
    "sl_default": 5.0,
    "buy_time": "18:59:59",   # wait until this Lagos time
    "buy_pct_balance": 1.0,   # use 1% of balance in total
    "buy_fixed_usdt": 10.0,
}

# -------------------------
# Utility helpers
# -------------------------
def log(msg=""):
    print(msg)
    sys.stdout.flush()

def parse_targets(target_str):
    s = (target_str or "").replace(" ", "")
    if not s:
        return [50.0]
    out = []
    for part in s.split(","):
        if not part:
            continue
        try:
            v = float(part)
            if v > 0:
                out.append(v)
        except:
            pass
    return out[:10] or [50.0]

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

def make_exchange(ex_id, api_key="", api_secret="", password=""):
    cls = getattr(ccxt, ex_id)
    params = {
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        }
    }
    if api_key and api_secret:
        params["apiKey"] = api_key
        params["secret"] = api_secret
    if password:
        params["password"] = password
    ex = cls(params)
    return ex

# Leveraged token filter (btc5L, SOMEBULL, etc.)
LEVERAGED_PATTERNS = [
    r".*\d+[LS]$",
    r".*(UP|DOWN)$",
    r".*(BULL|BEAR)$",
    r".*(3L|3S|5L|5S|10L|10S|20L|20S|50L|50S|100L|100S)$",
]

def is_leveraged_token(base):
    if not base:
        return False
    b = str(base).upper().replace("-", "").replace("_", "")
    for pat in LEVERAGED_PATTERNS:
        if re.match(pat, b):
            return True
    return False

def pick_spot_symbols(ex, ex_id, quote, vol_min, vol_max):
    mk = ex.load_markets()
    tix = ex.fetch_tickers()

    symbols = []
    for s, m in mk.items():
        try:
            if not m.get("spot"):
                continue
            if str(m.get("quote")) != quote:
                continue
            if m.get("active") is False:
                continue
            if ex_id == "gate" and is_leveraged_token(m.get("base")):
                continue
            t = tix.get(s, {})
            qv = safe_quote_volume(t)
            if qv is None:
                continue
            if vol_min <= qv <= vol_max:
                symbols.append((s, qv))
        except:
            continue
    return symbols  # list of (symbol, vol24)

def timeframe_to_ms(tf):
    unit = tf[-1]
    n = int(tf[:-1])
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 3_600_000
    if unit == "d":
        return n * 86_400_000
    raise ValueError("Bad timeframe "+tf)

def safe_fetch_ohlcv_range(ex, symbol, timeframe, since_ms, until_ms, limit=1000):
    tf_ms = timeframe_to_ms(timeframe)
    rows = []
    cursor = since_ms
    max_errors = 5
    errors = 0

    while cursor < until_ms:
        try:
            candles = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
            if not candles:
                break
            rows.extend(candles)
            last_ts = candles[-1][0]
            nxt = last_ts + tf_ms
            if nxt <= cursor:
                break
            cursor = nxt
            time.sleep(ex.rateLimit / 1000.0)
        except Exception as e:
            errors += 1
            if "time range" in str(e).lower():
                cursor += 24 * 3_600_000
            else:
                time.sleep(0.5)
            if errors >= max_errors:
                break

    if not rows:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])

    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df

def build_daily_windows(df, tzname, anchor_hour):
    if df.empty:
        return pd.DataFrame(columns=["t0","openA","max_high","min_low"])

    dt_utc = pd.to_datetime(df["ts"], unit="ms", utc=True)
    tz_loc = pytz.timezone(tzname)
    df = df.copy()
    df["dt_loc"] = dt_utc.dt.tz_convert(tz_loc)
    df["date"] = df["dt_loc"].dt.date

    rows = []
    for d, g in df.groupby("date"):
        g = g.copy()
        g["hour"] = g["dt_loc"].dt.hour
        g_after = g[g["hour"] >= anchor_hour]
        if not g_after.empty:
            row0 = g_after.iloc[0]
        else:
            row0 = g.iloc[0]
        mask = g["ts"] >= row0["ts"]
        win = g[mask]
        if win.empty:
            continue
        rows.append({
            "t0": int(row0["ts"]),
            "openA": float(row0["open"]),
            "max_high": float(win["high"].max()),
            "min_low": float(win["low"].min()),
        })
    return pd.DataFrame(rows)

def jeffreys_posterior(x, n):
    alpha = x + 0.5
    beta_ = n - x + 0.5
    return alpha / (alpha + beta_)

def suggest_sl_pct(p_ens, target_pct, sl_default):
    p = max(0.01, min(0.99, float(p_ens)))
    ratio = 0.25 + 0.75 * p
    sl_from_target = target_pct * (1.0 - p) * ratio
    return max(0.25 * target_pct, min(sl_default, sl_from_target))

def make_tp_price(ex, symbol, filled_price, tp_pct):
    raw = float(filled_price) * (1.0 + tp_pct / 100.0)
    return float(ex.price_to_precision(symbol, raw))

def make_amount(ex, symbol, base_amount):
    return float(ex.amount_to_precision(symbol, float(base_amount)))

# ========== COUNTDOWN UNTIL BUY TIME ==========
def wait_until_exchange_time(ex, target_lagos_dt):
    tz_lagos = pytz.timezone(LOCAL_TZ)
    target_utc = target_lagos_dt.astimezone(pytz.UTC)
    target_ms = int(target_utc.timestamp() * 1000)

    try:
        now_ms = ex.fetch_time()
    except Exception:
        now_ms = int(time.time() * 1000)

    total_secs = max(1, int((target_ms - now_ms) / 1000))

    while True:
        try:
            now_ms = ex.fetch_time()
        except Exception:
            now_ms = int(time.time() * 1000)

        remaining = (target_ms - now_ms) / 1000.0
        if remaining <= 0:
            now_lagos = datetime.now(tz_lagos).strftime('%H:%M:%S')
            now_utc = datetime.utcnow().strftime('%H:%M:%S')
            bar_len = 28
            bar = "#" * bar_len
            line = (
                f"\r⏳ Waiting until {target_lagos_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"[{bar}] 00:00:00 remaining | "
                f"Lagos now: {now_lagos} | UTC now: {now_utc}   "
            )
            print(line, end="")
            sys.stdout.flush()
            break

        elapsed = total_secs - remaining
        pct = max(0.0, min(1.0, elapsed / total_secs))
        bar_len = 28
        done = int(bar_len * pct)
        bar = "#" * done + "-" * (bar_len - done)

        hrs, rem = divmod(int(remaining), 3600)
        mins, secs = divmod(rem, 60)
        now_lagos = datetime.now(tz_lagos).strftime('%H:%M:%S')
        now_utc = datetime.utcnow().strftime('%H:%M:%S')

        line = (
            f"\r⏳ Waiting until {target_lagos_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"[{bar}] {hrs:02d}:{mins:02d}:{secs:02d} remaining | "
            f"Lagos now: {now_lagos} | UTC now: {now_utc}"
        )
        print(line, end="")
        sys.stdout.flush()
        time.sleep(1.0)

    print()

# -------------------------
# Core screener
# -------------------------
def run_screener(
    ex_id="mexc",
    quote="USDT",
    timeframe="1h",
    vol_min=50_000,
    vol_max=2_000_000,
    history_days=7,
    target_list=None,
    anchor_mode="hour",
    anchor_hour=17,
    use_local_tz=True,
    fees_slip=0.0015,
):
    target_list = target_list or [50.0]

    tz_name = LOCAL_TZ if use_local_tz else "UTC"
    tz_obj = pytz.timezone(tz_name)
    now_loc = datetime.now(tz_obj)

    if anchor_mode == "now":
        anchor_hour_used = now_loc.hour
    else:
        anchor_hour_used = int(anchor_hour)

    log("="*72)
    log(f"CEX Screener — {ex_id.upper()} — Multi-Target ({timeframe})")
    log(f"Quote: {quote} | Volume range: [{vol_min:,.0f}, {vol_max:,.0f}]")
    log(f"Anchor TZ: {tz_name} | Anchor mode: {anchor_mode}")
    log("="*72)

    ex = make_exchange(ex_id)
    log("🔌 Connecting (public)…")
    server_time_ms = ex.fetch_time()
    log(f"🕒 Exchange time: {datetime.utcfromtimestamp(server_time_ms/1000).strftime('%Y-%m-%d %H:%M:%S')} UTC")

    log("📈 Loading markets & tickers…")
    syms_with_vol = pick_spot_symbols(ex, ex_id, quote, vol_min, vol_max)
    if not syms_with_vol:
        log("❌ No symbols in that volume range.")
        return None, None, ex

    symbols = [s for (s, _) in syms_with_vol]
    vol_map = {s: v for (s, v) in syms_with_vol}
    log(f"✅ Symbols in range: {len(symbols)}")

    end_loc = now_loc
    start_loc = end_loc - timedelta(days=history_days+1)
    since_ms = int(start_loc.astimezone(pytz.UTC).timestamp()*1000)
    until_ms = int(end_loc.astimezone(pytz.UTC).timestamp()*1000)

    tf = timeframe
    all_rows = []
    log(f"⏳ Pulling ~{history_days} days OHLCV & daily windows ({tz_name})…")
    for i, s in enumerate(symbols, 1):
        try:
            df = safe_fetch_ohlcv_range(ex, s, tf, since_ms, until_ms)
            if df.empty:
                continue
            win = build_daily_windows(df, tz_name, anchor_hour_used)
            if win.empty:
                continue
            win["symbol"] = s
            all_rows.append(win)
        except Exception:
            continue
        if i % 50 == 0:
            log(f"...pulled {i}/{len(symbols)}")

    if not all_rows:
        log("❌ No daily windows built.")
        return None, None, ex

    panel = pd.concat(all_rows, ignore_index=True)
    results_all = []

    for target_pct in target_list:
        target_frac = target_pct / 100.0
        df = panel.copy()
        df["runup"] = df["max_high"] / df["openA"] - 1.0
        df["hit"] = (df["runup"] >= target_frac).astype(int)

        grp = df.groupby("symbol").agg(
            n_days=("hit", "count"),
            hits=("hit", "sum")
        ).reset_index()

        grp["p_ens"] = jeffreys_posterior(grp["hits"], grp["n_days"])
        grp["EV"] = grp["p_ens"] * target_frac - fees_slip
        grp["Target_pct"] = target_pct
        grp["Vol24h_quote"] = grp["symbol"].map(vol_map)
        grp["SL_suggest_pct"] = [
            suggest_sl_pct(p, target_pct, DEFAULTS["sl_default"])
            for p in grp["p_ens"].values
        ]

        grp = grp.sort_values(
            ["p_ens", "hits", "n_days", "Vol24h_quote"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)
        grp["Rank"] = np.arange(1, len(grp) + 1)
        results_all.append(grp)

    full = pd.concat(results_all, ignore_index=True)
    full = full[["Rank", "Target_pct", "symbol", "n_days", "hits",
                 "p_ens", "EV", "SL_suggest_pct", "Vol24h_quote"]]

    pretty = full.rename(columns={
        "Target_pct": "Target(%)",
        "symbol": "Pair",
        "n_days": "HistDays",
        "SL_suggest_pct": "SL_suggest(%)",
        "Vol24h_quote": "Vol24h_quote"
    })

    # Log first few rows in text
    log("Top rows from screener:")
    try:
        log(pretty.head(10).to_string(index=False))
    except Exception:
        log(str(pretty.head(10)))

    # Save CSV locally (only for logs / debugging)
    fname = f"./{ex_id}_multiTarget_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pretty.to_csv(fname, index=False)
    log(f"Saved CSV: {fname}")

    return pretty, fname, ex

# -------------------------
# Auto-buy (MEXC only)
# -------------------------
def auto_buy_from_results(
    ex,
    ex_id,
    results_df,
    ranks_to_buy,
    mode_buy_size="pct",
    pct_of_balance=1.0,
    fixed_usdt=10.0,
    tp_pct=30.0,
    use_tp=True,
    use_sl=False,
    sl_value=None,
    buy_time_str="now",
):
    if ex_id != "mexc":
        log("⚠️ Auto-buy currently implemented only for MEXC.")
        return

    if results_df is None or results_df.empty:
        log("❌ No scan results to buy from.")
        return

    ranks = []
    for part in (ranks_to_buy or "").replace(" ", "").split(","):
        if part:
            try:
                ranks.append(int(part))
            except:
                pass
    if not ranks:
        log("❌ No valid ranks given.")
        return

    chosen = results_df[results_df["Rank"].isin(ranks)].copy()
    chosen = chosen.sort_values("Rank")
    if chosen.empty:
        log("❌ None of those ranks found in results.")
        return

    log("\n🔑 Connecting (trading)…")
    trading = make_exchange(
        "mexc",
        api_key=DEFAULT_MEXC_KEY,
        api_secret=DEFAULT_MEXC_SECRET,
    )
    bal = trading.fetch_balance()
    quote_bal = float(bal.get("free", {}).get("USDT", 0.0))
    log(f"💼 USDT balance: {quote_bal}")

    if mode_buy_size == "pct":
        total_spend = quote_bal * (pct_of_balance / 100.0)
    else:
        total_spend = fixed_usdt * len(chosen)

    if total_spend <= 0:
        log("❌ Total spend <= 0.")
        return
    per_coin = total_spend / len(chosen)
    log(f"💰 Will spend ≈ {per_coin:.4f} USDT per coin on {len(chosen)} coin(s).")

    tz_lagos = pytz.timezone(LOCAL_TZ)
    if buy_time_str.strip().lower() == "now":
        target_dt = datetime.now(tz_lagos)
        log(f"\n🕒 Buy time: NOW ({target_dt.strftime('%H:%M:%S')} Lagos)")
    else:
        try:
            hh, mm, ss = [int(x) for x in buy_time_str.split(":")]
            now_lagos = datetime.now(tz_lagos)
            target_dt = now_lagos.replace(hour=hh, minute=mm, second=ss, microsecond=0)
            if target_dt <= now_lagos:
                target_dt += timedelta(days=1)
        except Exception:
            target_dt = datetime.now(tz_lagos)
            log("⚠️ Bad buy time, using NOW.")
        wait_until_exchange_time(trading, target_dt)

    log("\n🚀 Starting auto-buy…")
    for _, row in chosen.iterrows():
        symbol = row["Pair"]
        sl_pct = float(sl_value) if (use_sl and sl_value is not None) else float(row["SL_suggest(%)"])

        try:
            ticker = trading.fetch_ticker(symbol)
            last = float(ticker["last"])
            base_amt = per_coin / last
            base_amt = make_amount(trading, symbol, base_amt)

            log(f"💸 Market BUY {symbol}: ~{base_amt} base ≈ {per_coin:.4f} USDT @ {last:.6g}")
            order = trading.create_market_buy_order(symbol, base_amt)
            log("⏳ Fetching order/trades to get filled price…")

            filled_price = None
            try:
                full_order = trading.fetch_order(order["id"], symbol)
            except Exception:
                full_order = order

            trades = full_order.get("trades") or full_order.get("info", {}).get("trades")
            if trades:
                try:
                    last_trade = trades[-1]
                    filled_price = float(
                        last_trade.get("price")
                        or last_trade.get("info", {}).get("price")
                    )
                except Exception:
                    filled_price = None

            if filled_price is None:
                if full_order.get("price"):
                    filled_price = float(full_order["price"])
                elif full_order.get("average"):
                    filled_price = float(full_order["average"])
                else:
                    filled_price = last

            entry = float(filled_price)
            log(f"✅ Bought {symbol} around {entry:.8f} (filled price)")

            if use_tp and tp_pct > 0:
                tp_price = make_tp_price(trading, symbol, entry, tp_pct)
                tp_amount = base_amt
                eff = (tp_price / entry - 1) * 100.0
                log(
                    f"📌 TP LIMIT SELL {symbol}: amount={tp_amount} @ {tp_price} "
                    f"(target {tp_pct:.4f}% | effective {eff:.4f}%)"
                )
                tp_order = trading.create_limit_sell_order(symbol, tp_amount, tp_price)
                log("✅ TP order placed.")

            if use_sl and sl_pct and sl_pct > 0:
                sl_price = entry * (1 - sl_pct / 100.0)
                log(f"🛑 Suggested SL @ {sl_price:.8f} (-{sl_pct:.2f}%) — NOT auto-placed.")

        except Exception as e:
            log(f"❌ Error buying {symbol}: {e}")

    log("\n✅ Auto-buy loop finished.")

# -------------------------
# MAIN ENTRY: runs screener, then auto-buy at 16:59:59 Lagos
# -------------------------
def main():
    lagos = pytz.timezone(LOCAL_TZ)
    now_lagos = datetime.now(lagos)
    log(f"Starting scheduled MEXC bot. Lagos time now: {now_lagos.strftime('%Y-%m-%d %H:%M:%S')}")

    results, fname, ex_public = run_screener(
        ex_id="mexc",
        quote="USDT",
        timeframe="1h",
        vol_min=DEFAULTS["vol_min"],
        vol_max=DEFAULTS["vol_max"],
        history_days=DEFAULTS["history_days"],
        target_list=parse_targets(DEFAULTS["target_str"]),
        anchor_mode="hour",
        anchor_hour=DEFAULTS["anchor_hour"],
        use_local_tz=True,
        fees_slip=DEFAULTS["fees_slip"],
    )

    if results is None or results.empty:
        log("No screener results, exiting.")
        return

    auto_buy_from_results(
        ex=ex_public,
        ex_id="mexc",
        results_df=results,
        ranks_to_buy="1",  # buy Rank 1 only
        mode_buy_size="pct",
        pct_of_balance=DEFAULTS["buy_pct_balance"],
        fixed_usdt=DEFAULTS["buy_fixed_usdt"],
        tp_pct=DEFAULTS["tp_pct"],
        use_tp=True,
        use_sl=False,
        sl_value=None,
        buy_time_str=DEFAULTS["buy_time"],  # waits until 16:59:59 Lagos
    )

if __name__ == "__main__":
    main()
