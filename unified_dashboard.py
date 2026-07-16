"""
Unified Signal Dashboard -- scans every setup studied in this project:
    SHORT signals: Bounce Short, First Red Day, Double Top, Head-and-Shoulders Top
    LONG signals:  Double Bottom, Head-and-Shoulders Bottom, Cup & Handle

Generates ONE modern, self-contained HTML dashboard (dashboard.html) with
live entry/stop/target for every signal firing today, plus the REAL win
rate / avg R stats from backtesting each setup on your own crypto data
(not copied from any book -- computed fresh, same rigor used throughout
this project).

HOW TO RUN
    pip install pandas numpy ccxt requests
    python unified_dashboard.py

WHERE TO SEE IT
    Open the generated dashboard.html on your Mac, or AirDrop / local-WiFi
    host it to view on your phone (same instructions as before).

UPDATE HISTORICAL_STATS BELOW whenever you re-run the individual backtest
scripts (backtest_bounce_short_traintest.py, backtest_v7_entryB_finetune.py,
classic_patterns_scanner.py) with fresh numbers.
"""

import time
import requests
import numpy as np
import pandas as pd
import ccxt

# ==========================================================================
# CONFIG
# ==========================================================================
QUOTE = "USDT"
MIN_MARKET_CAP = 1_000_000     # soft filter -- only excludes coins we KNOW are below this
MAX_COINGECKO_PAGES = 40        # for market-cap enrichment lookup (not a hard universe limit anymore)
SLEEP_BETWEEN_CALLS = 0.15

STABLECOIN_EXCLUDE = {"USDT", "USDC", "USDE", "FDUSD", "TUSD", "USDS", "USD1", "RLUSD",
                       "DAI", "BFUSD", "PYUSD", "GUSD", "USDP"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

# CoinGecko category slugs -> friendly label shown on cards
CATEGORY_SLUGS = {
    "layer-1": "Layer 1",
    "layer-2": "Layer 2",
    "decentralized-finance-defi": "DeFi",
    "gaming": "Gaming",
    "meme-token": "Meme",
    "artificial-intelligence": "AI",
    "oracle": "Oracle",
    "centralized-exchange-token-cex": "Exchange Token",
    "real-world-assets-rwa": "RWA",
    "depin": "DePIN",
}

PIVOT_WINDOW = 4
STOP_BUFFER_PCT = 0.5

# ---- Bounce Short (validated params) ----
RVOL_MIN_BS = 5.0
DORMANCY_DAYS_MIN_BS = 20
BOUNCE_PROXIMITY_PCT_BS = 8.0
MIN_DOLLAR_BLOCK_BS = 10_000_000
TARGET_FADE_PCT_BS = 75.0

# ---- First Red Day ----
FRD_MIN_GREEN_RUN = 3
FRD_RANGE_PCT_MIN_3DAY = 300.0
FRD_RANGE_PCT_MIN_2DAY = 1000.0
FRD_TARGET_FADE_PCT = 50.0

# ---- Double Top / Bottom ----
DT_PEAK_TOLERANCE_PCT = 4.0
DT_MIN_DEPTH_PCT = 8.0

# ---- Head-and-Shoulders ----
HS_HEAD_MIN_EXCESS_PCT = 3.0
HS_SHOULDER_TOLERANCE_PCT = 7.0

# ---- Cup & Handle ----
CH_MIN_CUP_DEPTH_PCT = 15.0
CH_RIM_TOLERANCE_PCT = 5.0
CH_HANDLE_MAX_DEPTH_PCT = 15.0
CH_HANDLE_MAX_BARS = 15

BIG_PUMP_THRESHOLD_PCT = 30.0   # daily candle % gain to flag for the watchlist

# REAL backtested stats (fill these in from your own script outputs --
# these particular numbers are from the pattern_backtest_summary.csv /
# pattern_backtest_by_regime.csv / bounce & EMA34 results seen so far)
HISTORICAL_STATS = {
    "Bounce Short": dict(win_rate=55.0, avg_r=0.15, trades=31, confidence="medium"),
    "First Red Day": dict(win_rate=25.0, avg_r=-0.31, trades=4, confidence="low (mẫu quá nhỏ, hướng âm)"),
    "Double Top": dict(win_rate=64.7, avg_r=0.108, trades=669, confidence="high"),
    "Head-and-Shoulders Top": dict(win_rate=50.8, avg_r=0.086, trades=242, confidence="medium"),
    "Double Bottom": dict(win_rate=None, avg_r=None, trades=0, confidence="chưa backtest -- chạy pattern_backtest cho bottom patterns"),
    "Head-and-Shoulders Bottom": dict(win_rate=None, avg_r=None, trades=0, confidence="chưa backtest"),
    "Cup and Handle": dict(win_rate=None, avg_r=None, trades=0, confidence="chưa backtest"),
}


# ==========================================================================
# COIN UNIVERSE
# ==========================================================================
def get_all_binance_usdt_symbols(exchange):
    """The actual scan universe: EVERY active USDT spot pair on Binance,
    minus stablecoins and leveraged tokens (3x/UP/DOWN etc, which distort
    price action and aren't real spot patterns)."""
    markets = exchange.load_markets()
    symbols = []
    for sym, m in markets.items():
        if not sym.endswith(f"/{QUOTE}"):
            continue
        if not m.get("spot", True) or not m.get("active", True):
            continue
        base = sym.split("/")[0]
        if base in STABLECOIN_EXCLUDE:
            continue
        if base.endswith(LEVERAGED_SUFFIXES):
            continue
        symbols.append(sym)
    return symbols


def get_coingecko_market_cap_map():
    """symbol (upper, no pair suffix) -> market cap. Built once from
    CoinGecko's market-cap-sorted list; used only for display/soft-filter
    enrichment, not to restrict which Binance coins get scanned."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    mcap_map = {}
    page = 1
    while page <= MAX_COINGECKO_PAGES:
        params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page}
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                print("  Rate limited, waiting 20s...")
                time.sleep(20)
                continue
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}, retrying in 15s...")
            time.sleep(15)
            continue
        data = resp.json()
        if not data:
            break
        for c in data:
            sym = c["symbol"].upper()
            if sym not in mcap_map and c.get("market_cap"):
                mcap_map[sym] = c["market_cap"]
        page += 1
        time.sleep(2)
    return mcap_map


def get_coingecko_category_map():
    """symbol (upper) -> list of category labels, e.g. ['Layer 1', 'DeFi']."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    cat_map = {}
    for slug, label in CATEGORY_SLUGS.items():
        for page in (1, 2):
            params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250,
                      "page": page, "category": slug}
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    time.sleep(15)
                    continue
                resp.raise_for_status()
            except requests.exceptions.RequestException:
                time.sleep(10)
                continue
            data = resp.json()
            if not data:
                break
            for c in data:
                sym = c["symbol"].upper()
                cat_map.setdefault(sym, [])
                if label not in cat_map[sym]:
                    cat_map[sym].append(label)
            time.sleep(1.5)
    return cat_map


def compute_volatility_stats(df):
    last50 = df.tail(50)
    if len(last50) < 10:
        return None, None
    adr_pct_50d = ((last50["high"] - last50["low"]) / last50["close"] * 100).mean()
    lo, hi = last50["low"].min(), last50["high"].max()
    range_pct_50d = (hi - lo) / lo * 100 if lo > 0 else None
    return adr_pct_50d, range_pct_50d


def compute_accumulation_stats(df, market_cap):
    """Volume/MarketCap turnover ratio + a simple accumulation heuristic:
    recent volume rising well above the prior period while price stays in
    a tight range can indicate quiet buying (accumulation) rather than a
    breakout move."""
    if len(df) < 30:
        return None, False
    recent = df.tail(14)
    prior = df.iloc[-28:-14] if len(df) >= 28 else None

    recent_dollar_vol = (recent["close"] * recent["volume"]).mean()
    vol_mcap_pct = (recent_dollar_vol / market_cap * 100) if market_cap else None

    is_accumulating = False
    if prior is not None and len(prior) > 0:
        prior_dollar_vol = (prior["close"] * prior["volume"]).mean()
        vol_rising = prior_dollar_vol > 0 and recent_dollar_vol / prior_dollar_vol >= 1.5
        price_range_pct = (recent["high"].max() - recent["low"].min()) / recent["low"].min() * 100
        is_accumulating = vol_rising and price_range_pct <= 15.0

    return vol_mcap_pct, is_accumulating


def tradingview_link(symbol):
    base = symbol.split("/")[0]
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{base}{QUOTE}"


def fetch_daily(exchange, symbol, limit=500):
    raw = exchange.fetch_ohlcv(symbol, timeframe="1d", limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def flag_pivots(series, window=PIVOT_WINDOW, mode="high"):
    values = series.values
    n = len(values)
    is_pivot = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        segment = values[i - window: i + window + 1]
        if mode == "high" and values[i] == segment.max():
            is_pivot[i] = True
        elif mode == "low" and values[i] == segment.min():
            is_pivot[i] = True
    return is_pivot


# ==========================================================================
# LIVE CHECKS -- SHORT SIGNALS
# ==========================================================================
def check_bounce_short(symbol, df):
    df = df.copy()
    df["rvol"] = df["volume"] / df["volume"].rolling(20).mean()
    df["dollar_vol"] = df["close"] * df["volume"]
    n = len(df)
    last_i = n - 1
    spike_i = None
    for i in range(last_i - 1, 20, -1):
        if df.at[i, "rvol"] >= RVOL_MIN_BS and df.at[i, "dollar_vol"] >= MIN_DOLLAR_BLOCK_BS:
            spike_i = i
            break
    if spike_i is None:
        return None
    spike_high = df.at[spike_i, "high"]
    proximity = spike_high * (1 - BOUNCE_PROXIMITY_PCT_BS / 100)
    prior = df["high"].iloc[spike_i + 1:last_i]
    if len(prior) == 0 or (prior >= proximity).any():
        return None
    dormancy = last_i - spike_i - 1
    if dormancy < DORMANCY_DAYS_MIN_BS:
        return None
    today_high = df.at[last_i, "high"]
    if today_high < proximity or today_high > spike_high * 1.02:
        return None
    if df.at[last_i, "close"] >= df.at[last_i, "open"]:
        return None
    entry = df.at[last_i, "close"]
    stop = spike_high * (1 + STOP_BUFFER_PCT / 100)
    risk = stop - entry
    if risk <= 0:
        return None
    bounce_low = df["low"].iloc[max(0, last_i - dormancy):last_i].min()
    target = spike_high - (spike_high - bounce_low) * (TARGET_FADE_PCT_BS / 100)
    why = (f"Vùng {spike_high:.6g} từng có volume đột biến lớn -- nhiều người mua bị kẹt hàng ở đó. "
           f"Giá đã nằm im {dormancy} ngày rồi vừa hồi lại đúng vùng này và bị từ chối (đóng đỏ) -- "
           f"áp lực bán từ nhóm kẹt hàng cũ đang đè giá xuống lại.")
    return dict(direction="SHORT", setup="Bounce Short", symbol=symbol, entry=entry, stop=stop,
                target=target, risk_pct=risk / entry * 100, why=why)


def check_first_red_day(symbol, df):
    df = df.copy()
    df["dollar_vol"] = df["close"] * df["volume"]
    df["is_green"] = df["close"] > df["open"]
    n = len(df)
    last_i = n - 1

    j = last_i - 1
    while j > 1 and not df.at[j, "is_green"]:
        j -= 1
    if j <= 1:
        return None
    run_end = j
    run_start = run_end
    while run_start - 1 >= 0 and df.at[run_start - 1, "is_green"] and \
            df.at[run_start - 1, "dollar_vol"] < df.at[run_start, "dollar_vol"]:
        run_start -= 1
    run_len = run_end - run_start + 1
    if run_len < FRD_MIN_GREEN_RUN:
        return None

    run_low = df["low"].iloc[run_start:run_end + 1].min()
    run_high = df["high"].iloc[run_start:run_end + 1].max()
    range_pct = (run_high - run_low) / run_low * 100
    min_range = FRD_RANGE_PCT_MIN_2DAY if run_len == 2 else FRD_RANGE_PCT_MIN_3DAY
    if range_pct < min_range:
        return None

    if run_end != last_i - 1:
        return None
    if df.at[last_i, "close"] >= df.at[last_i, "open"]:
        return None

    stage1_entry = df.at[run_end, "close"]
    stage2_entry = df.at[last_i, "close"]
    avg_entry = stage1_entry * 0.25 + stage2_entry * 0.75
    stop = run_high * (1 + STOP_BUFFER_PCT / 100)
    risk = stop - avg_entry
    if risk <= 0:
        return None
    target = run_high - (run_high - run_low) * (FRD_TARGET_FADE_PCT / 100)
    why = (f"Coin đã chạy {run_len} ngày xanh liên tục, tăng {range_pct:.0f}% -- dấu hiệu quá nóng. "
           f"Hôm nay là ngày đỏ đầu tiên sau chuỗi tăng đó -- xác nhận lực mua đã cạn, người mua FOMO cuối chuỗi đang lỗ.")
    return dict(direction="SHORT", setup="First Red Day", symbol=symbol, entry=avg_entry, stop=stop,
                target=target, risk_pct=risk / avg_entry * 100, why=why)


def check_double_top(symbol, df):
    ph = flag_pivots(df["high"], mode="high")
    pl = flag_pivots(df["low"], mode="low")
    n = len(df)
    highs = [i for i in range(n) if ph[i]]
    lows = [i for i in range(n) if pl[i]]
    for idx in range(len(highs) - 1):
        p1, p2 = highs[idx], highs[idx + 1]
        troughs = [t for t in lows if p1 < t < p2]
        if not troughs:
            continue
        trough = min(troughs, key=lambda t: df.at[t, "low"])
        p1h, p2h, tl = df.at[p1, "high"], df.at[p2, "high"], df.at[trough, "low"]
        if abs(p2h - p1h) / p1h * 100 > DT_PEAK_TOLERANCE_PCT:
            continue
        if (max(p1h, p2h) - tl) / max(p1h, p2h) * 100 < DT_MIN_DEPTH_PCT:
            continue
        if n - 1 <= p2:
            continue
        if df.at[n - 1, "close"] < tl and all(df.at[k, "close"] >= tl for k in range(p2 + 1, n - 1)):
            entry = df.at[n - 1, "close"]
            pattern_high = max(p1h, p2h)
            stop = pattern_high * (1 + STOP_BUFFER_PCT / 100)
            risk = stop - entry
            if risk <= 0:
                continue
            target = tl - (pattern_high - tl)
            why = (f"Giá tạo 2 đỉnh gần bằng nhau ({pattern_high:.6g}) không vượt qua được -- lực mua cạn "
                   f"lần 2. Vừa phá xuống dưới đáy giữa 2 đỉnh ({tl:.6g}) -- xác nhận đảo chiều.")
            return dict(direction="SHORT", setup="Double Top", symbol=symbol, entry=entry, stop=stop,
                        target=target, risk_pct=risk / entry * 100, why=why)
    return None


def check_head_shoulders_top(symbol, df):
    ph = flag_pivots(df["high"], mode="high")
    pl = flag_pivots(df["low"], mode="low")
    n = len(df)
    highs = [i for i in range(n) if ph[i]]
    lows = [i for i in range(n) if pl[i]]
    for idx in range(len(highs) - 2):
        ls, h, rs = highs[idx], highs[idx + 1], highs[idx + 2]
        lsh, hh, rsh = df.at[ls, "high"], df.at[h, "high"], df.at[rs, "high"]
        excess = min((hh - lsh) / lsh * 100, (hh - rsh) / rsh * 100)
        if excess < HS_HEAD_MIN_EXCESS_PCT or abs(rsh - lsh) / lsh * 100 > HS_SHOULDER_TOLERANCE_PCT:
            continue
        t1s = [t for t in lows if ls < t < h]
        t2s = [t for t in lows if h < t < rs]
        if not t1s or not t2s:
            continue
        neckline = (df.at[min(t1s, key=lambda t: df.at[t, "low"]), "low"] +
                    df.at[min(t2s, key=lambda t: df.at[t, "low"]), "low"]) / 2
        if n - 1 <= rs:
            continue
        if df.at[n - 1, "close"] < neckline and all(df.at[k, "close"] >= neckline for k in range(rs + 1, n - 1)):
            entry = df.at[n - 1, "close"]
            stop = rsh * (1 + STOP_BUFFER_PCT / 100)
            risk = stop - entry
            if risk <= 0:
                continue
            target = neckline - (hh - neckline)
            why = (f"3 đỉnh, đỉnh giữa ({hh:.6g}) cao nhất nhưng đỉnh phải không vượt được -- lực mua yếu dần. "
                   f"Vừa phá xuống dưới neckline ({neckline:.6g}) -- xác nhận đảo chiều kinh điển.")
            return dict(direction="SHORT", setup="Head-and-Shoulders Top", symbol=symbol, entry=entry,
                        stop=stop, target=target, risk_pct=risk / entry * 100, why=why)
    return None


# ==========================================================================
# LIVE CHECKS -- LONG SIGNALS (mirror geometry)
# ==========================================================================
def check_double_bottom(symbol, df):
    ph = flag_pivots(df["high"], mode="high")
    pl = flag_pivots(df["low"], mode="low")
    n = len(df)
    highs = [i for i in range(n) if ph[i]]
    lows = [i for i in range(n) if pl[i]]
    for idx in range(len(lows) - 1):
        t1, t2 = lows[idx], lows[idx + 1]
        peaks = [p for p in highs if t1 < p < t2]
        if not peaks:
            continue
        peak = max(peaks, key=lambda p: df.at[p, "high"])
        t1l, t2l, ph_v = df.at[t1, "low"], df.at[t2, "low"], df.at[peak, "high"]
        if abs(t2l - t1l) / t1l * 100 > DT_PEAK_TOLERANCE_PCT:
            continue
        if (ph_v - min(t1l, t2l)) / min(t1l, t2l) * 100 < DT_MIN_DEPTH_PCT:
            continue
        if n - 1 <= peak:
            continue
        if df.at[n - 1, "close"] > ph_v and all(df.at[k, "close"] <= ph_v for k in range(peak + 1, n - 1)):
            entry = df.at[n - 1, "close"]
            pattern_low = min(t1l, t2l)
            stop = pattern_low * (1 - STOP_BUFFER_PCT / 100)
            risk = entry - stop
            if risk <= 0:
                continue
            target = ph_v + (ph_v - pattern_low)
            why = (f"Giá tạo 2 đáy gần bằng nhau ({pattern_low:.6g}) không phá thấp hơn được -- lực bán cạn "
                   f"lần 2. Vừa phá lên trên đỉnh giữa 2 đáy ({ph_v:.6g}) -- xác nhận đảo chiều tăng.")
            return dict(direction="LONG", setup="Double Bottom", symbol=symbol, entry=entry, stop=stop,
                        target=target, risk_pct=risk / entry * 100, why=why)
    return None


def check_head_shoulders_bottom(symbol, df):
    ph = flag_pivots(df["high"], mode="high")
    pl = flag_pivots(df["low"], mode="low")
    n = len(df)
    highs = [i for i in range(n) if ph[i]]
    lows = [i for i in range(n) if pl[i]]
    for idx in range(len(lows) - 2):
        ls, h, rs = lows[idx], lows[idx + 1], lows[idx + 2]
        lsl, hl, rsl = df.at[ls, "low"], df.at[h, "low"], df.at[rs, "low"]
        excess = min((lsl - hl) / hl * 100, (rsl - hl) / hl * 100)
        if excess < HS_HEAD_MIN_EXCESS_PCT or abs(rsl - lsl) / lsl * 100 > HS_SHOULDER_TOLERANCE_PCT:
            continue
        p1s = [p for p in highs if ls < p < h]
        p2s = [p for p in highs if h < p < rs]
        if not p1s or not p2s:
            continue
        neckline = (df.at[max(p1s, key=lambda p: df.at[p, "high"]), "high"] +
                    df.at[max(p2s, key=lambda p: df.at[p, "high"]), "high"]) / 2
        if n - 1 <= rs:
            continue
        if df.at[n - 1, "close"] > neckline and all(df.at[k, "close"] <= neckline for k in range(rs + 1, n - 1)):
            entry = df.at[n - 1, "close"]
            stop = rsl * (1 - STOP_BUFFER_PCT / 100)
            risk = entry - stop
            if risk <= 0:
                continue
            target = neckline + (neckline - hl)
            why = (f"3 đáy, đáy giữa ({hl:.6g}) thấp nhất nhưng đáy phải không phá thấp hơn -- lực bán yếu dần. "
                   f"Vừa phá lên trên neckline ({neckline:.6g}) -- xác nhận đảo chiều tăng kinh điển.")
            return dict(direction="LONG", setup="Head-and-Shoulders Bottom", symbol=symbol, entry=entry,
                        stop=stop, target=target, risk_pct=risk / entry * 100, why=why)
    return None


def check_big_pump_watch(symbol, df, threshold_pct=BIG_PUMP_THRESHOLD_PCT):
    """Not a trade signal -- just flags coins whose most recent daily candle
    pumped >= threshold_pct, so you can watch them for a future Bounce Short
    setup once they go dormant and later fail a retest of this level."""
    if len(df) < 2:
        return None
    n = len(df)
    last_i = n - 1
    prev_close = df.at[last_i - 1, "close"]
    today_close = df.at[last_i, "close"]
    if prev_close <= 0:
        return None
    pump_pct = (today_close - prev_close) / prev_close * 100
    if pump_pct < threshold_pct:
        return None
    return dict(symbol=symbol, pump_pct=pump_pct, day_high=df.at[last_i, "high"],
                day_low=df.at[last_i, "low"], day_close=today_close)


def check_cup_and_handle(symbol, df):
    ph = flag_pivots(df["high"], mode="high")
    pl = flag_pivots(df["low"], mode="low")
    n = len(df)
    highs = [i for i in range(n) if ph[i]]
    lows = [i for i in range(n) if pl[i]]
    for idx in range(len(highs) - 1):
        left_rim = highs[idx]
        candidates_bottom = [b for b in lows if b > left_rim]
        if not candidates_bottom:
            continue
        bottom = min(candidates_bottom[:5], key=lambda b: df.at[b, "low"]) if candidates_bottom else None
        if bottom is None:
            continue
        left_rim_h, bottom_l = df.at[left_rim, "high"], df.at[bottom, "low"]
        if (left_rim_h - bottom_l) / left_rim_h * 100 < CH_MIN_CUP_DEPTH_PCT:
            continue
        right_rim_candidates = [h for h in highs if h > bottom]
        if not right_rim_candidates:
            continue
        right_rim = right_rim_candidates[0]
        right_rim_h = df.at[right_rim, "high"]
        if abs(right_rim_h - left_rim_h) / left_rim_h * 100 > CH_RIM_TOLERANCE_PCT:
            continue

        handle_end = min(right_rim + CH_HANDLE_MAX_BARS, n - 1)
        if handle_end <= right_rim:
            continue
        handle_low = df["low"].iloc[right_rim + 1:handle_end + 1].min() if handle_end > right_rim else None
        if handle_low is None or pd.isna(handle_low):
            continue
        handle_depth_pct = (right_rim_h - handle_low) / right_rim_h * 100
        if handle_depth_pct > CH_HANDLE_MAX_DEPTH_PCT or handle_depth_pct < 0:
            continue

        rim_level = max(left_rim_h, right_rim_h)
        if n - 1 <= right_rim or n - 1 > handle_end + 5:
            continue
        if df.at[n - 1, "close"] > rim_level:
            entry = df.at[n - 1, "close"]
            stop = handle_low * (1 - STOP_BUFFER_PCT / 100)
            risk = entry - stop
            if risk <= 0:
                continue
            cup_depth = left_rim_h - bottom_l
            target = entry + cup_depth
            why = (f"Giá giảm sâu tạo hình chữ U ({(left_rim_h - bottom_l) / left_rim_h * 100:.0f}% từ đỉnh) rồi "
                   f"hồi phục lại gần đỉnh cũ ({rim_level:.6g}), sau đó có nhịp điều chỉnh nhẹ (handle) -- giờ giá "
                   f"vừa phá lên trên đỉnh -- xác nhận breakout tiếp diễn tăng.")
            return dict(direction="LONG", setup="Cup and Handle", symbol=symbol, entry=entry, stop=stop,
                        target=target, risk_pct=risk / entry * 100, why=why)
    return None


CHECK_FUNCTIONS = [
    check_bounce_short, check_first_red_day, check_double_top, check_head_shoulders_top,
    check_double_bottom, check_head_shoulders_bottom, check_cup_and_handle,
]


# ==========================================================================
# HTML DASHBOARD (modern design)
# ==========================================================================
def build_html(signals, all_coins, watchlist, scanned_at, n_coins):
    shorts = [s for s in signals if s["direction"] == "SHORT"]
    longs = [s for s in signals if s["direction"] == "LONG"]

    def stat_badge(setup):
        s = HISTORICAL_STATS.get(setup, {})
        if not s or s.get("win_rate") is None:
            return f'<span class="badge badge-muted">Chưa có thống kê ({s.get("confidence","")})</span>'
        wr = s["win_rate"]
        color = "badge-good" if wr >= 55 else ("badge-mid" if wr >= 45 else "badge-bad")
        return (f'<span class="badge {color}">{wr:.0f}% win rate</span>'
                f'<span class="badge badge-muted">avg_r {s["avg_r"]:+.2f}</span>'
                f'<span class="badge badge-muted">{s["trades"]} lệnh lịch sử</span>')

    def signal_card(s):
        dirclass = "dir-short" if s["direction"] == "SHORT" else "dir-long"
        avatar_txt = s['symbol'].replace('/USDT', '')[:2]
        entry_val_class = "v-amber"
        stop_val_class = "v-short"
        target_val_class = "v-long" if s["direction"] == "SHORT" else "v-long"
        if s["direction"] == "LONG":
            stop_val_class, target_val_class = "v-short", "v-long"

        mcap_txt = f"${s['market_cap']:,.0f}" if s.get("market_cap") else "N/A"
        adr_txt = f"{s['adr_pct_50d']:.1f}%" if s.get("adr_pct_50d") else "N/A"
        range_txt = f"{s['range_pct_50d']:.0f}%" if s.get("range_pct_50d") else "N/A"
        vol_mcap_txt = f"{s['vol_mcap_pct']:.2f}%" if s.get("vol_mcap_pct") else "N/A"
        accum_badge = '<span class="badge badge-good">🔵 Gom hàng</span>' if s.get("is_accumulating") else ""

        return f"""
        <div class="signal-card {dirclass}">
          <div class="signal-head">
            <div class="avatar">{avatar_txt}</div>
            <div class="head-text">
              <span class="coin">{s['symbol'].replace('/USDT','')}</span>
              <span class="setup-tag">{s['setup']}</span>
            </div>
            <a class="tv-link" href="{s.get('tv_link','#')}" target="_blank" rel="noopener">TradingView ↗</a>
          </div>

          <div class="badge-row">{stat_badge(s['setup'])}
            <span class="badge badge-cat">{s.get('category','N/A')}</span>
            {accum_badge}
          </div>

          <div class="stat-grid trade-grid">
            <div class="stat-block"><span class="stat-label">Stop</span><span class="stat-value {stop_val_class}">{s['stop']:.6g}</span></div>
            <div class="stat-block"><span class="stat-label">Entry</span><span class="stat-value {entry_val_class}">{s['entry']:.6g}</span></div>
            <div class="stat-block"><span class="stat-label">Target</span><span class="stat-value {target_val_class}">{s['target']:.6g}</span></div>
          </div>

          <div class="stat-grid">
            <div class="stat-block"><span class="stat-label">Risk</span><span class="stat-value">{s['risk_pct']:.1f}%</span></div>
            <div class="stat-block"><span class="stat-label">Market Cap</span><span class="stat-value">{mcap_txt}</span></div>
            <div class="stat-block"><span class="stat-label">ADR 50d</span><span class="stat-value">{adr_txt}</span></div>
            <div class="stat-block"><span class="stat-label">Range 50d</span><span class="stat-value">{range_txt}</span></div>
            <div class="stat-block"><span class="stat-label">Vol/MC</span><span class="stat-value">{vol_mcap_txt}</span></div>
          </div>

          <p class="why">{s['why']}</p>
        </div>"""

    def section(items, empty_msg):
        cards = "".join(signal_card(s) for s in items) if items else f'<p class="empty">{empty_msg}</p>'
        return f'<div class="signals-grid">{cards}</div>'

    def coin_row(c):
        mcap_txt = f"${c['market_cap']:,.0f}" if c.get("market_cap") else "N/A"
        adr_txt = f"{c['adr_pct_50d']:.1f}%" if c.get("adr_pct_50d") else "N/A"
        range_txt = f"{c['range_pct_50d']:.0f}%" if c.get("range_pct_50d") else "N/A"
        vol_mcap_txt = f"{c['vol_mcap_pct']:.2f}%" if c.get("vol_mcap_pct") else "N/A"
        accum_txt = "🔵 Gom hàng" if c.get("is_accumulating") else "-"
        coin_name = c["symbol"].replace("/USDT", "")
        return (f'<tr><td class="mono">{coin_name}</td><td>{c.get("category","N/A")}</td>'
                f'<td class="mono">{c["last_price"]:.6g}</td><td class="mono">{mcap_txt}</td>'
                f'<td class="mono">{adr_txt}</td><td class="mono">{range_txt}</td>'
                f'<td class="mono">{vol_mcap_txt}</td><td>{accum_txt}</td>'
                f'<td><a class="tv-link" href="{c["tv_link"]}" target="_blank" rel="noopener">Xem chart ↗</a></td></tr>')

    def watch_row(w):
        mcap_txt = f"${w['market_cap']:,.0f}" if w.get("market_cap") else "N/A"
        adr_txt = f"{w['adr_pct_50d']:.1f}%" if w.get("adr_pct_50d") else "N/A"
        range_txt = f"{w['range_pct_50d']:.0f}%" if w.get("range_pct_50d") else "N/A"
        vol_mcap_txt = f"{w['vol_mcap_pct']:.2f}%" if w.get("vol_mcap_pct") else "N/A"
        coin_name = w["symbol"].replace("/USDT", "")
        return (f'<tr><td class="mono">{coin_name}</td><td class="mono">+{w["pump_pct"]:.1f}%</td>'
                f'<td>{w.get("category","N/A")}</td><td class="mono">{mcap_txt}</td>'
                f'<td class="mono">{adr_txt}</td><td class="mono">{range_txt}</td>'
                f'<td class="mono">{vol_mcap_txt}</td>'
                f'<td><a class="tv-link" href="{w["tv_link"]}" target="_blank" rel="noopener">Xem chart ↗</a></td></tr>')

    watch_rows = "".join(watch_row(w) for w in sorted(watchlist, key=lambda x: -x["pump_pct"]))

    coin_rows = "".join(coin_row(c) for c in sorted(all_coins, key=lambda x: x["symbol"]))

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Signal Terminal</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg1: #030407; --bg2: #0a0c12; --bg3: #0d0f16;
    --surface: rgba(255,255,255,0.035); --surface-solid: #101218; --surface2: rgba(255,255,255,0.06);
    --border: rgba(255,255,255,0.08); --text: #F1F1F7; --text-dim: #9295B3; --amber: #FBBF24;
    --short: #F0616B; --short-soft: rgba(240,97,107,0.14); --long: #34D399; --long-soft: rgba(52,211,153,0.14);
    --purple: #8B5CF6;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; min-height:100vh; color:var(--text); font-family:'Inter',sans-serif;
          padding:16px; background:linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 45%, var(--bg3) 100%);
          background-attachment:fixed; }}
  .topbar {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;
             padding-bottom:16px; margin-bottom:16px; }}
  .brand {{ font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:700; letter-spacing:-0.5px; }}
  .brand span {{ color:var(--purple); }}
  .meta {{ font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text-dim); }}
  .tabs {{ display:flex; gap:6px; margin-bottom:20px; overflow-x:auto; padding-bottom:4px; -webkit-overflow-scrolling:touch; }}
  .tab-btn {{ background:var(--surface); border:1px solid var(--border); color:var(--text-dim);
              font-family:'JetBrains Mono',monospace; font-size:12px; padding:9px 14px; border-radius:10px;
              cursor:pointer; white-space:nowrap; flex-shrink:0; }}
  .tab-btn.active {{ color:var(--text); background:var(--surface2); border-color:var(--purple); }}
  .tab-panel {{ display:none; }}
  .tab-panel.active {{ display:block; }}
  h2 {{ font-family:'JetBrains Mono',monospace; font-size:13px; text-transform:uppercase; letter-spacing:1px;
        color:var(--text-dim); margin:0 0 14px; }}
  .signals-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(300px,1fr)); gap:14px; }}
  .signal-card {{ background:var(--surface); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
                  border:1px solid var(--border); border-radius:16px; padding:18px; }}
  .signal-card.dir-short {{ box-shadow:inset 3px 0 0 var(--short); }}
  .signal-card.dir-long {{ box-shadow:inset 3px 0 0 var(--long); }}
  .signal-head {{ display:flex; align-items:center; gap:10px; margin-bottom:14px; flex-wrap:wrap; }}
  .avatar {{ width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center;
             font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:700; flex-shrink:0; }}
  .dir-short .avatar {{ background:var(--short-soft); color:var(--short); }}
  .dir-long .avatar {{ background:var(--long-soft); color:var(--long); }}
  .head-text {{ display:flex; flex-direction:column; gap:2px; flex:1; min-width:0; }}
  .coin {{ font-family:'JetBrains Mono',monospace; font-size:16px; font-weight:700; }}
  .setup-tag {{ font-size:11px; color:var(--text-dim); }}
  .tv-link {{ font-size:11px; color:var(--purple); text-decoration:none; font-family:'JetBrains Mono',monospace;
              white-space:nowrap; }}
  .tv-link:hover {{ text-decoration:underline; }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(2, 1fr); gap:12px 16px; margin:14px 0;
                padding:14px 0; border-top:1px solid var(--border); border-bottom:1px solid var(--border); }}
  .stat-grid.trade-grid {{ grid-template-columns:repeat(3, 1fr); }}
  .stat-block {{ display:flex; flex-direction:column; gap:4px; min-width:0; }}
  .stat-label {{ font-size:10px; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.5px; }}
  .stat-value {{ font-size:14px; font-weight:700; font-family:'JetBrains Mono',monospace; overflow:hidden;
                 text-overflow:ellipsis; white-space:nowrap; }}
  .stat-value.v-short {{ color:var(--short); }}
  .stat-value.v-long {{ color:var(--long); }}
  .stat-value.v-amber {{ color:var(--amber); }}
  .badge {{ font-size:10.5px; padding:3px 9px; border-radius:20px; font-family:'JetBrains Mono',monospace;
            display:inline-block; }}
  .badge-good {{ background:var(--long-soft); color:var(--long); }}
  .badge-mid {{ background:rgba(251,191,36,0.14); color:var(--amber); }}
  .badge-bad {{ background:var(--short-soft); color:var(--short); }}
  .badge-cat {{ background:rgba(139,92,246,0.14); color:#c4b5fd; }}
  .badge-row {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }}
  .why {{ font-size:12.5px; color:var(--text-dim); line-height:1.55; margin:0; }}
  .empty {{ color:var(--text-dim); font-size:13px; grid-column:1/-1; }}
  .search-box {{ width:100%; max-width:320px; background:var(--surface); border:1px solid var(--border);
                 color:var(--text); padding:10px 14px; border-radius:10px; font-family:'JetBrains Mono',monospace;
                 font-size:13px; margin-bottom:14px; }}
  .search-box:focus {{ outline:1px solid var(--purple); }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  th, td {{ text-align:left; padding:10px 12px; border-bottom:1px solid var(--border); }}
  th {{ color:var(--text-dim); font-family:'JetBrains Mono',monospace; font-size:10.5px; text-transform:uppercase;
        position:sticky; top:0; background:var(--bg1); cursor:pointer; white-space:nowrap; }}
  td.mono {{ font-family:'JetBrains Mono',monospace; }}
  tr:hover {{ background:var(--surface); }}
  .table-wrap {{ overflow-x:auto; max-height:70vh; overflow-y:auto; border:1px solid var(--border);
                 border-radius:14px; background:var(--surface); backdrop-filter:blur(14px); }}

  @media (max-width: 640px) {{
    body {{ padding:10px; }}
    .brand {{ font-size:17px; }}
    .signals-grid {{ grid-template-columns:1fr; gap:10px; }}
    .stat-grid, .stat-grid.trade-grid {{ grid-template-columns:repeat(2, 1fr); }}
    .signal-card {{ padding:14px; border-radius:14px; }}
    th, td {{ padding:8px; font-size:11.5px; }}
    .tab-btn {{ font-size:11px; padding:8px 11px; }}
  }}
</style>
</head>
<body>
<div class="topbar">
  <div class="brand">SIGNAL<span>_</span>TERMINAL</div>
  <div class="meta">Quét lúc {scanned_at} · {n_coins} coin</div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('short', this)">🔴 Short signals ({len(shorts)})</button>
  <button class="tab-btn" onclick="switchTab('long', this)">🟢 Long signals ({len(longs)})</button>
  <button class="tab-btn" onclick="switchTab('watch', this)">👀 Theo dõi (Pump &gt;{BIG_PUMP_THRESHOLD_PCT:.0f}%) ({len(watchlist)})</button>
  <button class="tab-btn" onclick="switchTab('lookup', this)">📋 Tra cứu tất cả coin ({len(all_coins)})</button>
</div>

<div id="tab-short" class="tab-panel active">
  {section(shorts, "Không có tín hiệu short hôm nay")}
</div>

<div id="tab-long" class="tab-panel">
  {section(longs, "Không có tín hiệu long hôm nay")}
</div>

<div id="tab-watch" class="tab-panel">
  <p class="empty" style="margin-bottom:12px;">Coin có nến ngày tăng &gt;{BIG_PUMP_THRESHOLD_PCT:.0f}% -- không phải tín hiệu vào lệnh, chỉ để theo dõi: coin cần
  "nằm im" một thời gian rồi hồi lại test đúng vùng đỉnh này và bị từ chối mới thành tín hiệu Bounce Short thật.</p>
  <div class="table-wrap">
    <table id="watchTable">
      <thead><tr>
        <th onclick="sortTable(0,'watchTable')">Coin <span id="arrow_watchTable_0"></span></th>
        <th onclick="sortTable(1,'watchTable')">% tăng hôm nay <span id="arrow_watchTable_1"></span></th>
        <th onclick="sortTable(2,'watchTable')">Category <span id="arrow_watchTable_2"></span></th>
        <th onclick="sortTable(3,'watchTable')">Market Cap <span id="arrow_watchTable_3"></span></th>
        <th onclick="sortTable(4,'watchTable')">ADR 50d <span id="arrow_watchTable_4"></span></th>
        <th onclick="sortTable(5,'watchTable')">Range 50d <span id="arrow_watchTable_5"></span></th>
        <th onclick="sortTable(6,'watchTable')">Vol/MC <span id="arrow_watchTable_6"></span></th>
        <th>Chart</th>
      </tr></thead>
      <tbody>{watch_rows if watchlist else '<tr><td colspan="8" style="text-align:center;color:#888;">Không có coin nào pump hôm nay</td></tr>'}</tbody>
    </table>
  </div>
</div>

<div id="tab-lookup" class="tab-panel">
  <input type="text" id="coinSearch" class="search-box" placeholder="Gõ mã coin để tìm nhanh... (ví dụ BTC)" onkeyup="filterCoins()">
  <div class="table-wrap">
    <table id="coinTable">
      <thead><tr>
        <th onclick="sortTable(0,'coinTable')">Coin <span id="arrow_coinTable_0"></span></th>
        <th onclick="sortTable(1,'coinTable')">Category <span id="arrow_coinTable_1"></span></th>
        <th onclick="sortTable(2,'coinTable')">Giá <span id="arrow_coinTable_2"></span></th>
        <th onclick="sortTable(3,'coinTable')">Market Cap <span id="arrow_coinTable_3"></span></th>
        <th onclick="sortTable(4,'coinTable')">ADR 50d <span id="arrow_coinTable_4"></span></th>
        <th onclick="sortTable(5,'coinTable')">Range 50d <span id="arrow_coinTable_5"></span></th>
        <th onclick="sortTable(6,'coinTable')">Vol/MC <span id="arrow_coinTable_6"></span></th>
        <th onclick="sortTable(7,'coinTable')">Gom hàng? <span id="arrow_coinTable_7"></span></th>
        <th>Chart</th>
      </tr></thead>
      <tbody>{coin_rows}</tbody>
    </table>
  </div>
</div>

<script>
let sortDirs = {{}};
function sortTable(colIdx, tableId) {{
  const table = document.getElementById(tableId);
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const key = tableId + '_' + colIdx;
  sortDirs[key] = !sortDirs[key];
  const dir = sortDirs[key];
  rows.sort((a, b) => {{
    let x = (a.children[colIdx].textContent || '').trim();
    let y = (b.children[colIdx].textContent || '').trim();
    let xNum = parseFloat(x.replace(/[$,%+]/g, ''));
    let yNum = parseFloat(y.replace(/[$,%+]/g, ''));
    if (!isNaN(xNum) && !isNaN(yNum)) {{
      return dir ? xNum - yNum : yNum - xNum;
    }}
    return dir ? x.localeCompare(y) : y.localeCompare(x);
  }});
  rows.forEach(r => tbody.appendChild(r));
  table.querySelectorAll('th span').forEach(el => el.textContent = '');
  const arrowEl = document.getElementById('arrow_' + tableId + '_' + colIdx);
  if (arrowEl) arrowEl.textContent = dir ? '▲' : '▼';
}}
function switchTab(name, btn) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}}
function filterCoins() {{
  const q = document.getElementById('coinSearch').value.toUpperCase();
  document.querySelectorAll('#coinTable tbody tr').forEach(row => {{
    row.style.display = row.children[0].textContent.toUpperCase().includes(q) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    exchange = ccxt.binance()
    print("Fetching ALL Binance USDT pairs (the actual scan universe)...")
    symbols = get_all_binance_usdt_symbols(exchange)
    print(f"  {len(symbols)} coins on Binance (after excluding stablecoins/leveraged tokens).")

    print("Fetching CoinGecko market cap map (enrichment)...")
    mcap_map = get_coingecko_market_cap_map()
    print(f"  Got market cap for {len(mcap_map)} symbols.")

    print("Fetching CoinGecko category map (enrichment)...")
    cat_map = get_coingecko_category_map()
    print(f"  Got categories for {len(cat_map)} symbols.")

    print(f"Scanning {len(symbols)} coins across 7 setups + pump watchlist...")
    signals = []
    all_coins = []
    watchlist = []
    for idx, symbol in enumerate(symbols, 1):
        try:
            base = symbol.split("/")[0]
            mcap = mcap_map.get(base)
            if mcap is not None and mcap < MIN_MARKET_CAP:
                continue  # only exclude when we're SURE it's below the floor
            df = fetch_daily(exchange, symbol)
            if len(df) < 80:
                continue
            adr_pct_50d, range_pct_50d = compute_volatility_stats(df)
            vol_mcap_pct, is_accumulating = compute_accumulation_stats(df, mcap)
            category = ", ".join(cat_map.get(base, [])) or "N/A"
            tv_link = tradingview_link(symbol)

            all_coins.append(dict(
                symbol=symbol, category=category, market_cap=mcap,
                adr_pct_50d=adr_pct_50d, range_pct_50d=range_pct_50d,
                vol_mcap_pct=vol_mcap_pct, is_accumulating=is_accumulating,
                tv_link=tv_link, last_price=df["close"].iloc[-1],
            ))

            pump = check_big_pump_watch(symbol, df)
            if pump:
                pump["category"] = category
                pump["market_cap"] = mcap
                pump["adr_pct_50d"] = adr_pct_50d
                pump["range_pct_50d"] = range_pct_50d
                pump["vol_mcap_pct"] = vol_mcap_pct
                pump["tv_link"] = tv_link
                watchlist.append(pump)

            for fn in CHECK_FUNCTIONS:
                r = fn(symbol, df)
                if r:
                    r["market_cap"] = mcap
                    r["category"] = category
                    r["adr_pct_50d"] = adr_pct_50d
                    r["range_pct_50d"] = range_pct_50d
                    r["vol_mcap_pct"] = vol_mcap_pct
                    r["is_accumulating"] = is_accumulating
                    r["tv_link"] = tv_link
                    signals.append(r)
            if idx % 30 == 0:
                print(f"  ...{idx}/{len(symbols)} scanned")
        except Exception as e:
            print(f"  {symbol}: skipped ({e})")
        time.sleep(SLEEP_BETWEEN_CALLS)

    scanned_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    html = build_html(signals, all_coins, watchlist, scanned_at, len(symbols))
    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nSaved dashboard.html -- {len(signals)} signals, {len(watchlist)} pump watchlist, "
          f"{len(all_coins)} coins in lookup tab.")


if __name__ == "__main__":
    main()
