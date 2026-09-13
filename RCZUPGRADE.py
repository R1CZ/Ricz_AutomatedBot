# RCZ_v4_adaptive.py
# Full adaptive upgrade of RCZ_v3.py
# Objective: adaptive signal generation, adaptive component weights, adaptive thresholds,
# entry timing, freshness, microstructure revalidation, and preserved fast operation.

import MetaTrader5 as mt5
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
import traceback
import json
import os
import math
from datetime import datetime, timezone
from collections import deque

#============================================================================
# GLOBAL CONFIGURATION
#============================================================================

SYMBOL            = "XAUUSDm"
TIMEFRAME         = mt5.TIMEFRAME_M1
HTF_M5            = mt5.TIMEFRAME_M5
HTF_M15           = mt5.TIMEFRAME_M15

MAGIC_NUMBER      = 999777
LOT_SIZE          = 0.05
DEVIATION         = 5
XAUUSD_POINT      = 0.01
FILLING_MODE      = mt5.ORDER_FILLING_IOC

# Safety / risk limits remain fixed because they are protective boundaries.
MAX_DAILY_LOSS_USD    = 50.0
MAX_TRADES_PER_DAY    = 30
COOLDOWN_SECONDS      = 45

FAST_LOOP_SECONDS          = 0.05
SLOW_ANALYSIS_SECONDS      = 0.65
HTF_REFRESH_SECONDS        = 4.0
MIN_HOLD_SECONDS           = 12
LIVE_STATE_LOG_SECONDS     = 2.0
LOG_LIVE_STATE             = True

LEARNING_FILE      = "apex_learning_v8.json"
OPEN_META_FILE     = "apex_open_meta_v8.json"

MAX_RECORDS        = 1400
STAT_WINDOW        = 850
HALF_LIFE_HOURS    = 36.0

WEIGHT_MIN         = 0.55
WEIGHT_MAX         = 1.65
THRESH_MIN         = 0.47
THRESH_MAX         = 0.78

LOGISTIC_DIM       = 10

PARTIAL_PROFIT_ENABLE = True
PARTIAL_CLOSE_1_PCT   = 0.45
PARTIAL_CLOSE_2_PCT   = 0.50
PARTIAL_1_R           = 0.35
PARTIAL_2_R           = 0.70
EARLY_BE_R            = 0.45
PROFIT_LOCK_R         = 0.85
TRAIL_START_R         = 1.35
PROFIT_PROTECT_FAST_SECONDS = 0.25

COMPONENTS = [
    "structure",
    "alignment",
    "momentum",
    "order_flow",
    "volatility",
    "smc",
    "divergence",
    "impulse",
    "pullback",
    "candle",
    "micro",
]

#============================================================================
# UTILS
#============================================================================

def log(level, msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [{level.upper():5}] {msg}")

def safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        if isinstance(value, (float, int, np.floating, np.integer)):
            if np.isnan(value):
                return float(default)
            return float(value)
        v = float(value)
        if math.isnan(v):
            return float(default)
        return v
    except Exception:
        return float(default)

def safe_int(value, default=0):
    try:
        return int(safe_float(value, float(default)))
    except Exception:
        return int(default)

def clamp(value, lo, hi):
    return max(float(lo), min(float(hi), float(value)))

def percentile(values, p):
    try:
        arr = np.asarray(list(values), dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return 0.0
        return float(np.percentile(arr, p))
    except Exception:
        return 0.0

def pct_rank(arr, val):
    try:
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 20:
            return 0.5
        return float(np.mean(arr <= val))
    except Exception:
        return 0.5

def sigmoid(x):
    x = clamp(x, -8.0, 8.0)
    return 1.0 / (1.0 + math.exp(-x))

def recency_weight(ts, half_life_hours=HALF_LIFE_HOURS):
    try:
        now = time.time()
        ts = safe_float(ts, now)
        if ts <= 0:
            ts = now
        age_h = max(0.0, (now - ts) / 3600.0)
        hl = max(1.0, float(half_life_hours))
        return float(0.5 ** (age_h / hl))
    except Exception:
        return 1.0

def json_default(obj):
    try:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)
    except Exception:
        return str(obj)

def get_session_name() -> str:
    now = datetime.now(timezone.utc)
    t_min = now.hour * 60 + now.minute
    if 11 * 60 <= t_min < 13 * 60:
        return "OVERLAP"
    if 7 * 60 <= t_min < 12 * 60:
        return "LONDON"
    if 12 * 60 <= t_min < 20 * 60:
        return "NEW_YORK"
    if 0 <= t_min < 7 * 60:
        return "ASIAN"
    return "OFF_PEAK"

def check_filling_mode(symbol):
    info = mt5.symbol_info(symbol)
    if not info:
        return mt5.ORDER_FILLING_FOK
    m = info.filling_mode
    if m & 2:
        return mt5.ORDER_FILLING_IOC
    elif m & 1:
        return mt5.ORDER_FILLING_FOK
    else:
        return mt5.ORDER_FILLING_RETURN

def connect():
    global FILLING_MODE, XAUUSD_POINT
    if not mt5.initialize():
        log("ERROR", f"MT5 Init Failed: {mt5.last_error()}")
        quit()
    if not mt5.symbol_select(SYMBOL, True):
        log("ERROR", f"Symbol '{SYMBOL}' not found.")
        quit()

    FILLING_MODE = check_filling_mode(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    if info and info.point > 0:
        XAUUSD_POINT = float(info.point)
        log("INFO", f"Broker point derived: {XAUUSD_POINT}")
    else:
        log("WARN", f"Could not derive broker point. Using fallback {XAUUSD_POINT}")

    acc = mt5.account_info()
    log(
        "INFO",
        f"Connected | Symbol: {SYMBOL} | Filling: {FILLING_MODE} | "
        f"Balance: {acc.balance:.2f} {acc.currency}"
    )

def get_spread_points() -> float:
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        return 999.0
    return (tick.ask - tick.bid) / XAUUSD_POINT

def get_market_data(symbol, timeframe, n=500):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) < 120:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    df.drop(columns=['real_volume'], inplace=True, errors='ignore')

    df['ema_9']   = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21']  = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50']  = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['sma_200'] = df['close'].rolling(200).mean()

    df['rsi'] = ta.rsi(df['close'], length=14)

    try:
        stoch = ta.stochrsi(df['close'], length=14, rsi_length=14, k=3, d=3)
        if stoch is not None and not stoch.empty:
            df['stochrsi_k'] = stoch.iloc[:, 0]
            df['stochrsi_d'] = stoch.iloc[:, 1]
    except Exception:
        pass

    try:
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df['macd_hist']   = macd['MACDh_12_26_9']
            df['macd_line']   = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
    except Exception:
        pass

    df['atr']   = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_7'] = ta.atr(df['high'], df['low'], df['close'], length=7)

    try:
        bb = ta.bbands(df['close'], length=20, std=2.0)
        if bb is not None and not bb.empty:
            cols = bb.columns.tolist()
            cu = next((c for c in cols if c.startswith('BBU_')), None)
            cl = next((c for c in cols if c.startswith('BBL_')), None)
            cm = next((c for c in cols if c.startswith('BBM_')), None)
            if cu and cl and cm:
                df['bb_upper'] = bb[cu]
                df['bb_lower'] = bb[cl]
                df['bb_mid']   = bb[cm]
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].replace(0, np.nan)
                df['bb_pct']   = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    except Exception:
        pass

    try:
        st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
        if st is not None and not st.empty:
            for col in st.columns:
                if col.startswith('SUPERTd_'):
                    df['supertrend_dir'] = st[col]
                elif (
                    col.startswith('SUPERT_')
                    and not col.startswith('SUPERTd_')
                    and not col.startswith('SUPERTl_')
                    and not col.startswith('SUPERTs_')
                ):
                    df['supertrend_val'] = st[col]
    except Exception:
        pass

    df['vol_ma']     = df['volume'].rolling(20).mean()
    df['vol_ratio']  = df['volume'] / df['vol_ma'].replace(0, np.nan)
    df['candle_dir'] = np.where(df['close'] >= df['open'], 1, -1)
    df['candle_body']  = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['body_ratio']   = df['candle_body'] / df['candle_range'].replace(0, np.nan)
    df['delta']        = df['volume'] * df['candle_dir']
    df['cum_delta']    = df['delta'].rolling(20).sum()

    if 'spread' in df.columns:
        df['spread_pts'] = df['spread']
    else:
        df['spread_pts'] = 0

    core_cols = ['ema_9', 'ema_21', 'ema_50', 'rsi', 'atr', 'atr_7']
    core_cols = [c for c in core_cols if c in df.columns]
    if core_cols:
        df.dropna(subset=core_cols, inplace=True)

    df.reset_index(drop=True, inplace=True)
    if len(df) < 90:
        return None
    return df

def find_fractals(highs, lows, wing=2):
    swing_highs = []
    swing_lows = []
    for i in range(wing, len(highs) - wing):
        if highs[i] == max(highs[i - wing: i + wing + 1]):
            swing_highs.append((i, float(highs[i])))
        if lows[i] == min(lows[i - wing: i + wing + 1]):
            swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows

#============================================================================
# ADAPTIVE SPREAD MODEL
#============================================================================

class AdaptiveSpreadModel:
    def __init__(self):
        self.spreads = deque(maxlen=3500)
        self.session_spreads = {}
        self.current = 0.0
        self.rank = 50.0
        self.state = "NORMAL"
        self.median = 0.0
        self.p75 = 0.0
        self.p95 = 0.0
        self.p99 = 0.0
        self.hard_block_pts = 999.0
        self._min_samples = 30

    def update(self, spread_pts: float, session_name: str = "UNKNOWN"):
        self.current = float(spread_pts)
        self.spreads.append(self.current)
        dq = self.session_spreads.setdefault(session_name or "UNKNOWN", deque(maxlen=2200))
        dq.append(self.current)
        self._calibrate()

    def _calibrate(self):
        if len(self.spreads) < self._min_samples:
            self.rank = 50.0
            self.state = "NORMAL"
            self.hard_block_pts = max(10.0, self.current * 5.0)
            return

        arr = np.asarray(self.spreads, dtype=float)
        self.median = percentile(arr, 50)
        self.p75 = percentile(arr, 75)
        self.p95 = percentile(arr, 95)
        self.p99 = percentile(arr, 99)
        self.rank = float(np.mean(arr <= self.current) * 100.0)

        if self.rank >= 99.5:
            self.state = "EXTREME"
        elif self.rank >= 97.5:
            self.state = "HIGH"
        elif self.rank >= 90.0:
            self.state = "ELEVATED"
        else:
            self.state = "NORMAL"

        self.hard_block_pts = max(percentile(arr, 99.85), self.p99 * 1.05, 10.0)

    def session_percentile(self, session_name: str, pct: float):
        dq = self.session_spreads.get(session_name)
        if dq and len(dq) >= self._min_samples:
            return percentile(dq, pct)
        if len(self.spreads) >= self._min_samples:
            return percentile(self.spreads, pct)
        return max(self.current, 1.0)

#============================================================================
# MICROSTRUCTURE ENGINE
#============================================================================

class MicrostructureEngine:
    def __init__(self):
        self.ticks = deque(maxlen=700)
        self.velocity = 0.0
        self.accel = 0.0
        self.pressure = 0.5
        self.streak = 0
        self.abs_velocity = 0.0

    def update(self, tick, atr: float):
        if not tick:
            return
        now = time.time()
        mid = (tick.ask + tick.bid) / 2.0
        self.ticks.append((now, float(mid)))

        recent = [(ts, px) for ts, px in self.ticks if now - ts <= 2.2]
        if len(recent) < 6:
            return

        ts_arr = [x[0] for x in recent]
        px_arr = np.asarray([x[1] for x in recent], dtype=float)
        diffs = np.diff(px_arr)
        dt = max(float(ts_arr[-1] - ts_arr[0]), 1e-4)

        atr = max(safe_float(atr, 1.0), XAUUSD_POINT)
        net = float(px_arr[-1] - px_arr[0])

        self.velocity = clamp(net / (atr * dt), -4.0, 4.0)
        self.abs_velocity = abs(self.velocity)

        half = max(1, len(diffs) // 2)
        self.accel = clamp((float(np.mean(diffs[half:])) - float(np.mean(diffs[:half]))) / atr, -3.0, 3.0)
        self.pressure = clamp(float(np.mean(diffs > 0.0)), 0.0, 1.0)

        streak = 0
        if len(diffs) > 0:
            direction = 1 if diffs[-1] > 0 else -1
            for d in reversed(diffs):
                if (d > 0 and direction == 1) or (d < 0 and direction == -1):
                    streak += 1
                else:
                    break
            streak *= direction
        self.streak = int(clamp(streak, -12, 12))

    def entry_bias(self) -> float:
        bias = (
            self.velocity * 0.45
            + (self.pressure - 0.5) * 0.90
            + self.accel * 0.35
            + self.streak * 0.012
        )
        return clamp(bias, -1.0, 1.0)

#============================================================================
# NEWS / ABNORMAL MARKET STATE
#============================================================================

class NewsAdaptiveFilter:
    def __init__(self):
        self.state = "NORMAL"
        self.last_shock = None

    def update(self, spread_model: AdaptiveSpreadModel, state: dict, df, micro: MicrostructureEngine) -> str:
        abnormal = 0.0

        if spread_model.rank >= 99.5:
            abnormal += 0.45
        elif spread_model.rank >= 98.0:
            abnormal += 0.25

        if state.get("vol_rank", 0.5) >= 0.96:
            abnormal += 0.20

        try:
            last_range = safe_float(df['high'].iloc[-1] - df['low'].iloc[-1])
            atr = max(safe_float(state.get("atr", 1.0)), XAUUSD_POINT)
            vol_ratio = safe_float(df['vol_ratio'].iloc[-1], 1.0)
            if last_range > atr * 3.0 and vol_ratio > 1.8:
                abnormal += 0.35
        except Exception:
            pass

        if micro.abs_velocity > 1.7:
            abnormal += 0.15

        now = datetime.now()

        if abnormal >= 0.60:
            self.last_shock = now
            self.state = "SHOCK"
        elif abnormal >= 0.35:
            self.state = "PRE_SHOCK"
        elif self.last_shock is not None:
            delta = (now - self.last_shock).total_seconds()
            if delta <= 90:
                self.state = "SHOCK"
            elif delta <= 420:
                self.state = "POST_SHOCK"
            elif delta <= 900:
                self.state = "RECOVERY"
            else:
                self.state = "NORMAL"
        else:
            self.state = "NORMAL"

        return self.state

#============================================================================
# LEARNING MEMORY
#============================================================================

class LearningMemory:
    def __init__(self, path: str):
        self.path = path
        self.records = []
        self.logistic_w = np.zeros(LOGISTIC_DIM)
        self._save_counter = 0
        self.load()

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.records = data.get("records", [])[-MAX_RECORDS:]
                lw = data.get("logistic_w", None)
                if lw is not None and len(lw) == LOGISTIC_DIM:
                    self.logistic_w = np.asarray(lw, dtype=float)
        except Exception as e:
            log("WARN", f"Learning load failed: {e}")
            self.records = []
            self.logistic_w = np.zeros(LOGISTIC_DIM)

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(
                    {
                        "records": self.records[-MAX_RECORDS:],
                        "logistic_w": self.logistic_w.tolist(),
                    },
                    f,
                    indent=2,
                    default=json_default
                )
        except Exception as e:
            log("WARN", f"Learning save failed: {e}")

    def add_record(self, rec: dict):
        rec["ts"] = int(time.time())
        self.records.append(rec)
        if len(self.records) > MAX_RECORDS:
            self.records = self.records[-MAX_RECORDS:]

        try:
            self.update_logistic(rec)
        except Exception as e:
            log("WARN", f"Logistic update failed: {e}")

        self._save_counter += 1
        if self._save_counter % 3 == 0:
            self.save()

    def _filters_from_context(self, context: dict):
        if not context:
            return {}
        out = {}
        for k in ["session", "regime", "vol_bucket", "direction", "entry_type", "spread_bucket"]:
            v = context.get(k)
            if v not in (None, "", "UNKNOWN"):
                out[k] = v
        return out

    def _context_candidates(self, context: dict):
        base = self._filters_from_context(context)
        candidates = []

        def add(f):
            key = tuple(sorted(f.items()))
            if key not in [tuple(sorted(x.items())) for x in candidates]:
                candidates.append(f)

        add(base)
        for drop in ["spread_bucket", "entry_type", "direction", "vol_bucket", "regime", "session"]:
            add({k: v for k, v in base.items() if k != drop})

        if base.get("regime"):
            add({"regime": base["regime"]})
        if base.get("session"):
            add({"session": base["session"]})
        add({})
        return candidates

    def query(self, session=None, regime=None, vol_bucket=None, direction=None,
              entry_type=None, spread_bucket=None):
        recs = self.records[-STAT_WINDOW:]
        out = []
        for r in recs:
            if session is not None and r.get("session") != session:
                continue
            if regime is not None and r.get("regime") != regime:
                continue
            if vol_bucket is not None and r.get("vol_bucket") != vol_bucket:
                continue
            if direction is not None and r.get("direction") != direction:
                continue
            if entry_type is not None and r.get("entry_type") != entry_type:
                continue
            if spread_bucket is not None and r.get("spread_bucket") != spread_bucket:
                continue
            out.append(r)
        return out

    def weighted_stats(self, recs: list):
        base = {
            "n": len(recs),
            "effective_n": 0.0,
            "win_rate": 0.5,
            "expectancy_r": None,
            "avg_mae_r": None,
            "avg_mfe_r": None,
            "avg_pnl": 0.0,
            "last_loss_streak": 0,
        }
        if not recs:
            return base

        sum_w = 0.0
        sum_w2 = 0.0
        win_w = 0.0
        pnl_w = 0.0
        r_w = 0.0
        r_sw = 0.0
        mae_w = 0.0
        mae_sw = 0.0
        mfe_w = 0.0
        mfe_sw = 0.0

        for r in recs:
            w = recency_weight(r.get("ts", time.time()))
            if w <= 1e-6:
                continue

            pnl = safe_float(r.get("pnl", 0.0))
            sum_w += w
            sum_w2 += w * w
            pnl_w += w * pnl
            if pnl > 0:
                win_w += w

            rv = r.get("r")
            if rv is not None:
                rv = safe_float(rv)
                r_w += w * rv
                r_sw += w

            mae = r.get("mae_r")
            if mae is not None:
                mae = safe_float(mae)
                mae_w += w * mae
                mae_sw += w

            mfe = r.get("mfe_r")
            if mfe is not None:
                mfe = safe_float(mfe)
                mfe_w += w * mfe
                mfe_sw += w

        if sum_w <= 1e-6:
            return base

        last_loss_streak = 0
        for r in reversed(recs):
            if safe_float(r.get("pnl", 0.0)) <= 0:
                last_loss_streak += 1
            else:
                break

        return {
            "n": len(recs),
            "effective_n": (sum_w * sum_w) / sum_w2 if sum_w2 > 0 else 0.0,
            "win_rate": win_w / sum_w,
            "expectancy_r": r_w / r_sw if r_sw > 0 else None,
            "avg_mae_r": mae_w / mae_sw if mae_sw > 0 else None,
            "avg_mfe_r": mfe_w / mfe_sw if mfe_sw > 0 else None,
            "avg_pnl": pnl_w / sum_w,
            "last_loss_streak": last_loss_streak,
        }

    def context_stats(self, context: dict):
        candidates = self._context_candidates(context)
        for f in candidates:
            min_n = max(6, 18 - len(f) * 2)
            recs = self.query(**f)
            stats = self.weighted_stats(recs)
            if stats["n"] >= min_n and stats["effective_n"] >= max(5.0, min_n * 0.45):
                stats["level"] = len(f)
                return stats

        stats = self.weighted_stats(self.records[-100:])
        stats["level"] = -1
        return stats

    def get_component_stats(self, component: str, context: dict):
        candidates = self._context_candidates(context)
        for f in candidates:
            min_n = max(5, 16 - len(f) * 2)
            recs = self.query(**f)
            active = []
            for r in recs:
                contribs = r.get("contributions", {})
                if not isinstance(contribs, dict):
                    continue
                if safe_float(contribs.get(component, 0.0)) > 0.005:
                    active.append(r)
            stats = self.weighted_stats(active)
            if stats["n"] >= min_n and stats["effective_n"] >= max(4.0, min_n * 0.45):
                return stats

        return {
            "n": 0,
            "effective_n": 0.0,
            "win_rate": 0.5,
            "expectancy_r": None,
            "avg_mae_r": None,
            "avg_mfe_r": None,
            "avg_pnl": 0.0,
            "last_loss_streak": 0,
        }

    def get_component_reliability(self, component: str, context: dict):
        stats = self.get_component_stats(component, context)
        if stats.get("n", 0) < 6 or stats.get("effective_n", 0.0) < 4.0:
            return 0.5

        wr = safe_float(stats.get("win_rate", 0.5), 0.5)
        exp = safe_float(stats.get("expectancy_r", 0.0), 0.0)
        conf = clamp(stats.get("effective_n", 0.0) / 30.0, 0.0, 1.0)

        rel = 0.5 + (wr - 0.5) * 0.80 * conf + exp * 0.10 * conf
        return clamp(rel, 0.15, 0.85)

    def _weighted_percentile(self, values, weights, pct):
        if not values:
            return None
        pairs = sorted(zip(values, weights), key=lambda x: x[0])
        total = sum(weights)
        if total <= 0:
            return None
        target = total * clamp(pct, 0.0, 100.0) / 100.0
        cum = 0.0
        for v, w in pairs:
            cum += w
            if cum >= target:
                return v
        return pairs[-1][0]

    def get_base_threshold(self, context: dict):
        candidates = self._context_candidates(context)
        for f in candidates:
            recs = [r for r in self.query(**f) if safe_float(r.get("pnl", 0.0)) > 0]
            vals = []
            weights = []
            for r in recs:
                v = safe_float(r.get("entry_score", r.get("raw_score", 0.55)), 0.55)
                if v > 0:
                    vals.append(v)
                    weights.append(recency_weight(r.get("ts", time.time())))
            if len(vals) >= 12:
                p = self._weighted_percentile(vals, weights, 38.0)
                if p is not None:
                    return clamp(p, 0.50, 0.68)
        return 0.56

    def get_score_scale(self, context: dict):
        candidates = self._context_candidates(context)
        for f in candidates:
            recs = [r for r in self.query(**f) if safe_float(r.get("pnl", 0.0)) > 0]
            vals = []
            weights = []
            for r in recs:
                v = safe_float(r.get("lead_score", r.get("raw_score", 0.6)), 0.6)
                if v > 0:
                    vals.append(v)
                    weights.append(recency_weight(r.get("ts", time.time())))
            if len(vals) >= 10:
                p = self._weighted_percentile(vals, weights, 75.0)
                if p is not None:
                    return clamp(p, 0.35, 1.75)
        return 0.75

    def bin_calibration(self, raw: float, context: dict):
        raw = clamp(raw, 0.0, 1.0)
        candidates = self._context_candidates(context)
        chosen = []
        for f in candidates:
            recs = self.query(**f)
            if len(recs) >= 14:
                chosen = recs
                break

        if not chosen:
            return raw

        wins_w = 0.0
        total_w = 0.0
        width = 0.07

        for r in chosen:
            rc = safe_float(r.get("raw_score", r.get("raw_conviction", 0.5)), 0.5)
            if abs(rc - raw) <= width:
                w = recency_weight(r.get("ts", time.time()))
                total_w += w
                if safe_float(r.get("pnl", 0.0)) > 0:
                    wins_w += w

        if total_w <= 0:
            return raw

        wr = (wins_w + 1.0) / (total_w + 2.0)
        conf = clamp(total_w / 25.0, 0.0, 1.0) * 0.85
        return clamp(raw * (1.0 - conf) + wr * conf, 0.0, 1.0)

    def _feature_vector(self, rec: dict):
        return np.array(
            [
                1.0,
                clamp(safe_float(rec.get("raw_score", rec.get("raw_conviction", 0.5)), 0.5), 0.0, 1.0),
                clamp(safe_float(rec.get("agreement", 0.0), 0.0), 0.0, 1.0),
                clamp(safe_float(rec.get("conflict", 0.0), 0.0), 0.0, 1.0),
                clamp(safe_float(rec.get("reliability_avg", 0.5), 0.5), 0.0, 1.0),
                clamp(safe_float(rec.get("entry_quality", 0.5), 0.5), 0.0, 1.0),
                clamp(safe_float(rec.get("vol_rank", 0.5), 0.5), 0.0, 1.0),
                clamp(safe_float(rec.get("spread_rank", 0.5), 0.5), 0.0, 1.0),
                0.0 if rec.get("news_state", "NORMAL") == "NORMAL" else 1.0,
                1.0 if rec.get("direction") == "BUY" else 0.0,
            ],
            dtype=float
        )

    def update_logistic(self, rec: dict):
        x = self._feature_vector(rec)
        z = float(np.dot(self.logistic_w, x))
        p = sigmoid(z)
        y = 1.0 if safe_float(rec.get("pnl", 0.0)) > 0 else 0.0
        lr = 0.012
        grad = (p - y) * x + 0.0002 * self.logistic_w
        self.logistic_w -= lr * grad
        self.logistic_w = np.clip(self.logistic_w, -3.0, 3.0)

    def logistic_predict(self, features: np.ndarray):
        try:
            return sigmoid(float(np.dot(self.logistic_w, features)))
        except Exception:
            return 0.5

    def trades_per_hour(self, hours=6.0):
        cutoff = time.time() - hours * 3600.0
        n = sum(1 for r in self.records if safe_float(r.get("ts", 0.0)) >= cutoff)
        return n / max(hours, 0.1)

#============================================================================
# ADAPTIVE WEIGHT ENGINE
#============================================================================

class AdaptiveWeightEngine:
    def __init__(self):
        self.smooth = {}

    def get_weights(self, memory: LearningMemory, context: dict):
        key = "|".join(
            [
                str(context.get("session", "")),
                str(context.get("regime", "")),
                str(context.get("vol_bucket", "")),
                str(context.get("direction", "")),
                str(context.get("entry_type", "")),
            ]
        )

        weights = {}
        reliabilities = {}

        for c in COMPONENTS:
            stats = memory.get_component_stats(c, context)
            rel = memory.get_component_reliability(c, context)
            reliabilities[c] = rel

            exp = safe_float(stats.get("expectancy_r", 0.0), 0.0)
            wr = safe_float(stats.get("win_rate", 0.5), 0.5)
            edge = (wr - 0.5) * 0.90 + exp * 0.12

            target = 1.0 + (rel - 0.5) * 1.40 + edge * 0.35
            target = clamp(target, WEIGHT_MIN, WEIGHT_MAX)

            sk = f"{key}:{c}"
            prev = self.smooth.get(sk, target)
            w = prev * 0.85 + target * 0.15
            self.smooth[sk] = w
            weights[c] = clamp(w, WEIGHT_MIN, WEIGHT_MAX)

        rel_avg = float(np.mean([reliabilities[c] for c in COMPONENTS])) if COMPONENTS else 0.5
        return weights, reliabilities, rel_avg

#============================================================================
# ADAPTIVE THRESHOLD ENGINE
#============================================================================

class AdaptiveThresholdEngine:
    def get(self, memory: LearningMemory, context: dict, state: dict,
            news_state: str, spread_model: AdaptiveSpreadModel):
        base = memory.get_base_threshold(context)
        stats = memory.context_stats(context)
        adj = 0.0

        if stats.get("n", 0) >= 12 and stats.get("effective_n", 0.0) >= 7:
            exp = stats.get("expectancy_r")
            if exp is not None:
                exp = safe_float(exp, 0.0)
                if exp < -0.05:
                    adj += min(0.05, abs(exp) * 0.18)
                elif exp > 0.12 and stats.get("win_rate", 0.5) > 0.55:
                    adj -= min(0.03, exp * 0.08)

            if stats.get("last_loss_streak", 0) >= 3:
                adj += 0.012

        if news_state == "SHOCK":
            adj += 0.035
        elif news_state in ("PRE_SHOCK", "POST_SHOCK"):
            adj += 0.018
        elif news_state == "RECOVERY":
            adj += 0.008

        if state.get("vol_rank", 0.5) > 0.90:
            adj += 0.015
        elif state.get("vol_rank", 0.5) < 0.10:
            adj -= 0.006

        if spread_model.state in ("HIGH", "EXTREME"):
            adj += 0.010

        freq = memory.trades_per_hour()
        if freq < 1.0:
            adj -= 0.015
        elif freq > 6.0:
            adj += 0.010

        return clamp(base + adj, THRESH_MIN, THRESH_MAX)

#============================================================================
# ADAPTIVE PROBABILITY MODEL
#============================================================================

class AdaptiveProbabilityModel:
    def calibrate(self, memory: LearningMemory, raw: float, context: dict, features: np.ndarray):
        raw = clamp(raw, 0.0, 1.0)
        bin_cal = memory.bin_calibration(raw, context)
        logit = memory.logistic_predict(features)

        stats = memory.context_stats(context)
        conf = clamp(stats.get("effective_n", 0.0) / 45.0, 0.0, 1.0)

        if conf <= 0:
            return raw

        blended = 0.55 * bin_cal + 0.45 * logit
        out = raw * (1.0 - conf) + blended * conf
        return clamp(out, 0.0, 1.0)

#============================================================================
# MARKET STATE ENGINE
#============================================================================

def default_market_state():
    return {
        "atr": 1.0,
        "atr_rank": 0.5,
        "bbw_rank": 0.5,
        "vol_rank": 0.5,
        "vol_state": "NORMAL",
        "adx": 20.0,
        "chop": 50.0,
        "adx_rank": 0.5,
        "chop_rank": 0.5,
        "trend_score": 0.5,
        "range_score": 0.5,
        "regime": "NORMAL",
        "market_speed": 0.5,
        "breakout_prob": 0.5,
        "reversal_prob": 0.5,
        "spread_rank": 0.5,
        "momentum_state": 0.0,
    }

class MarketStateEngine:
    @staticmethod
    def compute(df, spread_model: AdaptiveSpreadModel, micro: MicrostructureEngine):
        try:
            if df is None or len(df) < 120:
                return default_market_state()

            atr_arr = df['atr'].dropna().values
            atr7_arr = df['atr_7'].dropna().values if 'atr_7' in df.columns else atr_arr
            bbw_arr = df['bb_width'].dropna().values if 'bb_width' in df.columns else np.array([])

            atr_now = float(atr_arr[-1]) if len(atr_arr) else 1.0
            atr_rank = pct_rank(atr_arr[-320:], atr_now)
            atr7_rank = pct_rank(atr7_arr[-320:], float(atr7_arr[-1]) if len(atr7_arr) else atr_now)
            bbw_rank = pct_rank(bbw_arr[-320:], float(bbw_arr[-1])) if len(bbw_arr) else 0.5

            vol_score = clamp(0.55 * atr_rank + 0.25 * bbw_rank + 0.20 * atr7_rank, 0.0, 1.0)

            if vol_score <= 0.12:
                vol_state = "EXTREME_LOW"
            elif vol_score <= 0.32:
                vol_state = "LOW"
            elif vol_score <= 0.70:
                vol_state = "NORMAL"
            elif vol_score <= 0.90:
                vol_state = "HIGH"
            else:
                vol_state = "EXTREME_HIGH"

            try:
                adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
                adx_val = safe_float(adx_df['ADX_14'].iloc[-1], 20.0) if adx_df is not None and not adx_df.empty else 20.0
                adx_series = adx_df['ADX_14'].dropna().values[-260:] if adx_df is not None else np.array([20.0])
            except Exception:
                adx_val = 20.0
                adx_series = np.array([20.0])

            try:
                tr = ta.true_range(df['high'], df['low'], df['close'])
                chop_raw = (
                    100 * np.log10(
                        tr.rolling(14).sum() /
                        (df['high'].rolling(14).max() - df['low'].rolling(14).min()).replace(0, np.nan)
                    ) / np.log10(14)
                )
                chop_val = safe_float(chop_raw.iloc[-1], 50.0)
                chop_series = chop_raw.dropna().values[-260:]
            except Exception:
                chop_val = 50.0
                chop_series = np.array([50.0])

            adx_rank = pct_rank(adx_series, adx_val)
            chop_rank = pct_rank(chop_series, chop_val)

            ema_sep_series = (
                abs(df['ema_9'] - df['ema_50']) / df['atr'].replace(0, np.nan)
            ).dropna().values[-260:]
            ema_sep_now = float(ema_sep_series[-1]) if len(ema_sep_series) else 0.0
            ema_sep_rank = pct_rank(ema_sep_series, ema_sep_now)

            trend_score = clamp(0.55 * adx_rank + 0.25 * ema_sep_rank + 0.20 * (1.0 - chop_rank), 0.0, 1.0)
            range_score = clamp(0.55 * chop_rank + 0.25 * (1.0 - adx_rank) + 0.20 * (1.0 - min(ema_sep_rank, 1.0)), 0.0, 1.0)

            if trend_score > 0.64 and range_score < 0.46:
                regime = "TRENDING"
            elif range_score > 0.64 or (range_score > 0.55 and trend_score < 0.45):
                regime = "RANGING"
            else:
                regime = "NORMAL"

            range_ratio = (df['candle_range'] / df['atr'].replace(0, np.nan)).dropna().values[-220:]
            speed_candle = pct_rank(range_ratio, float(range_ratio[-1]) if len(range_ratio) else 1.0)
            speed_tick = clamp(abs(micro.velocity) * 0.45, 0.0, 1.0) if micro else 0.5
            market_speed = clamp(0.65 * speed_candle + 0.35 * speed_tick, 0.0, 1.0)

            highs = df['high'].values
            lows = df['low'].values
            close = float(df['close'].values[-1])

            hh20 = float(np.max(highs[-21:-1])) if len(highs) > 21 else close
            ll20 = float(np.min(lows[-21:-1])) if len(lows) > 21 else close

            dist_high = (close - hh20) / max(atr_now, XAUUSD_POINT)
            dist_low = (ll20 - close) / max(atr_now, XAUUSD_POINT)
            squeeze = 1.0 - bbw_rank
            breakout_prob = clamp(
                max(0.0, dist_high) * 0.38 + max(0.0, dist_low) * 0.38 + squeeze * 0.24,
                0.0,
                1.0
            )

            rsi_now = safe_float(df['rsi'].iloc[-1], 50.0)
            rsi_extreme = 0.0
            if rsi_now > 70:
                rsi_extreme = min((rsi_now - 70.0) / 25.0, 1.0)
            elif rsi_now < 30:
                rsi_extreme = min((30.0 - rsi_now) / 25.0, 1.0)

            body_ratio_arr = df['body_ratio'].dropna().values[-200:]
            body_rank = pct_rank(body_ratio_arr, float(body_ratio_arr[-1]) if len(body_ratio_arr) else 0.5)
            reversal_prob = clamp(rsi_extreme * 0.55 + (1.0 - body_rank) * 0.20 + vol_score * 0.25, 0.0, 1.0)

            momentum_state = 0.0
            if 'macd_hist' in df.columns:
                mh = df['macd_hist'].dropna().values[-220:]
                if len(mh) > 20:
                    momentum_state = clamp(pct_rank(mh, float(mh[-1])) - 0.5, -0.5, 0.5) * 2.0

            return {
                "atr": float(atr_now),
                "atr_rank": float(atr_rank),
                "bbw_rank": float(bbw_rank),
                "vol_rank": float(vol_score),
                "vol_state": vol_state,
                "adx": float(adx_val),
                "chop": float(chop_val),
                "adx_rank": float(adx_rank),
                "chop_rank": float(chop_rank),
                "trend_score": float(trend_score),
                "range_score": float(range_score),
                "regime": regime,
                "market_speed": float(market_speed),
                "breakout_prob": float(breakout_prob),
                "reversal_prob": float(reversal_prob),
                "spread_rank": float(spread_model.rank / 100.0 if spread_model else 0.5),
                "momentum_state": float(momentum_state),
            }
        except Exception:
            return default_market_state()

#============================================================================
# ADAPTIVE COMPONENT ENGINES
#============================================================================

def neutral_component(label="Neutral"):
    return {
        "direction": 0,
        "strength": 0.0,
        "label": label,
        "weight": 1.0,
        "reliability": 0.5,
    }

class AdaptiveComponentEngine:
    def __init__(self, memory: LearningMemory):
        self.memory = memory

    def structure(self, df, state: dict):
        try:
            if df is None or len(df) < 45:
                return neutral_component("Structure: insufficient")

            wing = int(clamp(2 + state.get("vol_rank", 0.5) * 2 + (1 - state.get("trend_score", 0.5)), 2, 5))
            highs = df['high'].values[-90:]
            lows = df['low'].values[-90:]
            closes = df['close'].values[-90:]
            curr = float(closes[-1])
            atr = max(safe_float(state.get("atr", 1.0)), XAUUSD_POINT)

            sh, sl = find_fractals(highs, lows, wing)
            if len(sh) < 2 or len(sl) < 2:
                return neutral_component("Structure: no fractals")

            bull = 0
            bear = 0
            for i in range(1, len(sh)):
                if sh[i][1] > sh[i - 1][1]:
                    bull += 1
                elif sh[i][1] < sh[i - 1][1]:
                    bear += 1
            for i in range(1, len(sl)):
                if sl[i][1] > sl[i - 1][1]:
                    bull += 1
                elif sl[i][1] < sl[i - 1][1]:
                    bear += 1

            total = bull + bear
            if total == 0:
                return neutral_component("Structure: flat")

            base_strength = abs(bull - bear) / float(total)

            direction = 1 if bull > bear else -1 if bear > bull else 0
            strength = clamp(base_strength * 0.62, 0.0, 1.0)

            last_sh = sh[-1][1]
            last_sl = sl[-1][1]
            prev_close = float(closes[-2])

            if curr > last_sh and prev_close <= last_sh:
                bos = (curr - last_sh) / atr
                direction = 1
                strength = clamp(strength + 0.20 + bos * 0.18, 0.0, 1.0)
            elif curr < last_sl and prev_close >= last_sl:
                bos = (last_sl - curr) / atr
                direction = -1
                strength = clamp(strength + 0.20 + bos * 0.18, 0.0, 1.0)

            return {
                "direction": direction,
                "strength": clamp(strength, 0.0, 1.0),
                "label": f"Structure:{'BULL' if direction==1 else 'BEAR' if direction==-1 else 'FLAT'} wing:{wing}",
                "weight": 1.0,
                "reliability": 0.5,
            }
        except Exception:
            return neutral_component("Structure: error")

    def alignment(self, df_m1, df_m5, df_m15, state: dict):
        try:
            def tf_dir(df):
                if df is None or len(df) < 8:
                    return 0, 0.0
                close = safe_float(df['close'].iloc[-1])
                e9 = safe_float(df['ema_9'].iloc[-1], close)
                e21 = safe_float(df['ema_21'].iloc[-1], close)
                e50 = safe_float(df['ema_50'].iloc[-1], close)
                st = safe_int(df['supertrend_dir'].iloc[-1], 0) if 'supertrend_dir' in df.columns else 0

                ema_dir = 1 if e9 > e21 else -1
                ema50_ok = (ema_dir == 1 and close > e50) or (ema_dir == -1 and close < e50)

                score = 0.35
                if ema50_ok:
                    score += 0.25
                if st == ema_dir:
                    score += 0.25
                if (ema_dir == 1 and close > e9) or (ema_dir == -1 and close < e9):
                    score += 0.15

                if st != 0 and st != ema_dir:
                    return 0, clamp(score * 0.45, 0.0, 1.0)
                return ema_dir, clamp(score, 0.0, 1.0)

            d1, s1 = tf_dir(df_m1)
            d5, s5 = tf_dir(df_m5)
            d15, s15 = tf_dir(df_m15)

            m1_w = clamp(0.35 + 0.25 * state.get("market_speed", 0.5) + 0.10 * state.get("vol_rank", 0.5), 0.20, 0.60)
            m15_w = clamp(0.25 + 0.25 * state.get("trend_score", 0.5) - 0.15 * state.get("market_speed", 0.5), 0.15, 0.50)
            m5_w = clamp(1.0 - m1_w - m15_w, 0.15, 0.55)

            norm = m1_w + m5_w + m15_w
            m1_w /= norm
            m5_w /= norm
            m15_w /= norm

            signed = d1 * s1 * m1_w + d5 * s5 * m5_w + d15 * s15 * m15_w
            direction = 1 if signed > 0.08 else -1 if signed < -0.08 else 0
            strength = clamp(abs(signed) * 1.45, 0.0, 1.0)

            return {
                "direction": direction,
                "strength": strength,
                "label": f"Align:{'BULL' if direction==1 else 'BEAR' if direction==-1 else 'MIX'} M1:{d1} M5:{d5} M15:{d15}",
                "weight": 1.0,
                "reliability": 0.5,
                "m1": d1,
                "m5": d5,
                "m15": d15,
            }
        except Exception:
            return neutral_component("Alignment: error")

    def momentum(self, df, state: dict):
        try:
            if df is None or 'rsi' not in df.columns or len(df) < 60:
                return neutral_component("Momentum: insufficient")

            rsi_arr = df['rsi'].dropna().values[-240:]
            rsi_now = float(rsi_arr[-1]) if len(rsi_arr) else 50.0
            trend = state.get("trend_score", 0.5)

            buy_min = clamp(percentile(rsi_arr, 55 + 15 * trend), 50.5, 68.0)
            sell_max = clamp(percentile(rsi_arr, 45 - 15 * trend), 32.0, 49.5)

            mh = df['macd_hist'].dropna().values if 'macd_hist' in df.columns else np.array([])
            macd_dir = 0
            macd_accel = 0.0
            if len(mh) >= 4:
                macd_dir = 1 if mh[-1] > 0 else -1 if mh[-1] < 0 else 0
                if mh[-1] > mh[-2] > mh[-3] and mh[-1] > 0:
                    macd_accel = 1.0
                elif mh[-1] < mh[-2] < mh[-3] and mh[-1] < 0:
                    macd_accel = -1.0

            stoch_cross = 0
            if 'stochrsi_k' in df.columns and 'stochrsi_d' in df.columns and len(df) >= 3:
                sk0 = safe_float(df['stochrsi_k'].iloc[-1], 50.0)
                sd0 = safe_float(df['stochrsi_d'].iloc[-1], 50.0)
                sk1 = safe_float(df['stochrsi_k'].iloc[-2], 50.0)
                sd1 = safe_float(df['stochrsi_d'].iloc[-2], 50.0)
                if sk1 <= sd1 and sk0 > sd0:
                    stoch_cross = 1
                elif sk1 >= sd1 and sk0 < sd0:
                    stoch_cross = -1

            buy = 0.0
            sell = 0.0

            if rsi_now > buy_min:
                buy += 0.55 + 0.35 * clamp((rsi_now - buy_min) / max(65 - buy_min, 1.0), 0.0, 1.0)
            elif rsi_now < sell_max:
                sell += 0.55 + 0.35 * clamp((sell_max - rsi_now) / max(sell_max - 35, 1.0), 0.0, 1.0)

            if macd_accel > 0:
                buy += 0.55
            elif macd_accel < 0:
                sell += 0.55

            if macd_dir > 0:
                buy += 0.20
            elif macd_dir < 0:
                sell += 0.20

            if stoch_cross > 0:
                buy += 0.35
            elif stoch_cross < 0:
                sell += 0.35

            direction = 1 if buy > sell and buy >= 0.85 else -1 if sell > buy and sell >= 0.85 else 0
            strength = clamp(max(buy, sell) / 1.70, 0.0, 1.0)

            return {
                "direction": direction,
                "strength": strength,
                "label": f"Momentum:{'BULL' if direction==1 else 'BEAR' if direction==-1 else 'NEUT'} RSI:{rsi_now:.1f}",
                "weight": 1.0,
                "reliability": 0.5,
                "accel": macd_accel,
                "rsi": rsi_now,
            }
        except Exception:
            return neutral_component("Momentum: error")

    def order_flow(self, df, state: dict):
        try:
            if df is None or len(df) < 25:
                return neutral_component("OrderFlow: insufficient")

            recent = df.iloc[-14:]
            vol_arr = df['volume'].dropna().values[-140:]
            vol_threshold = percentile(vol_arr, 35) if len(vol_arr) else 0.0

            sig = recent[recent['volume'] > vol_threshold]
            if len(sig) == 0:
                sig = recent

            bull_vol = float(sig.loc[sig['close'] > sig['open'], 'volume'].sum())
            bear_vol = float(sig.loc[sig['close'] < sig['open'], 'volume'].sum())
            total = bull_vol + bear_vol
            if total <= 0:
                return neutral_component("OrderFlow: no volume")

            bp = bull_vol / total
            sp = bear_vol / total

            vr = safe_float(df['vol_ratio'].iloc[-1], 1.0)
            vr_arr = df['vol_ratio'].dropna().values[-220:]
            vr_req = clamp(percentile(vr_arr, 60), 1.05, 1.90) if len(vr_arr) else 1.20

            cum_delta_dir = 0
            if 'cum_delta' in df.columns and len(df) >= 6:
                cd = safe_float(df['cum_delta'].iloc[-1])
                cd_prev = safe_float(df['cum_delta'].iloc[-5])
                if cd > cd_prev * 1.04 and cd > 0:
                    cum_delta_dir = 1
                elif cd < cd_prev * 0.96 and cd < 0:
                    cum_delta_dir = -1

            adaptive_th = clamp(0.58 + 0.04 * state.get("vol_rank", 0.5), 0.55, 0.66)

            direction = 0
            strength = 0.0

            if bp >= adaptive_th:
                direction = 1
                strength = clamp((bp - 0.5) * 2.3 + (0.08 if cum_delta_dir == 1 else 0.0), 0.0, 1.0)
            elif sp >= adaptive_th:
                direction = -1
                strength = clamp((sp - 0.5) * 2.3 + (0.08 if cum_delta_dir == -1 else 0.0), 0.0, 1.0)
            elif cum_delta_dir != 0:
                direction = cum_delta_dir
                strength = 0.38

            if vr < vr_req * 0.80:
                strength *= 0.75

            spread_penalty = 1.0 - clamp(state.get("spread_rank", 0.5) * 0.25, 0.0, 0.25)
            strength *= spread_penalty

            return {
                "direction": direction,
                "strength": clamp(strength, 0.0, 1.0),
                "label": f"Flow:{'BULL' if direction==1 else 'BEAR' if direction==-1 else 'BAL'} B:{bp:.0%} VR:{vr:.1f}",
                "weight": 1.0,
                "reliability": 0.5,
            }
        except Exception:
            return neutral_component("OrderFlow: error")

    def volatility(self, df, state: dict):
        try:
            if df is None or len(df) < 30:
                return neutral_component("Volatility: insufficient")

            expanding = state.get("vol_rank", 0.5) > 0.65 or state.get("bbw_rank", 0.5) > 0.80
            close = safe_float(df['close'].iloc[-1])
            bb_mid = safe_float(df['bb_mid'].iloc[-1], close) if 'bb_mid' in df.columns else close

            direction = 0
            if expanding:
                if close > bb_mid:
                    direction = 1
                elif close < bb_mid:
                    direction = -1

            strength = clamp(0.30 + state.get("vol_rank", 0.5) * 0.55, 0.0, 1.0) if expanding else 0.18

            return {
                "direction": direction,
                "strength": strength,
                "label": f"Vol:{state.get('vol_state','NORMAL')}",
                "weight": 1.0,
                "reliability": 0.5,
            }
        except Exception:
            return neutral_component("Volatility: error")

    def smc(self, df, state: dict):
        try:
            if df is None or len(df) < 30:
                return neutral_component("SMC: insufficient")

            closes = df['close'].values
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            curr = float(closes[-1])
            atr = max(safe_float(state.get("atr", 1.0)), XAUUSD_POINT)

            buy_score = 0.0
            sell_score = 0.0
            labels = []

            # Liquidity sweep
            ref_high = float(np.max(highs[-18:-3]))
            ref_low = float(np.min(lows[-18:-3]))
            last_high = float(highs[-1])
            last_low = float(lows[-1])
            last_close = float(closes[-1])

            if last_low < ref_low and last_close > ref_low:
                depth = (ref_low - last_low) / atr
                if depth >= 0.22:
                    buy_score += clamp(0.20 + depth * 0.10, 0.0, 0.35)
                    labels.append("BullSweep")

            if last_high > ref_high and last_close < ref_high:
                depth = (last_high - ref_high) / atr
                if depth >= 0.22:
                    sell_score += clamp(0.20 + depth * 0.10, 0.0, 0.35)
                    labels.append("BearSweep")

            # Market structure break
            sh, sl = find_fractals(highs[-50:], lows[-50:], wing=3)
            if sh and sl:
                last_sh = sh[-1][1]
                last_sl = sl[-1][1]
                prev_close = float(closes[-2])

                if curr > last_sh and prev_close <= last_sh:
                    buy_score += clamp(0.18 + ((curr - last_sh) / atr) * 0.14, 0.0, 0.36)
                    labels.append("BullMSB")
                if curr < last_sl and prev_close >= last_sl:
                    sell_score += clamp(0.18 + ((last_sl - curr) / atr) * 0.14, 0.0, 0.36)
                    labels.append("BearMSB")

            # FVG
            for i in range(len(df) - 4, max(len(df) - 22, 2), -1):
                h1 = float(highs[i - 1])
                l1 = float(lows[i - 1])
                h3 = float(highs[i + 1])
                l3 = float(lows[i + 1])

                if l3 > h1 and (l3 - h1) >= atr * 0.14:
                    if h1 - atr * 0.25 <= curr <= l3 + atr * 0.25:
                        buy_score += 0.16 if h1 <= curr <= l3 else 0.10
                        labels.append("BullFVG")
                        break

                if h3 < l1 and (l1 - h3) >= atr * 0.14:
                    if h3 - atr * 0.25 <= curr <= l1 + atr * 0.25:
                        sell_score += 0.16 if h3 <= curr <= l1 else 0.10
                        labels.append("BearFVG")
                        break

            # Order block
            lookback = min(30, len(df) - 3)
            for i in range(len(df) - 3, len(df) - lookback, -1):
                if opens[i] > closes[i] and (opens[i] - closes[i]) >= atr * 0.18:
                    next_body = abs(closes[i + 1] - opens[i + 1])
                    if closes[i + 1] > opens[i + 1] and next_body > atr * 0.30:
                        ob_hi, ob_lo = float(highs[i]), float(lows[i])
                        if -atr * 0.15 <= (ob_hi - curr) <= atr * 1.7:
                            buy_score += 0.18 if ob_lo <= curr <= ob_hi else 0.11
                            labels.append("BullOB")
                            break

            for i in range(len(df) - 3, len(df) - lookback, -1):
                if closes[i] > opens[i] and (closes[i] - opens[i]) >= atr * 0.18:
                    next_body = abs(closes[i + 1] - opens[i + 1])
                    if opens[i + 1] > closes[i + 1] and next_body > atr * 0.30:
                        ob_hi, ob_lo = float(highs[i]), float(lows[i])
                        if -atr * 0.15 <= (curr - ob_lo) <= atr * 1.7:
                            sell_score += 0.18 if ob_lo <= curr <= ob_hi else 0.11
                            labels.append("BearOB")
                            break

            # Regime-aware scaling
            trend = state.get("trend_score", 0.5)
            rang = state.get("range_score", 0.5)

            buy_score *= (1.0 + 0.25 * trend)
            sell_score *= (1.0 + 0.25 * trend)

            if rang > 0.60:
                buy_score *= 1.08
                sell_score *= 1.08

            direction = 1 if buy_score > sell_score and buy_score > 0.04 else -1 if sell_score > buy_score and sell_score > 0.04 else 0
            strength = clamp(max(buy_score, sell_score), 0.0, 1.0)

            return {
                "direction": direction,
                "strength": strength,
                "label": "SMC:" + "+".join(labels[:3]) if labels else "SMC:None",
                "weight": 1.0,
                "reliability": 0.5,
            }
        except Exception:
            return neutral_component("SMC: error")

    def divergence(self, df, state: dict):
        try:
            if df is None or 'rsi' not in df.columns or len(df) < 55:
                return neutral_component("Divergence: insufficient")

            prices = np.nan_to_num(df['close'].values[-65:], nan=0.0)
            rsi = np.nan_to_num(df['rsi'].values[-65:], nan=50.0)

            ph, _ = find_fractals(prices, np.zeros_like(prices), wing=4)
            _, pl = find_fractals(np.zeros_like(prices), prices, wing=4)
            rh, _ = find_fractals(rsi, np.zeros_like(rsi), wing=4)
            _, rl = find_fractals(np.zeros_like(rsi), rsi, wing=4)

            if len(ph) < 2 or len(pl) < 2 or len(rh) < 2 or len(rl) < 2:
                return neutral_component("Divergence: none")

            ph1, ph2 = ph[-2][1], ph[-1][1]
            pl1, pl2 = pl[-2][1], pl[-1][1]
            rh1, rh2 = rh[-2][1], rh[-1][1]
            rl1, rl2 = rl[-2][1], rl[-1][1]
            curr_rsi = float(rsi[-1])

            direction = 0
            strength = 0.0
            label = "Divergence: none"

            if ph2 > ph1 * 1.001 and rh2 < rh1 * 0.999 and curr_rsi > 55:
                direction = -1
                strength = clamp(0.45 + (ph2 - ph1) / max(ph1, 1.0) * 80.0, 0.0, 0.90)
                label = "RegBearDiv"

            if pl2 < pl1 * 0.999 and rl2 > rl1 * 1.001 and curr_rsi < 45:
                direction = 1
                strength = clamp(0.45 + (pl1 - pl2) / max(pl1, 1.0) * 80.0, 0.0, 0.90)
                label = "RegBullDiv"

            # Regime adjustment
            if state.get("regime") == "TRENDING":
                strength *= 0.88
            elif state.get("regime") == "RANGING":
                strength *= 1.10

            return {
                "direction": direction,
                "strength": clamp(strength, 0.0, 1.0),
                "label": label,
                "weight": 1.0,
                "reliability": 0.5,
            }
        except Exception:
            return neutral_component("Divergence: error")

    def impulse(self, df, state: dict):
        try:
            if df is None or len(df) < 35:
                return neutral_component("Impulse: insufficient")

            closes = df['close'].values
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            vol = df['volume'].values
            atr = max(safe_float(state.get("atr", 1.0)), XAUUSD_POINT)

            body_ratios = (df['candle_body'] / df['atr'].replace(0, np.nan)).dropna().values[-190:]
            body_req = clamp(percentile(body_ratios, 82 - 10 * state.get("range_score", 0.5)), 0.85, 2.5)

            vr_arr = df['vol_ratio'].dropna().values[-190:]
            vol_req = clamp(percentile(vr_arr, 70), 1.10, 2.20)

            range_bars = min(18, len(df) - 3)
            s = slice(len(df) - range_bars - 2, len(df) - 2)
            range_high = float(np.max(highs[s]))
            range_low = float(np.min(lows[s]))
            range_size = range_high - range_low

            if range_size < atr * 0.35 or range_size > atr * 7.5:
                return neutral_component("Impulse: no range")

            c_open = float(opens[-2])
            c_close = float(closes[-2])
            c_vol = float(vol[-2])
            c_body = abs(c_close - c_open)
            vol_ratio = c_vol / max(float(np.mean(vol[-20:])), 1e-9)

            bull = c_close > range_high and c_close > c_open
            bear = c_close < range_low and c_close < c_open

            if not bull and not bear:
                return neutral_component("Impulse: no break")

            if c_body < atr * body_req or vol_ratio < vol_req:
                return neutral_component("Impulse: weak break")

            direction = 1 if bull else -1
            strength = clamp(
                (c_body / atr) * 0.32 + vol_ratio * 0.28 + abs(c_close - (range_high if bull else range_low)) / atr * 0.25,
                0.0,
                1.0
            )

            return {
                "direction": direction,
                "strength": strength,
                "label": f"Impulse:{'UP' if direction==1 else 'DOWN'} body:{c_body/atr:.1f}x vol:{vol_ratio:.1f}x",
                "weight": 1.0,
                "reliability": 0.5,
            }
        except Exception:
            return neutral_component("Impulse: error")

    def pullback(self, df, state: dict, alignment_comp: dict):
        try:
            if df is None or len(df) < 45 or 'ema_21' not in df.columns:
                return neutral_component("Pullback: insufficient")

            m5 = alignment_comp.get("m5", 0)
            m15 = alignment_comp.get("m15", 0)
            if m5 == 0 and m15 == 0:
                return neutral_component("Pullback: no HTF trend")

            curr = safe_float(df['close'].iloc[-1])
            ema21 = safe_float(df['ema_21'].iloc[-1])
            ema50 = safe_float(df['ema_50'].iloc[-1])
            atr = max(safe_float(state.get("atr", 1.0)), XAUUSD_POINT)
            rsi = safe_float(df['rsi'].iloc[-1], 50.0)

            highs_30 = df['high'].values[-30:]
            lows_30 = df['low'].values[-30:]
            swing_hi = float(np.max(highs_30))
            swing_lo = float(np.min(lows_30))
            full_range = swing_hi - swing_lo
            if full_range < atr * 0.5:
                return neutral_component("Pullback: range too small")

            trend = state.get("trend_score", 0.5)
            vol_rank = state.get("vol_rank", 0.5)
            depth_min = clamp(0.15 + 0.05 * trend, 0.12, 0.30)
            depth_max = clamp(0.68 - 0.08 * vol_rank, 0.45, 0.75)

            direction = 0
            strength = 0.0
            label = "Pullback: none"

            if m5 >= 0 and m15 >= 0:
                depth = (swing_hi - curr) / full_range
                at_ema21 = abs(curr - ema21) <= atr * clamp(0.55 + 0.20 * vol_rank, 0.45, 0.85)
                rsi_ok = 35 <= rsi <= 55
                if depth_min <= depth <= depth_max and at_ema21 and rsi_ok:
                    direction = 1
                    strength = clamp(0.52 + trend * 0.28 + (0.10 if curr > ema50 else 0.0), 0.0, 1.0)
                    label = f"PullbackBuy depth:{depth:.0%}"

            if m5 <= 0 and m15 <= 0:
                depth = (curr - swing_lo) / full_range
                at_ema21 = abs(curr - ema21) <= atr * clamp(0.55 + 0.20 * vol_rank, 0.45, 0.85)
                rsi_ok = 45 <= rsi <= 65
                if depth_min <= depth <= depth_max and at_ema21 and rsi_ok:
                    direction = -1
                    strength = clamp(0.52 + trend * 0.28 + (0.10 if curr < ema50 else 0.0), 0.0, 1.0)
                    label = f"PullbackSell depth:{depth:.0%}"

            return {
                "direction": direction,
                "strength": strength,
                "label": label,
                "weight": 1.0,
                "reliability": 0.5,
            }
        except Exception:
            return neutral_component("Pullback: error")

    def candle_tick(self, df, micro: MicrostructureEngine, state: dict):
        try:
            if df is None or len(df) < 8:
                return neutral_component("Candle: insufficient")

            atr = max(safe_float(state.get("atr", 1.0)), XAUUSD_POINT)
            o = float(df['open'].iloc[-1])
            h = float(df['high'].iloc[-1])
            l = float(df['low'].iloc[-1])
            c = float(df['close'].iloc[-1])

            body = c - o
            rng = h - l
            body_pct = abs(body) / rng if rng > 0 else 0.0

            br_arr = df['body_ratio'].dropna().values[-170:]
            body_req = clamp(percentile(br_arr, 65), 0.42, 0.80)

            direction = 0
            strength = 0.0
            label = "Candle: neutral"

            running = body_pct >= body_req and rng > atr * 0.42

            if running and body > 0:
                direction = 1
                strength = clamp(0.45 + body_pct * 0.35 + max(0.0, micro.entry_bias()) * 0.25, 0.0, 1.0)
                label = f"Candle:BullRun {body_pct:.0%}"
            elif running and body < 0:
                direction = -1
                strength = clamp(0.45 + body_pct * 0.35 + max(0.0, -micro.entry_bias()) * 0.25, 0.0, 1.0)
                label = f"Candle:BearRun {body_pct:.0%}"
            else:
                bias = micro.entry_bias()
                if bias > 0.22:
                    direction = 1
                    strength = clamp(abs(bias) * 0.50, 0.0, 0.65)
                    label = "Candle:TickBiasUp"
                elif bias < -0.22:
                    direction = -1
                    strength = clamp(abs(bias) * 0.50, 0.0, 0.65)
                    label = "Candle:TickBiasDown"

            return {
                "direction": direction,
                "strength": strength,
                "label": label,
                "weight": 1.0,
                "reliability": 0.5,
                "running": running,
            }
        except Exception:
            return neutral_component("Candle: error")

    def micro_component(self, micro: MicrostructureEngine, state: dict):
        try:
            bias = micro.entry_bias()
            direction = 1 if bias > 0.18 else -1 if bias < -0.18 else 0
            strength = clamp(abs(bias), 0.0, 1.0)
            return {
                "direction": direction,
                "strength": strength,
                "label": f"Micro:{bias:+.2f}",
                "weight": 1.0,
                "reliability": 0.5,
                "bias": bias,
            }
        except Exception:
            return neutral_component("Micro: error")

    def compute_all(self, df_m1, df_m5, df_m15, state: dict, micro: MicrostructureEngine):
        comps = {}
        comps["structure"] = self.structure(df_m1, state)
        comps["alignment"] = self.alignment(df_m1, df_m5, df_m15, state)
        comps["momentum"] = self.momentum(df_m1, state)
        comps["order_flow"] = self.order_flow(df_m1, state)
        comps["volatility"] = self.volatility(df_m1, state)
        comps["smc"] = self.smc(df_m1, state)
        comps["divergence"] = self.divergence(df_m1, state)
        comps["impulse"] = self.impulse(df_m1, state)
        comps["pullback"] = self.pullback(df_m1, state, comps["alignment"])
        comps["candle"] = self.candle_tick(df_m1, micro, state)
        comps["micro"] = self.micro_component(micro, state)
        return comps

#============================================================================
# ADAPTIVE AGGREGATOR
#============================================================================

class AdaptiveSignalAggregator:
    def __init__(self, memory: LearningMemory):
        self.memory = memory

    def aggregate(self, comps: dict, context: dict):
        buy = 0.0
        sell = 0.0
        contributions = {c: 0.0 for c in COMPONENTS}

        for name, out in comps.items():
            if name not in COMPONENTS:
                continue

            direction = safe_int(out.get("direction", 0), 0)
            strength = safe_float(out.get("strength", 0.0), 0.0)
            if direction == 0 or strength <= 0:
                continue

            weight = safe_float(out.get("weight", 1.0), 1.0)
            reliability = safe_float(out.get("reliability", 0.5), 0.5)
            rel_factor = clamp(1.0 + (reliability - 0.5) * 1.25, 0.60, 1.45)

            value = strength * weight * rel_factor

            if direction > 0:
                buy += value
                contributions[name] = value
            elif direction < 0:
                sell += value
                contributions[name] = value

        total = buy + sell + 1e-9
        agreement = abs(buy - sell) / total
        conflict = min(buy, sell) / total

        # Contradiction penalty
        penalty = 1.0 - clamp(conflict * 0.70, 0.0, 0.45)
        buy *= penalty
        sell *= penalty

        # Confirmation bonus
        if buy > sell:
            buy *= (1.0 + 0.12 * agreement)
        elif sell > buy:
            sell *= (1.0 + 0.12 * agreement)

        lead_score = max(buy, sell)
        leading_sign = 1 if buy >= sell else -1

        if lead_score <= 0.02 or agreement < 0.08:
            direction_sign = 0
        else:
            direction_sign = leading_sign

        scale = self.memory.get_score_scale(context)
        raw_score = clamp(lead_score / max(scale, 1e-6), 0.0, 1.0)

        return {
            "leading_sign": leading_sign,
            "direction_sign": direction_sign,
            "buy_score": float(buy),
            "sell_score": float(sell),
            "lead_score": float(lead_score),
            "raw_score": float(raw_score),
            "agreement": float(agreement),
            "conflict": float(conflict),
            "contributions": contributions,
        }

#============================================================================
# ADAPTIVE EXIT MODEL
#============================================================================

class AdaptiveExitModel:
    @staticmethod
    def get_sl_tp(memory: LearningMemory, context: dict, state: dict, entry_type: str):
        stats = memory.context_stats(context)
        trend = state.get("trend_score", 0.5)
        rang = state.get("range_score", 0.5)
        vol = state.get("vol_rank", 0.5)

        sl = 1.70 + 0.30 * rang - 0.15 * trend + 0.25 * vol
        tp = 2.60 + 2.20 * trend - 0.80 * rang + 0.30 * vol

        if stats.get("n", 0) >= 15:
            mfe = stats.get("avg_mfe_r")
            mae = stats.get("avg_mae_r")
            if mfe is not None:
                tp = min(tp, clamp(safe_float(mfe, tp) * 0.95, 1.8, 6.5))
            if mae is not None:
                sl = max(sl, clamp(safe_float(mae, sl) * 1.10, 1.2, 2.6))

        if entry_type == "IMPULSE_START":
            sl *= 1.03
            tp *= 1.04
        elif entry_type == "PULLBACK_CONTINUATION":
            sl *= 0.97
            tp *= 0.97

        return clamp(sl, 1.20, 2.70), clamp(tp, 1.80, 6.80)

    @staticmethod
    def get_management_params(memory: LearningMemory, context: dict, state: dict, entry_type: str):
        stats = memory.context_stats(context)
        vol_rank = state.get("vol_rank", 0.5)

        trail_atr = 0.85 + 0.75 * vol_rank
        if entry_type == "IMPULSE_START":
            trail_atr *= 1.05
        elif entry_type == "PULLBACK_CONTINUATION":
            trail_atr *= 0.95

        mfe = stats.get("avg_mfe_r")
        if mfe is not None:
            mfe = safe_float(mfe, 1.0)
            if mfe < 1.0:
                trail_atr *= 0.93
            elif mfe > 2.0:
                trail_atr *= 1.04

            be_r = clamp(mfe * 0.30, 0.30, 0.70)
            lock_r = clamp(mfe * 0.55, 0.55, 1.20)
            trail_start_r = clamp(mfe * 0.85, 0.90, 2.00)
        else:
            be_r = EARLY_BE_R
            lock_r = PROFIT_LOCK_R
            trail_start_r = TRAIL_START_R

        return {
            "trail_atr": clamp(trail_atr, 0.60, 2.00),
            "be_r": be_r,
            "lock_r": lock_r,
            "trail_start_r": trail_start_r,
        }

#============================================================================
# ADAPTIVE SIGNAL ENGINE
#============================================================================

class AdaptiveSignalEngine:
    def __init__(self, memory: LearningMemory, spread_model: AdaptiveSpreadModel, micro: MicrostructureEngine):
        self.memory = memory
        self.spread_model = spread_model
        self.micro = micro
        self.components = AdaptiveComponentEngine(memory)
        self.weight_engine = AdaptiveWeightEngine()
        self.threshold_engine = AdaptiveThresholdEngine()
        self.probability = AdaptiveProbabilityModel()
        self.aggregator = AdaptiveSignalAggregator(memory)

    def _entry_type(self, leading_sign: int, comps: dict):
        imp = comps.get("impulse", {})
        pb = comps.get("pullback", {})

        if imp.get("direction", 0) == leading_sign and safe_float(imp.get("strength", 0.0)) > 0.35:
            return "IMPULSE_START"
        if pb.get("direction", 0) == leading_sign and safe_float(pb.get("strength", 0.0)) > 0.25:
            return "PULLBACK_CONTINUATION"
        return "STANDARD"

    def _entry_quality(self, direction_sign: int, comps: dict, agg: dict, state: dict,
                        calibrated: float, tp_mult: float):
        try:
            if direction_sign == 0:
                return 0.0

            tick = mt5.symbol_info_tick(SYMBOL)
            df = None
            # df is not passed to avoid expensive re-fetch; use latest component state only.
            # EMA distance approximation uses micro and state. If unavailable, neutral.
            if not tick:
                return 0.35

            atr = max(safe_float(state.get("atr", 1.0)), XAUUSD_POINT)
            spread_price = self.spread_model.current * XAUUSD_POINT
            expected_move = atr * max(tp_mult, 1.0)

            cost_ratio = spread_price / max(expected_move, 1e-6)
            spread_score = clamp(1.0 - cost_ratio / 0.25, 0.0, 1.0)

            bias = self.micro.entry_bias()
            micro_score = clamp(0.5 + 0.5 * bias * direction_sign, 0.0, 1.0)

            momentum = comps.get("momentum", {})
            momentum_score = safe_float(momentum.get("strength", 0.0)) if momentum.get("direction", 0) == direction_sign else 0.0

            candle = comps.get("candle", {})
            candle_score = safe_float(candle.get("strength", 0.0)) if candle.get("direction", 0) == direction_sign else 0.0

            structure = comps.get("structure", {})
            structure_score = safe_float(structure.get("strength", 0.0)) if structure.get("direction", 0) == direction_sign else 0.0

            agreement = safe_float(agg.get("agreement", 0.0))
            conflict = safe_float(agg.get("conflict", 0.0))

            eq = (
                0.24 * calibrated
                + 0.16 * micro_score
                + 0.14 * momentum_score
                + 0.12 * candle_score
                + 0.10 * structure_score
                + 0.12 * spread_score
                + 0.12 * agreement
                - 0.16 * conflict
            )

            return clamp(eq, 0.0, 1.0)
        except Exception:
            return 0.35

    def analyze(self, df_m1, df_m5, df_m15, state: dict, session_name: str, news_state: str):
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None or df_m1 is None:
            return None

        spread_pts = (tick.ask - tick.bid) / XAUUSD_POINT
        self.spread_model.update(spread_pts, session_name)

        context = {
            "session": session_name,
            "regime": state.get("regime", "NORMAL"),
            "vol_bucket": state.get("vol_state", "NORMAL"),
            "spread_bucket": self.spread_model.state,
            "direction": None,
            "entry_type": "STANDARD",
        }

        comps = self.components.compute_all(df_m1, df_m5, df_m15, state, self.micro)

        # Preliminary aggregate for entry type inference
        prelim = self.aggregator.aggregate(comps, context)
        leading_sign = prelim["leading_sign"]
        entry_type = self._entry_type(leading_sign, comps)

        context["direction"] = "BUY" if leading_sign > 0 else "SELL"
        context["entry_type"] = entry_type

        weights, reliabilities, reliability_avg = self.weight_engine.get_weights(self.memory, context)

        for c in COMPONENTS:
            comps[c]["weight"] = weights.get(c, 1.0)
            comps[c]["reliability"] = reliabilities.get(c, 0.5)

        agg = self.aggregator.aggregate(comps, context)

        leading_sign = agg["leading_sign"]
        direction_sign = agg["direction_sign"]
        raw_score = agg["raw_score"]
        agreement = agg["agreement"]
        conflict = agg["conflict"]

        threshold = self.threshold_engine.get(
            self.memory,
            context,
            state,
            news_state,
            self.spread_model
        )

        sl_mult, tp_mult = AdaptiveExitModel.get_sl_tp(self.memory, context, state, entry_type)

        features = np.array(
            [
                1.0,
                clamp(raw_score, 0.0, 1.0),
                clamp(agreement, 0.0, 1.0),
                clamp(conflict, 0.0, 1.0),
                clamp(reliability_avg, 0.0, 1.0),
                0.5,  # entry_quality placeholder, updated below
                clamp(state.get("vol_rank", 0.5), 0.0, 1.0),
                clamp(state.get("spread_rank", 0.5), 0.0, 1.0),
                0.0 if news_state == "NORMAL" else 1.0,
                1.0 if leading_sign > 0 else 0.0,
            ],
            dtype=float
        )

        calibrated = self.probability.calibrate(self.memory, raw_score, context, features)

        entry_quality = self._entry_quality(direction_sign, comps, agg, state, calibrated, tp_mult)
        features[5] = clamp(entry_quality, 0.0, 1.0)
        calibrated = self.probability.calibrate(self.memory, raw_score, context, features)

        entry_score = clamp(0.58 * calibrated + 0.42 * entry_quality, 0.0, 1.0)

        allowed = (
            direction_sign != 0
            and entry_score >= threshold
            and calibrated >= max(0.45, threshold - 0.07)
            and entry_quality >= max(0.34, threshold - 0.18)
            and conflict < 0.62
        )

        direction = "NO_TRADE"
        if allowed:
            direction = "BUY" if direction_sign > 0 else "SELL"

        leading_dir = "BUY" if leading_sign > 0 else "SELL"
        atr = safe_float(state.get("atr", 1.0), 1.0)

        max_age = clamp(3.0 - state.get("vol_rank", 0.5) * 2.0, 0.8, 3.0)
        chase_limit = clamp(0.28 + 0.22 * (1.0 - state.get("vol_rank", 0.5)), 0.18, 0.50)

        report = {
            "Structure": comps["structure"]["label"],
            "Alignment": comps["alignment"]["label"],
            "Momentum": comps["momentum"]["label"],
            "OrderFlow": comps["order_flow"]["label"],
            "Volatility": comps["volatility"]["label"],
            "SMC": comps["smc"]["label"],
            "Divergence": comps["divergence"]["label"],
            "Impulse": comps["impulse"]["label"],
            "Pullback": comps["pullback"]["label"],
            "Candle": comps["candle"]["label"],
            "Micro": comps["micro"]["label"],
            "EntryType": entry_type,
            "_buy_score": agg["buy_score"],
            "_sell_score": agg["sell_score"],
            "_raw_score": raw_score,
            "_calibrated_prob": calibrated,
            "_entry_quality": entry_quality,
            "_entry_score": entry_score,
            "_threshold": threshold,
            "_agreement": agreement,
            "_conflict": conflict,
            "_lead_score": agg["lead_score"],
            "_contributions": agg["contributions"],
            "_momentum_dir": comps["momentum"].get("direction", 0),
            "_momentum_strength": safe_float(comps["momentum"].get("strength", 0.0)),
            "_orderflow_dir": comps["order_flow"].get("direction", 0),
            "_orderflow_strength": safe_float(comps["order_flow"].get("strength", 0.0)),
            "_candle_dir": comps["candle"].get("direction", 0),
            "_candle_running": bool(comps["candle"].get("running", False)),
            "_micro_bias": safe_float(comps["micro"].get("bias", 0.0)),
        }

        return {
            "ts": time.time(),
            "direction": direction,
            "leading_dir": leading_dir,
            "direction_sign": direction_sign,
            "leading_sign": leading_sign,
            "entry_score": entry_score,
            "raw_score": raw_score,
            "calibrated_prob": calibrated,
            "entry_quality": entry_quality,
            "threshold": threshold,
            "agreement": agreement,
            "conflict": conflict,
            "lead_score": agg["lead_score"],
            "context": context,
            "entry_type": entry_type,
            "regime": state.get("regime", "NORMAL"),
            "atr": atr,
            "sl_mult": sl_mult,
            "tp_mult": tp_mult,
            "mid": (tick.ask + tick.bid) / 2.0,
            "spread_pts": spread_pts,
            "max_age": max_age,
            "chase_limit": chase_limit,
            "expected_move": atr * tp_mult,
            "report": report,
            "components": comps,
        }

#============================================================================
# TRADING BOT
#============================================================================

class TradingBot:
    def __init__(self, memory: LearningMemory, spread_model: AdaptiveSpreadModel,
                 micro: MicrostructureEngine, signal_engine: AdaptiveSignalEngine):
        self.memory = memory
        self.spread_model = spread_model
        self.micro = micro
        self.signal_engine = signal_engine
        self.news = NewsAdaptiveFilter()

        self.cooldown_until = datetime.now()
        self.daily_realized_pnl = 0.0
        self.daily_trades = 0
        self.day_start_equity = None
        self.daily_loss_realized = 0.0
        self._last_reset = datetime.now().date()

        self.open_meta = {}
        self.last_atr = 1.0
        self._last_meta_save = time.time()
        self._last_meta_sync = 0.0
        self._last_fast_protect = 0.0

        self._load_open_meta()

    #------------------------------------------------------------------------
    # META / STATE
    #------------------------------------------------------------------------

    def _load_open_meta(self):
        try:
            if os.path.exists(OPEN_META_FILE):
                with open(OPEN_META_FILE, "r") as f:
                    data = json.load(f)
                self.open_meta = data.get("open_meta", {})
        except Exception as e:
            log("WARN", f"Open meta load failed: {e}")
            self.open_meta = {}

    def _save_open_meta(self):
        try:
            with open(OPEN_META_FILE, "w") as f:
                json.dump({"open_meta": self.open_meta}, f, indent=2, default=json_default)
            self._last_meta_save = time.time()
        except Exception as e:
            log("WARN", f"Open meta save failed: {e}")

    def _maybe_save_meta(self, force=False):
        if force or time.time() - self._last_meta_save > 20:
            self._save_open_meta()

    def _get_live_position(self, ticket):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return None
        ticket = safe_int(ticket, 0)
        for p in positions:
            if p.magic == MAGIC_NUMBER and int(p.ticket) == ticket:
                return p
        return None

    def sync_open_meta_from_live(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return

        changed = False
        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue
            key = str(pos.ticket)
            if key not in self.open_meta:
                self.open_meta[key] = self._fallback_meta(pos)
                changed = True
            meta = self.open_meta[key]
            old_vol = safe_float(meta.get("live_volume", -1.0), -1.0)
            old_sl = safe_float(meta.get("live_sl", -1.0), -1.0)

            meta["live_volume"] = float(pos.volume)
            meta["live_sl"] = float(pos.sl)
            meta["live_tp"] = float(pos.tp)
            meta["live_price_open"] = float(pos.price_open)

            if old_vol != float(pos.volume) or abs(old_sl - float(pos.sl)) > XAUUSD_POINT:
                changed = True

        if changed:
            self._maybe_save_meta()

    def _fallback_meta(self, pos):
        direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
        atr = self.last_atr if self.last_atr > 0 else 1.0

        if pos.sl > 0:
            initial_sl = pos.sl
        else:
            initial_sl = pos.price_open - atr * 2.0 if direction == "BUY" else pos.price_open + atr * 2.0

        sl_dist = abs(pos.price_open - initial_sl)
        risk_usd = self._estimate_risk_usd(sl_dist, pos.volume)

        return {
            "ticket": int(pos.ticket),
            "entry_time_ts": int(getattr(pos, "time", time.time())),
            "session": "UNKNOWN",
            "regime": "UNKNOWN",
            "vol_bucket": "NORMAL",
            "spread_bucket": "NORMAL",
            "direction": direction,
            "entry_type": "STANDARD",
            "entry_score": 0.55,
            "raw_score": 0.55,
            "calibrated_prob": 0.50,
            "entry_quality": 0.50,
            "agreement": 0.0,
            "conflict": 0.0,
            "lead_score": 0.0,
            "reliability_avg": 0.5,
            "entry_price": float(pos.price_open),
            "sl": float(pos.sl),
            "tp": float(pos.tp),
            "initial_sl": float(initial_sl),
            "sl_dist": float(sl_dist),
            "tp_dist": float(abs(pos.tp - pos.price_open)) if pos.tp > 0 else float(atr * 3.0),
            "risk_usd": float(risk_usd),
            "atr": float(atr),
            "spread_pts": self.spread_model.current,
            "mfe_pts": 0.0,
            "mae_pts": 0.0,
            "stage": 0,
            "invalidation_count": 0,
            "contributions": {},
            "news_state": "NORMAL",
        }

    def log_live_state(self, source="LIVE"):
        if not LOG_LIVE_STATE:
            return
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            return

        spread_pts = (tick.ask - tick.bid) / XAUUSD_POINT if XAUUSD_POINT > 0 else 0.0

        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue

            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            curr = tick.bid if is_buy else tick.ask
            profit_pts = ((curr - pos.price_open) if is_buy else (pos.price_open - curr)) / XAUUSD_POINT
            profit_money = safe_float(getattr(pos, "profit", 0.0), 0.0)

            meta = self.open_meta.get(str(pos.ticket), {})
            mfe = safe_float(meta.get("mfe_pts", 0.0))
            mae = safe_float(meta.get("mae_pts", 0.0))
            stage = safe_int(meta.get("stage", 0), 0)

            log(
                "STATE",
                f"{source} | {'BUY' if is_buy else 'SELL'} #{pos.ticket} "
                f"Vol:{pos.volume:.2f} Entry:{pos.price_open:.2f} Curr:{curr:.2f} "
                f"Spread:{spread_pts:.0f}pts PnL:{profit_pts:+.0f}pts/${profit_money:+.2f} "
                f"MFE:{mfe:+.0f} MAE:{mae:+.0f} SL:{pos.sl:.2f} TP:{pos.tp:.2f} Stage:{stage}"
            )

    #------------------------------------------------------------------------
    # RISK / DAILY
    #------------------------------------------------------------------------

    def _reset_daily_if_needed(self):
        today = datetime.now().date()
        if today != self._last_reset:
            log(
                "INFO",
                f"Daily Reset | RealizedPnL:${self.daily_realized_pnl:.2f} | Trades:{self.daily_trades}"
            )
            self.daily_realized_pnl = 0.0
            self.daily_trades = 0
            self.day_start_equity = None
            self.daily_loss_realized = 0.0
            self._last_reset = today

    def _update_daily_equity_drawdown(self) -> float:
        acc = mt5.account_info()
        if acc is None:
            return 0.0
        if self.day_start_equity is None:
            self.day_start_equity = acc.equity
        return max(0.0, self.day_start_equity - acc.equity)

    def is_risk_ok(self) -> bool:
        self._reset_daily_if_needed()
        equity_dd = self._update_daily_equity_drawdown()
        realized_loss = max(0.0, -self.daily_realized_pnl)
        self.daily_loss_realized = max(realized_loss, equity_dd)

        if self.daily_loss_realized >= MAX_DAILY_LOSS_USD:
            log("WARN", f"Daily loss cap hit: ${self.daily_loss_realized:.2f}")
            return False

        if self.daily_trades >= MAX_TRADES_PER_DAY:
            log("WARN", f"Trade cap hit: {self.daily_trades}")
            return False

        return True

    def has_open_position(self) -> bool:
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return False
        return any(p.magic == MAGIC_NUMBER for p in positions)

    #------------------------------------------------------------------------
    # FAST LOOP
    #------------------------------------------------------------------------

    def fast_update(self):
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            return

        spread_pts = get_spread_points()
        session_name = get_session_name()

        self.micro.update(tick, self.last_atr)
        self.spread_model.update(spread_pts, session_name)

        self._update_open_excursions()

        if time.time() - self._last_meta_sync >= 1.0:
            self._last_meta_sync = time.time()
            try:
                self.sync_open_meta_from_live()
            except Exception as e:
                log("ERROR", f"Meta sync error: {e}")

        if time.time() - self._last_fast_protect >= PROFIT_PROTECT_FAST_SECONDS:
            self._last_fast_protect = time.time()
            try:
                self._run_profit_protection(source="FAST")
            except Exception as e:
                log("ERROR", f"Fast profit protection error: {e}")

    def _update_open_excursions(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            return

        changed = False
        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue

            key = str(pos.ticket)
            if key not in self.open_meta:
                self.open_meta[key] = self._fallback_meta(pos)
                changed = True

            meta = self.open_meta[key]

            if pos.type == mt5.ORDER_TYPE_BUY:
                favorable = (tick.bid - pos.price_open) / XAUUSD_POINT
                adverse = (pos.price_open - tick.bid) / XAUUSD_POINT
            else:
                favorable = (pos.price_open - tick.ask) / XAUUSD_POINT
                adverse = (tick.ask - pos.price_open) / XAUUSD_POINT

            old_mfe = safe_float(meta.get("mfe_pts", 0.0))
            old_mae = safe_float(meta.get("mae_pts", 0.0))

            meta["mfe_pts"] = max(old_mfe, favorable)
            meta["mae_pts"] = max(old_mae, adverse)

            if meta["mfe_pts"] != old_mfe or meta["mae_pts"] != old_mae:
                changed = True

        if changed:
            self._maybe_save_meta()

    #------------------------------------------------------------------------
    # EXECUTION HELPERS
    #------------------------------------------------------------------------

    def _estimate_risk_usd(self, sl_dist: float, volume: float) -> float:
        try:
            info = mt5.symbol_info(SYMBOL)
            if info:
                if info.trade_tick_size > 0 and info.trade_tick_value > 0:
                    v = sl_dist / info.trade_tick_size * info.trade_tick_value * volume
                    if v > 0:
                        return float(v)
                contract = info.trade_contract_size if info.trade_contract_size > 0 else 100.0
                return float(sl_dist * contract * volume)
        except Exception:
            pass
        return float(sl_dist * 100.0 * volume)

    def _rounded_partial_volume(self, pos, fraction: float) -> float:
        info = mt5.symbol_info(SYMBOL)
        step = safe_float(getattr(info, "volume_step", 0.01), 0.01) if info else 0.01
        minv = safe_float(getattr(info, "volume_min", 0.01), 0.01) if info else 0.01
        if step <= 0:
            step = 0.01
        if minv <= 0:
            minv = 0.01

        current_volume = safe_float(pos.volume, 0.0)
        raw = current_volume * safe_float(fraction, 0.0)
        vol = math.floor(raw / step) * step
        vol = round(vol, 8)

        if vol < minv:
            vol = minv if minv < current_volume else 0.0

        if vol >= current_volume:
            vol = current_volume - step
            if vol < minv:
                vol = 0.0

        return max(0.0, round(vol, 8))

    def _partial_close(self, pos, volume: float, reason: str) -> bool:
        volume = safe_float(volume, 0.0)
        if volume <= 0.0:
            return False

        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            return False

        is_buy = pos.type == mt5.ORDER_TYPE_BUY
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        close_price = tick.bid if is_buy else tick.ask

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": float(volume),
            "type": close_type,
            "position": pos.ticket,
            "price": float(close_price),
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": "v8|pc",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": FILLING_MODE,
        }

        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            time.sleep(0.05)
            live = self._get_live_position(pos.ticket)
            remaining = safe_float(live.volume, safe_float(pos.volume) - volume) if live else safe_float(pos.volume) - volume
            log("MGMT", f"Partial close confirmed #{pos.ticket} closed:{volume:.2f} remaining:{remaining:.2f} | {reason}")
            self.log_live_state(source="AFTER-PC")
            return True

        rc = res.retcode if res else "N/A"
        log("WARN", f"Partial close failed [{rc}] #{pos.ticket} vol:{volume:.2f}")
        return False

    def _try_modify_sl(self, pos, desired_sl: float, curr_price: float, reason: str) -> bool:
        info = mt5.symbol_info(SYMBOL)
        stop_level_pts = safe_int(getattr(info, "trade_stops_level", 0), 0) if info else 0
        stop_dist = max(stop_level_pts * XAUUSD_POINT, 2 * XAUUSD_POINT)

        is_buy = pos.type == mt5.ORDER_TYPE_BUY

        if is_buy:
            max_allowed = curr_price - stop_dist
            desired_sl = min(desired_sl, max_allowed)
            if desired_sl <= 0:
                return False
            if pos.sl != 0 and desired_sl <= pos.sl + XAUUSD_POINT * 0.5:
                return False
        else:
            min_allowed = curr_price + stop_dist
            desired_sl = max(desired_sl, min_allowed)
            if pos.sl != 0 and desired_sl >= pos.sl - XAUUSD_POINT * 0.5:
                return False

        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": SYMBOL,
            "sl": float(desired_sl),
            "tp": float(pos.tp),
        })

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            time.sleep(0.03)
            live = self._get_live_position(pos.ticket)
            actual_sl = safe_float(live.sl, desired_sl) if live else desired_sl
            log("MGMT", f"{reason} #{pos.ticket} SL confirmed:{actual_sl:.2f} requested:{desired_sl:.2f}")
            self.log_live_state(source=f"AFTER-{reason}")
            return True

        rc = result.retcode if result else "N/A"
        log("WARN", f"SL modify failed [{rc}] ticket:{pos.ticket}")
        return False

    def _profit_protect_thresholds(self, atr: float, initial_risk: float, spread_price: float, meta=None):
        meta = meta or {}
        p1_r = min(safe_float(meta.get("partial_1_r", PARTIAL_1_R), PARTIAL_1_R), PARTIAL_1_R)
        p2_r = min(safe_float(meta.get("partial_2_r", PARTIAL_2_R), PARTIAL_2_R), PARTIAL_2_R)
        be_r = min(safe_float(meta.get("be_r", EARLY_BE_R), EARLY_BE_R), EARLY_BE_R)
        lock_r = min(safe_float(meta.get("lock_r", PROFIT_LOCK_R), PROFIT_LOCK_R), PROFIT_LOCK_R)
        trail_r = min(safe_float(meta.get("trail_start_r", TRAIL_START_R), TRAIL_START_R), TRAIL_START_R)

        p1 = max(p1_r * initial_risk, 0.45 * atr, spread_price * 2.0)
        p2 = max(p2_r * initial_risk, 0.90 * atr, spread_price * 3.0)
        be = max(be_r * initial_risk, 0.60 * atr, spread_price * 2.0)
        lock = max(lock_r * initial_risk, 1.00 * atr, spread_price * 3.0)
        trail = max(trail_r * initial_risk, 1.60 * atr)

        return p1, p2, be, lock, trail

    def _run_profit_protection(self, source="FAST"):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return

        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            return

        atr = self.last_atr if self.last_atr > 0 else 1.0
        spread_price = safe_float(self.spread_model.current, 0.0) * XAUUSD_POINT
        point = XAUUSD_POINT

        info = mt5.symbol_info(SYMBOL)
        stop_level_pts = safe_int(getattr(info, "trade_stops_level", 0), 0) if info else 0
        stop_dist = max(stop_level_pts * point, 2 * point)

        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue

            key = str(pos.ticket)
            if key not in self.open_meta:
                self.open_meta[key] = self._fallback_meta(pos)

            meta = self.open_meta[key]
            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            curr = tick.bid if is_buy else tick.ask
            profit_dist = (curr - pos.price_open) if is_buy else (pos.price_open - curr)

            if profit_dist <= 0:
                continue

            initial_sl = safe_float(meta.get("initial_sl", pos.sl), pos.sl)
            initial_risk = abs(pos.price_open - initial_sl) if initial_sl > 0 else atr * 2.0
            if initial_risk <= 0:
                initial_risk = atr * 2.0

            p1, p2, be_th, lock_th, trail_th = self._profit_protect_thresholds(
                atr, initial_risk, spread_price, meta
            )

            stage = safe_int(meta.get("stage", 0), 0)
            profit_pts = profit_dist / point

            if PARTIAL_PROFIT_ENABLE and profit_dist >= p1 and not meta.get("partial_1", False):
                vol = self._rounded_partial_volume(pos, PARTIAL_CLOSE_1_PCT)
                if vol > 0:
                    if self._partial_close(pos, vol, f"PC1 +{profit_pts:.0f}pts"):
                        meta["partial_1"] = True
                        meta["partial_1_vol"] = float(vol)
                        meta["last_partial_ts"] = time.time()
                        self._maybe_save_meta(force=True)

                        time.sleep(0.05)
                        tick2 = mt5.symbol_info_tick(SYMBOL)
                        curr2 = tick2.bid if is_buy else tick2.ask if tick2 else curr
                        be_sl = (pos.price_open + 2 * point) if is_buy else (pos.price_open - 2 * point)
                        if self._try_modify_sl(pos, be_sl, curr2, "BE-PC1"):
                            meta["stage"] = max(stage, 2)
                            self._maybe_save_meta(force=True)
                        continue
                else:
                    meta["partial_1"] = True
                    meta["partial_2"] = True
                    meta["partial_1_vol"] = 0.0
                    meta["partial_2_vol"] = 0.0
                    self._maybe_save_meta(force=True)

            if (
                PARTIAL_PROFIT_ENABLE
                and meta.get("partial_1", False)
                and not meta.get("partial_2", False)
                and profit_dist >= p2
                and time.time() - safe_float(meta.get("last_partial_ts", 0.0), 0.0) > 2.0
            ):
                vol = self._rounded_partial_volume(pos, PARTIAL_CLOSE_2_PCT)
                if vol > 0:
                    if self._partial_close(pos, vol, f"PC2 +{profit_pts:.0f}pts"):
                        meta["partial_2"] = True
                        meta["partial_2_vol"] = float(vol)
                        meta["last_partial_ts"] = time.time()
                        self._maybe_save_meta(force=True)

                        time.sleep(0.05)
                        tick2 = mt5.symbol_info_tick(SYMBOL)
                        curr2 = tick2.bid if is_buy else tick2.ask if tick2 else curr
                        lock_sl = (pos.price_open + 0.25 * atr) if is_buy else (pos.price_open - 0.25 * atr)
                        if self._try_modify_sl(pos, lock_sl, curr2, "LOCK-PC2"):
                            meta["stage"] = max(stage, 3)
                            self._maybe_save_meta(force=True)
                        continue
                else:
                    meta["partial_2"] = True
                    meta["partial_2_vol"] = 0.0
                    self._maybe_save_meta(force=True)

            stage = safe_int(meta.get("stage", 0), 0)

            if stage < 2 and profit_dist >= be_th:
                be_sl = (pos.price_open + 2 * point) if is_buy else (pos.price_open - 2 * point)
                if self._try_modify_sl(pos, be_sl, curr, "BE"):
                    meta["stage"] = 2
                    self._maybe_save_meta(force=True)
                    continue

            if stage < 3 and profit_dist >= lock_th:
                lock_sl = (pos.price_open + 0.35 * atr) if is_buy else (pos.price_open - 0.35 * atr)
                if self._try_modify_sl(pos, lock_sl, curr, "LOCK"):
                    meta["stage"] = 3
                    self._maybe_save_meta(force=True)
                    continue

            if stage >= 4 or profit_dist >= trail_th:
                trail_atr = safe_float(meta.get("trail_atr", 1.0), 1.0)
                trail_dist = max(atr * trail_atr, stop_dist, 10 * point)
                new_sl = curr - trail_dist if is_buy else curr + trail_dist
                if self._try_modify_sl(pos, new_sl, curr, "TRAIL"):
                    meta["stage"] = 4
                    self._maybe_save_meta(force=True)

    #------------------------------------------------------------------------
    # CLOSED TRADE LEARNING
    #------------------------------------------------------------------------

    def process_closed_trades(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        live_tickets = set()
        if positions:
            for p in positions:
                if p.magic == MAGIC_NUMBER:
                    live_tickets.add(int(p.ticket))

        out_entries = {
            getattr(mt5, "DEAL_ENTRY_OUT", 1),
            getattr(mt5, "DEAL_ENTRY_INOUT", 2)
        }

        for ticket_s, meta in list(self.open_meta.items()):
            try:
                ticket = int(ticket_s)
            except Exception:
                del self.open_meta[ticket_s]
                continue

            if ticket in live_tickets:
                continue

            meta["_close_attempts"] = safe_int(meta.get("_close_attempts", 0), 0) + 1

            from_ts = safe_int(meta.get("entry_time_ts", int(time.time() - 86400)), int(time.time() - 86400)) - 120
            to_ts = int(time.time()) + 120

            deals = None
            try:
                deals = mt5.history_deals_get(
                    datetime.fromtimestamp(from_ts),
                    datetime.fromtimestamp(to_ts),
                    SYMBOL
                )
            except Exception:
                try:
                    deals = mt5.history_deals_get(
                        datetime.fromtimestamp(from_ts),
                        datetime.fromtimestamp(to_ts),
                        "*"
                    )
                except Exception:
                    deals = None

            if not deals:
                if meta["_close_attempts"] > 12:
                    del self.open_meta[ticket_s]
                    self._maybe_save_meta(force=True)
                continue

            out_deals = []
            for d in deals:
                try:
                    if getattr(d, "symbol", "") != SYMBOL:
                        continue
                    if safe_int(getattr(d, "magic", 0), 0) != MAGIC_NUMBER:
                        continue
                    if safe_int(getattr(d, "position_id", 0), 0) != ticket:
                        continue
                    if safe_int(getattr(d, "entry", -1), -1) in out_entries:
                        out_deals.append(d)
                except Exception:
                    continue

            if not out_deals:
                if meta["_close_attempts"] > 12:
                    del self.open_meta[ticket_s]
                    self._maybe_save_meta(force=True)
                continue

            pnl = 0.0
            for d in out_deals:
                pnl += (
                    safe_float(getattr(d, "profit", 0.0))
                    + safe_float(getattr(d, "commission", 0.0))
                    + safe_float(getattr(d, "swap", 0.0))
                )

            last_deal = max(out_deals, key=lambda d: safe_int(getattr(d, "time", 0), 0))
            exit_price = safe_float(getattr(last_deal, "price", 0.0))
            exit_ts_raw = getattr(last_deal, "time", int(time.time()))
            if isinstance(exit_ts_raw, datetime):
                exit_ts = int(exit_ts_raw.timestamp())
            else:
                exit_ts = safe_int(exit_ts_raw, int(time.time()))

            sl_dist = safe_float(meta.get("sl_dist", 0.0))
            risk_usd = safe_float(meta.get("risk_usd", 0.0))
            r = pnl / risk_usd if risk_usd > 0 else None

            mfe_r = (
                (safe_float(meta.get("mfe_pts", 0.0)) * XAUUSD_POINT) / sl_dist
                if sl_dist > 0 else None
            )
            mae_r = (
                (safe_float(meta.get("mae_pts", 0.0)) * XAUUSD_POINT) / sl_dist
                if sl_dist > 0 else None
            )

            holding = max(0, exit_ts - safe_int(meta.get("entry_time_ts", exit_ts), exit_ts))

            record = dict(meta)
            record.update({
                "exit_price": exit_price,
                "exit_time": exit_ts,
                "pnl": pnl,
                "result": "WIN" if pnl > 0 else "LOSS",
                "r": r,
                "mfe_r": mfe_r,
                "mae_r": mae_r,
                "holding_sec": holding,
                "spread_rank": safe_float(self.spread_model.rank, 50.0) / 100.0,
            })
            record.pop("_close_attempts", None)

            self.memory.add_record(record)
            self.daily_realized_pnl += pnl
            del self.open_meta[ticket_s]
            self._maybe_save_meta(force=True)

            log(
                "LEARN",
                f"{record['result']} {record.get('direction')} {record.get('entry_type')} | "
                f"PnL:{pnl:+.2f} | R:{'NA' if r is None else f'{r:+.2f}'} | "
                f"Session:{record.get('session')} | Regime:{record.get('regime')} | "
                f"N:{len(self.memory.records)}"
            )

    #------------------------------------------------------------------------
    # REVALIDATION
    #------------------------------------------------------------------------

    def revalidate_signal(self, signal: dict, state: dict):
        try:
            if not signal:
                return False, "No signal"

            age = time.time() - safe_float(signal.get("ts", 0.0))
            if age > safe_float(signal.get("max_age", 2.0), 2.0):
                return False, f"Signal stale {age:.2f}s"

            tick = mt5.symbol_info_tick(SYMBOL)
            if not tick:
                return False, "No tick"

            atr = max(safe_float(state.get("atr", signal.get("atr", 1.0)), 1.0), XAUUSD_POINT)
            mid = (tick.ask + tick.bid) / 2.0
            disp = mid - safe_float(signal.get("mid", mid))

            direction = signal.get("direction")
            chase_limit = safe_float(signal.get("chase_limit", 0.35), 0.35)

            if direction == "BUY":
                if disp > atr * chase_limit:
                    return False, f"Chasing BUY {disp/atr:.2f}ATR"
                if disp < -atr * 0.45:
                    return False, f"Adverse move before BUY {disp/atr:.2f}ATR"
            elif direction == "SELL":
                if -disp > atr * chase_limit:
                    return False, f"Chasing SELL {-disp/atr:.2f}ATR"
                if disp > atr * 0.45:
                    return False, f"Adverse move before SELL {disp/atr:.2f}ATR"
            else:
                return False, "No direction"

            spread_pts = get_spread_points()
            if spread_pts > self.spread_model.hard_block_pts:
                return False, f"Spread hard block {spread_pts:.0f}pts"

            if self.spread_model.state == "EXTREME" and safe_float(signal.get("entry_quality", 0.0)) < 0.72:
                return False, "Extreme spread without strong entry quality"

            bias = self.micro.entry_bias()
            if direction == "BUY" and bias < -0.45:
                return False, f"Micro opposite {bias:.2f}"
            if direction == "SELL" and bias > 0.45:
                return False, f"Micro opposite {bias:.2f}"

            spread_price = spread_pts * XAUUSD_POINT
            expected_move = safe_float(signal.get("expected_move", atr * 3.0), atr * 3.0)
            cost_ratio = spread_price / max(expected_move, 1e-6)

            if cost_ratio > 0.35 and safe_float(signal.get("entry_quality", 0.0)) < 0.65:
                return False, f"Spread cost too high {cost_ratio:.2f}"

            return True, "OK"
        except Exception as e:
            return False, f"Revalidation error: {e}"

    #------------------------------------------------------------------------
    # FLIP / ENTRY
    #------------------------------------------------------------------------

    def can_flip(self, pos, signal: dict, state: dict):
        meta = self.open_meta.get(str(pos.ticket), {})
        old_score = safe_float(meta.get("entry_score", 0.55), 0.55)
        new_score = safe_float(signal.get("entry_score", 0.0), 0.0)

        margin = clamp(0.06 + state.get("vol_rank", 0.5) * 0.03, 0.05, 0.12)

        if new_score < old_score + margin:
            log("MGMT", f"Flip blocked: new score {new_score:.3f} < old {old_score:.3f} + margin {margin:.3f}")
            return False

        if safe_float(signal.get("calibrated_prob", 0.0)) < 0.52:
            log("MGMT", "Flip blocked: calibrated probability too low")
            return False

        report = signal.get("report", {})
        wanted = 1 if signal.get("direction") == "BUY" else -1
        if safe_int(report.get("_momentum_dir", 0), 0) != wanted:
            log("MGMT", "Flip blocked: momentum not aligned")
            return False

        if state.get("vol_state") == "EXTREME_HIGH" and new_score < 0.70:
            log("MGMT", "Flip blocked: extreme volatility requires stronger signal")
            return False

        return True

    def flip_position(self, new_direction: str):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return

        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue

            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            want_buy = new_direction == "BUY"

            if (is_buy and want_buy) or (not is_buy and not want_buy):
                continue

            tick = mt5.symbol_info_tick(SYMBOL)
            if not tick:
                return

            curr = tick.bid if is_buy else tick.ask
            profit_pts = ((curr - pos.price_open) if is_buy else (pos.price_open - curr)) / XAUUSD_POINT

            if profit_pts < -20:
                log("MGMT", f"Flip blocked #{pos.ticket} at {profit_pts:.0f}pts (in loss)")
                continue

            close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if is_buy else tick.ask

            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": float(close_price),
                "deviation": DEVIATION,
                "magic": MAGIC_NUMBER,
                "comment": "v8|flip",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": FILLING_MODE,
            }

            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                self.open_meta.pop(str(pos.ticket), None)
                self._maybe_save_meta(force=True)
                log(
                    "MGMT",
                    f"Flipped #{pos.ticket} {'BUY->SELL' if is_buy else 'SELL->BUY'} @ {close_price:.2f} "
                    f"P&L:{profit_pts:.0f}pts"
                )
                self.log_live_state(source="AFTER-FLIP")
            else:
                rc = res.retcode if res else "N/A"
                log("WARN", f"Flip failed [{rc}] #{pos.ticket}")

    def _find_latest_position_ticket(self, direction: str):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return None

        wanted = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        candidates = [p for p in positions if p.magic == MAGIC_NUMBER and p.type == wanted]
        if not candidates:
            return None

        return max(candidates, key=lambda p: getattr(p, "time", 0)).ticket

    def execute_trade(self, signal: dict, df_m1, state: dict):
        try:
            if not signal or signal.get("direction") not in ("BUY", "SELL"):
                return

            if datetime.now() < self.cooldown_until:
                return

            if not self.is_risk_ok():
                return

            direction = signal["direction"]

            if self.has_open_position():
                positions = mt5.positions_get(symbol=SYMBOL)
                if positions:
                    for p in positions:
                        if p.magic != MAGIC_NUMBER:
                            continue
                        pos_is_buy = p.type == mt5.ORDER_TYPE_BUY
                        same_dir = (pos_is_buy and direction == "BUY") or (not pos_is_buy and direction == "SELL")
                        if same_dir:
                            return
                        else:
                            if not self.can_flip(p, signal, state):
                                return
                            self.flip_position(direction)
                            time.sleep(0.12)
                            if self.has_open_position():
                                return

            ok, reason = self.revalidate_signal(signal, state)
            if not ok:
                log("BLOCK", f"Entry blocked: {reason}")
                return

            atr = safe_float(signal.get("atr", 1.0), 1.0)
            sl_mult = safe_float(signal.get("sl_mult", 2.0), 2.0)
            tp_mult = safe_float(signal.get("tp_mult", 3.5), 3.5)

            sl_dist = atr * sl_mult
            tp_dist = atr * tp_mult

            tick = mt5.symbol_info_tick(SYMBOL)
            if not tick:
                return

            price = tick.ask if direction == "BUY" else tick.bid

            info = mt5.symbol_info(SYMBOL)
            stop_level_pts = safe_int(getattr(info, "trade_stops_level", 0), 0) if info else 0
            stop_dist = max(stop_level_pts * XAUUSD_POINT, 2 * XAUUSD_POINT)

            if sl_dist < stop_dist * 2:
                sl_dist = stop_dist * 2
            if tp_dist < stop_dist * 2:
                tp_dist = stop_dist * 2

            sl = price - sl_dist if direction == "BUY" else price + sl_dist
            tp = price + tp_dist if direction == "BUY" else price - tp_dist

            risk_usd = self._estimate_risk_usd(sl_dist, LOT_SIZE)

            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": LOT_SIZE,
                "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": float(price),
                "sl": float(sl),
                "tp": float(tp),
                "deviation": DEVIATION,
                "magic": MAGIC_NUMBER,
                "comment": f"v8|{str(signal.get('regime','NORM'))[:3]}|{signal.get('entry_score',0):.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": FILLING_MODE,
            }

            max_retries = 3
            for attempt in range(max_retries):
                res = mt5.order_send(req)

                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    context = signal.get("context", {})
                    stats = self.memory.context_stats(context)

                    cd_secs = COOLDOWN_SECONDS
                    exp = stats.get("expectancy_r")
                    if exp is not None:
                        exp = safe_float(exp, 0.0)
                        if exp < -0.10:
                            cd_secs = int(cd_secs * 1.25)
                        elif exp > 0.20:
                            cd_secs = int(cd_secs * 0.90)

                    if state.get("vol_rank", 0.5) > 0.90:
                        cd_secs = int(cd_secs * 1.10)

                    cd_secs = int(clamp(cd_secs, 25, 240))
                    self.daily_trades += 1
                    self.cooldown_until = datetime.now() + __import__("datetime").timedelta(seconds=cd_secs)

                    time.sleep(0.10)
                    ticket = None
                    for _ in range(5):
                        ticket = self._find_latest_position_ticket(direction)
                        if ticket:
                            break
                        time.sleep(0.05)

                    if ticket:
                        meta = {
                            "ticket": int(ticket),
                            "entry_time_ts": int(time.time()),
                            "session": context.get("session", "UNKNOWN"),
                            "regime": signal.get("regime", "UNKNOWN"),
                            "vol_bucket": context.get("vol_bucket", "NORMAL"),
                            "spread_bucket": context.get("spread_bucket", "NORMAL"),
                            "direction": direction,
                            "entry_type": signal.get("entry_type", "STANDARD"),
                            "entry_score": float(signal.get("entry_score", 0.0)),
                            "raw_score": float(signal.get("raw_score", 0.0)),
                            "calibrated_prob": float(signal.get("calibrated_prob", 0.0)),
                            "entry_quality": float(signal.get("entry_quality", 0.0)),
                            "agreement": float(signal.get("agreement", 0.0)),
                            "conflict": float(signal.get("conflict", 0.0)),
                            "lead_score": float(signal.get("lead_score", 0.0)),
                            "reliability_avg": float(signal.get("report", {}).get("reliability_avg", 0.5)),
                            "entry_price": float(price),
                            "sl": float(sl),
                            "tp": float(tp),
                            "initial_sl": float(sl),
                            "sl_dist": float(sl_dist),
                            "tp_dist": float(tp_dist),
                            "risk_usd": float(risk_usd),
                            "atr": float(atr),
                            "spread_pts": float(signal.get("spread_pts", get_spread_points())),
                            "mfe_pts": 0.0,
                            "mae_pts": 0.0,
                            "stage": 0,
                            "invalidation_count": 0,
                            "contributions": signal.get("report", {}).get("_contributions", {}),
                            "news_state": context.get("news_state", "NORMAL"),
                        }
                        self.open_meta[str(ticket)] = meta
                        self._maybe_save_meta(force=True)

                    wr = stats.get("win_rate", 0.0) * 100 if stats.get("n", 0) else 0.0
                    ev = stats.get("expectancy_r", None)

                    log("TRADE", "=" * 72)
                    log(
                        "TRADE",
                        f"ENTRY {direction} | Score:{signal.get('entry_score',0):.3f} | "
                        f"Cal:{signal.get('calibrated_prob',0):.3f} | EQ:{signal.get('entry_quality',0):.3f} | "
                        f"Thr:{signal.get('threshold',0):.3f} | Attempt:{attempt+1}"
                    )
                    log(
                        "TRADE",
                        f"   Entry:{price:.2f} SL:{sl:.2f} TP:{tp:.2f} | Risk:{sl_dist:.2f} RR:{tp_dist/sl_dist:.1f}:1"
                    )
                    log(
                        "TRADE",
                        f"   Session:{context.get('session')} | Regime:{signal.get('regime')} | Entry:{signal.get('entry_type')}"
                    )
                    log(
                        "TRADE",
                        f"   HistWR:{wr:.0f}% | N:{stats.get('n',0)} | EV:{'NA' if ev is None else f'{safe_float(ev):+.2f}R'}"
                    )
                    log("TRADE", "=" * 72)
                    self.log_live_state(source="ENTRY-CONFIRMED")
                    return

                elif res and res.retcode == mt5.TRADE_RETCODE_REQUOTE:
                    time.sleep(0.08)
                    tick = mt5.symbol_info_tick(SYMBOL)
                    if not tick:
                        break
                    price = tick.ask if direction == "BUY" else tick.bid
                    sl = price - sl_dist if direction == "BUY" else price + sl_dist
                    tp = price + tp_dist if direction == "BUY" else price - tp_dist
                    req['price'] = float(price)
                    req['sl'] = float(sl)
                    req['tp'] = float(tp)
                    log("INFO", f"Requote retry #{attempt+1} @ {price:.2f}")
                    continue

                elif res and res.retcode in (mt5.TRADE_RETCODE_PRICE_OFF, mt5.TRADE_RETCODE_PRICE_CHANGED):
                    time.sleep(0.05)
                    tick = mt5.symbol_info_tick(SYMBOL)
                    if not tick:
                        break
                    price = tick.ask if direction == "BUY" else tick.bid
                    sl = price - sl_dist if direction == "BUY" else price + sl_dist
                    tp = price + tp_dist if direction == "BUY" else price - tp_dist
                    req['price'] = float(price)
                    req['sl'] = float(sl)
                    req['tp'] = float(tp)
                    continue

                else:
                    rc = res.retcode if res else "N/A"
                    cm = res.comment if res else "No response"
                    log("ERROR", f"Order failed [{rc}]: {cm}")
                    break

        except Exception as e:
            log("ERROR", f"Execution error: {e}")
            traceback.print_exc()

    #------------------------------------------------------------------------
    # POSITION MANAGEMENT / INVALIDATION
    #------------------------------------------------------------------------

    def manage_positions(self, df_m1, state: dict):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return

        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            return

        atr = safe_float(df_m1['atr'].iloc[-1], self.last_atr) if 'atr' in df_m1.columns else self.last_atr
        self.last_atr = atr

        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue

            key = str(pos.ticket)
            if key not in self.open_meta:
                self.open_meta[key] = self._fallback_meta(pos)

            meta = self.open_meta[key]
            context = {
                "session": meta.get("session"),
                "regime": meta.get("regime"),
                "vol_bucket": meta.get("vol_bucket"),
                "direction": meta.get("direction"),
                "entry_type": meta.get("entry_type"),
                "spread_bucket": meta.get("spread_bucket"),
            }

            mgmt = AdaptiveExitModel.get_management_params(
                self.memory,
                context,
                state,
                meta.get("entry_type", "STANDARD")
            )

            meta["trail_atr"] = safe_float(mgmt.get("trail_atr", 1.0), 1.0)
            meta["be_r"] = safe_float(mgmt.get("be_r", EARLY_BE_R), EARLY_BE_R)
            meta["lock_r"] = safe_float(mgmt.get("lock_r", PROFIT_LOCK_R), PROFIT_LOCK_R)
            meta["trail_start_r"] = safe_float(mgmt.get("trail_start_r", TRAIL_START_R), TRAIL_START_R)

        self._run_profit_protection(source="SLOW")
        self._maybe_save_meta()

    def close_position(self, pos, reason: str):
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            return

        is_buy = pos.type == mt5.ORDER_TYPE_BUY
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        close_price = tick.bid if is_buy else tick.ask

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": float(close_price),
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": "v8|inv",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": FILLING_MODE,
        }

        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            self.open_meta.pop(str(pos.ticket), None)
            self._maybe_save_meta(force=True)
            log("MGMT", f"Early exit #{pos.ticket} | {reason}")
            self.log_live_state(source="AFTER-EXIT")
        else:
            rc = res.retcode if res else "N/A"
            log("WARN", f"Early exit failed [{rc}] #{pos.ticket}")

    def evaluate_invalidation(self, df_m1, signal: dict, state: dict):
        try:
            if not signal:
                return

            positions = mt5.positions_get(symbol=SYMBOL)
            if not positions:
                return

            tick = mt5.symbol_info_tick(SYMBOL)
            if not tick:
                return

            report = signal.get("report", {})
            atr = max(safe_float(state.get("atr", self.last_atr), self.last_atr), XAUUSD_POINT)

            for pos in positions:
                if pos.magic != MAGIC_NUMBER:
                    continue

                key = str(pos.ticket)
                if key not in self.open_meta:
                    self.open_meta[key] = self._fallback_meta(pos)

                meta = self.open_meta[key]
                direction = meta.get("direction", "BUY")
                dir_int = 1 if direction == "BUY" else -1
                opp_int = -dir_int

                entry_ts = safe_int(meta.get("entry_time_ts", int(time.time())), int(time.time()))
                hold_sec = time.time() - entry_ts

                is_buy = pos.type == mt5.ORDER_TYPE_BUY
                curr = tick.bid if is_buy else tick.ask
                profit_dist = (curr - pos.price_open) if is_buy else (pos.price_open - curr)
                profit_pts = profit_dist / XAUUSD_POINT
                adverse_dist = max(0.0, -profit_dist)
                adverse_atr = adverse_dist / atr if atr > 0 else 0.0

                if hold_sec < MIN_HOLD_SECONDS and adverse_atr < 1.0:
                    meta["invalidation_count"] = 0
                    continue

                score = 0.0
                reasons = []

                if adverse_atr > 0.25:
                    score += min(adverse_atr / 1.0, 1.0) * 0.35
                    reasons.append(f"Adverse:{adverse_atr:.2f}ATR")

                opp_score = safe_float(report.get("_sell_score", 0.0)) if direction == "BUY" else safe_float(report.get("_buy_score", 0.0))
                if opp_score > 0.25:
                    score += min(opp_score / 0.60, 1.0) * 0.25
                    reasons.append(f"OppScore:{opp_score:.2f}")

                if safe_int(report.get("_momentum_dir", 0), 0) == opp_int:
                    ms = safe_float(report.get("_momentum_strength", 0.0))
                    if ms > 0.35:
                        score += 0.15 * ms
                        reasons.append("Momentum")

                if safe_int(report.get("_orderflow_dir", 0), 0) == opp_int:
                    ofs = safe_float(report.get("_orderflow_strength", 0.0))
                    if ofs > 0.45:
                        score += 0.15
                        reasons.append("OrderFlow")

                if safe_int(report.get("_candle_dir", 0), 0) == opp_int and report.get("_candle_running", False):
                    score += 0.08
                    reasons.append("CandleRun")

                micro_bias = safe_float(report.get("_micro_bias", 0.0))
                if (opp_int == 1 and micro_bias > 0.45) or (opp_int == -1 and micro_bias < -0.45):
                    score += 0.08
                    reasons.append("Micro")

                if state.get("vol_state") == "EXTREME_HIGH":
                    score += 0.03
                    reasons.append("ExtVol")

                inv_threshold = clamp(0.52 + state.get("vol_rank", 0.5) * 0.06, 0.50, 0.70)

                if score >= inv_threshold and len(reasons) >= 2:
                    meta["invalidation_count"] = safe_int(meta.get("invalidation_count", 0), 0) + 1
                else:
                    meta["invalidation_count"] = 0

                if meta["invalidation_count"] >= 3:
                    if profit_pts > 20 and safe_int(meta.get("stage", 0), 0) >= 2 and score < 0.72:
                        meta["invalidation_count"] = max(1, meta["invalidation_count"] - 1)
                        continue

                    reason_str = " + ".join(reasons)
                    self.close_position(pos, f"INV {score:.2f} | {reason_str}")
                    meta["invalidation_count"] = 0

        except Exception as e:
            log("ERROR", f"Invalidation error: {e}")

#============================================================================
# MAIN LOOP
#============================================================================

if __name__ == "__main__":
    try:
        connect()

        memory = LearningMemory(LEARNING_FILE)
        spread_model = AdaptiveSpreadModel()
        micro = MicrostructureEngine()
        signal_engine = AdaptiveSignalEngine(memory, spread_model, micro)
        bot = TradingBot(memory, spread_model, micro, signal_engine)

        log("INFO", "=" * 78)
        log("INFO", "  XAUUSD APEX BOT v8.0 — FULL ADAPTIVE SIGNAL ARCHITECTURE")
        log("INFO", "=" * 78)
        log("INFO", f"  Symbol   : {SYMBOL} | Lot: {LOT_SIZE} | Point: {XAUUSD_POINT}")
        log("INFO", f"  Engine   : Adaptive components + adaptive weights + entry quality")
        log("INFO", f"  Learning : {LEARNING_FILE} | OpenMeta: {OPEN_META_FILE}")
        log("INFO", f"  Risk     : ${MAX_DAILY_LOSS_USD}/day | {MAX_TRADES_PER_DAY} trades/day")
        log("INFO", f"  Loop     : Fast {FAST_LOOP_SECONDS}s | Slow {SLOW_ANALYSIS_SECONDS}s | HTF {HTF_REFRESH_SECONDS}s")
        log("INFO", "=" * 78)

        last_fast = datetime.now()
        last_slow = datetime.now()
        last_closed_check = datetime.now()
        last_htf = datetime.now()
        last_status = datetime.now()
        last_state_log = datetime.now()

        df_m5_cache = None
        df_m15_cache = None

        while True:
            try:
                now = datetime.now()

                # FAST LOOP
                if (now - last_fast).total_seconds() >= FAST_LOOP_SECONDS:
                    last_fast = now
                    try:
                        bot.fast_update()
                    except Exception as e:
                        log("ERROR", f"Fast loop error: {e}")

                    if (now - last_closed_check).total_seconds() >= 1.0:
                        last_closed_check = now
                        try:
                            bot.process_closed_trades()
                        except Exception as e:
                            log("ERROR", f"Closed-trade processing error: {e}")

                # SLOW ANALYSIS LOOP
                if (now - last_slow).total_seconds() >= SLOW_ANALYSIS_SECONDS:
                    last_slow = now

                    df_m1 = get_market_data(SYMBOL, TIMEFRAME, n=500)
                    if df_m1 is None:
                        time.sleep(0.05)
                        continue

                    bot.last_atr = safe_float(df_m1['atr'].iloc[-1], 1.0) if 'atr' in df_m1.columns else 1.0

                    state = MarketStateEngine.compute(df_m1, spread_model, micro)
                    session_name = get_session_name()
                    news_state = bot.news.update(spread_model, state, df_m1, micro)

                    try:
                        bot.manage_positions(df_m1, state)
                    except Exception as e:
                        log("ERROR", f"Position management error: {e}")

                    if bot.has_open_position() and (now - last_state_log).total_seconds() >= LIVE_STATE_LOG_SECONDS:
                        try:
                            bot.log_live_state("LIVE")
                        except Exception as e:
                            log("ERROR", f"Live state log error: {e}")
                        last_state_log = now

                    if (
                        (now - last_htf).total_seconds() >= HTF_REFRESH_SECONDS
                        or df_m5_cache is None
                        or df_m15_cache is None
                    ):
                        df_m5_cache = get_market_data(SYMBOL, HTF_M5, n=500)
                        df_m15_cache = get_market_data(SYMBOL, HTF_M15, n=500)
                        last_htf = now

                    signal = None
                    try:
                        signal = signal_engine.analyze(
                            df_m1,
                            df_m5_cache,
                            df_m15_cache,
                            state,
                            session_name,
                            news_state
                        )
                    except Exception as e:
                        log("ERROR", f"Analysis error: {e}")
                        traceback.print_exc()

                    if bot.has_open_position() and signal:
                        try:
                            bot.evaluate_invalidation(df_m1, signal, state)
                        except Exception as e:
                            log("ERROR", f"Invalidation error: {e}")

                    if signal and signal.get("direction") in ("BUY", "SELL"):
                        tick_now = mt5.symbol_info_tick(SYMBOL)
                        price_now = f"Bid:{tick_now.bid:.2f} Ask:{tick_now.ask:.2f}" if tick_now else "NoTick"

                        log(
                            "SIGNAL",
                            f"{signal['direction']} | Price:{price_now} | Score:{signal['entry_score']:.3f} | "
                            f"Cal:{signal['calibrated_prob']:.3f} | EQ:{signal['entry_quality']:.3f} | "
                            f"Thr:{signal['threshold']:.3f} | Session:{session_name} | Regime:{state['regime']} | "
                            f"Entry:{signal['entry_type']} | Vol:{state['vol_state']} | Spread:{spread_model.current:.0f}pts"
                        )

                        if now >= bot.cooldown_until and bot.is_risk_ok():
                            try:
                                bot.execute_trade(signal, df_m1, state)
                            except Exception as e:
                                log("ERROR", f"Execution error: {e}")
                                traceback.print_exc()
                        else:
                            log("INFO", f"Signal {signal['direction']} ignored: cooldown/risk")

                    if (now - last_status).total_seconds() >= 20:
                        log(
                            "INFO",
                            f"{session_name} | {state['regime']} | ADX:{state['adx']:.1f} | Chop:{state['chop']:.1f} | "
                            f"Vol:{state['vol_state']} | Speed:{state['market_speed']:.2f} | "
                            f"Spread:{spread_model.current:.0f}pts({spread_model.rank:.0f}%) | "
                            f"Trades:{bot.daily_trades}/{MAX_TRADES_PER_DAY} | "
                            f"Loss:${bot.daily_loss_realized:.2f}/{MAX_DAILY_LOSS_USD} | "
                            f"LearnN:{len(memory.records)}"
                        )
                        last_status = now

                time.sleep(0.01)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                log("ERROR", f"Loop Error: {e}")
                traceback.print_exc()
                time.sleep(2)

    except KeyboardInterrupt:
        mt5.shutdown()
        log("INFO", "Bot stopped cleanly.")
    except Exception as e:
        log("ERROR", f"Fatal: {e}")
        traceback.print_exc()
        mt5.shutdown()