#!/usr/bin/env python3
"""
TrendPulse Breakout Scanner (Fyers data) -> Telegram alerts

Rules: Pine script jaise hi (ATR 21 x 6.3 trailing stop, Aqua = avg(VWMA100, trail),
Trigger = signal candle High/Low, SL = Aqua, Targets = Trigger +/- Gap x N).

Keys CODE MEIN MAT LIKHO. Environment variables use karo:
    FYERS_CLIENT_ID      (jaise 49XXXXXXXX-100)
    FYERS_ACCESS_TOKEN   (sirf eyJ... wala hissa)
    TG_TOKEN, TG_CHAT_ID

Install:  pip install fyers-apiv3 pandas numpy requests
Run:      python tp_fyers_scanner.py                 # market hours mein, due timeframes
          python tp_fyers_scanner.py --dry --force   # test, Telegram nahi bhejega
"""
import os, sys, json, math, time, argparse
import datetime as dt
import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
STATE_FILE = "state.json"

UNIVERSE = ("360ONE,ABB,ABCAPITAL,ADANIENSOL,ADANIENT,ADANIGREEN,ADANIPORTS,ADANIPOWER,ALKEM,AMBER,AMBUJACEM,ANGELONE,APLAPOLLO,APOLLOHOSP,ASHOKLEY,ASIANPAINT,ASTRAL,AUBANK,AUROPHARMA,AXISBANK,"
    "BAJAJ-AUTO,BAJAJFINSV,BAJFINANCE,BANDHANBNK,BANKBARODA,BANKINDIA,BDL,BEL,BHARATFORG,BHARTIARTL,BHEL,BIOCON,BLUESTARCO,BOSCHLTD,BPCL,BRITANNIA,BSE,"
    "CAMS,CANBK,CDSL,CGPOWER,CHOLAFIN,CIPLA,COALINDIA,COFORGE,COLPAL,CONCOR,CROMPTON,CUMMINSIND,CYIENT,"
    "DABUR,DALBHARAT,DELHIVERY,DIVISLAB,DIXON,DLF,DMART,DRREDDY,EICHERMOT,ETERNAL,EXIDEIND,FEDERALBNK,FORTIS,"
    "GAIL,GLENMARK,GMRAIRPORT,GODREJCP,GODREJPROP,GRASIM,HAL,HAVELLS,HCLTECH,HDFCAMC,HDFCBANK,HDFCLIFE,HEROMOTOCO,HFCL,HINDALCO,HINDPETRO,HINDUNILVR,HINDZINC,HUDCO,"
    "ICICIBANK,ICICIGI,ICICIPRULI,IDEA,IDFCFIRSTB,IEX,IGL,INDHOTEL,INDIANB,INDIGO,INDUSINDBK,INDUSTOWER,INFY,INOXWIND,IOC,IRCTC,IREDA,IRFC,ITC,"
    "JINDALSTEL,JIOFIN,JSWENERGY,JSWSTEEL,JUBLFOOD,KALYANKJIL,KAYNES,KEI,KFINTECH,KOTAKBANK,KPITTECH,LAURUSLABS,LICHSGFIN,LICI,LODHA,LT,LTF,LTIM,LUPIN,"
    "M&M,MANAPPURAM,MANKIND,MARICO,MARUTI,MAXHEALTH,MAZDOCK,MFSL,MOTILALOFS,MPHASIS,MUTHOOTFIN,NATIONALUM,NAUKRI,NBCC,NCC,NESTLEIND,NHPC,NMDC,NTPC,NUVAMA,NYKAA,"
    "OBEROIRLTY,OFSS,OIL,ONGC,PAGEIND,PATANJALI,PAYTM,PERSISTENT,PETRONET,PFC,PGEL,PHOENIXLTD,PIDILITIND,PIIND,PNB,PNBHOUSING,POLICYBZR,POLYCAB,POONAWALLA,POWERGRID,PRESTIGE,"
    "RBLBANK,RECLTD,RELIANCE,RVNL,SAIL,SBICARD,SBILIFE,SBIN,SHREECEM,SHRIRAMFIN,SIEMENS,SOLARINDS,SONACOMS,SRF,SUNPHARMA,SUPREMEIND,SUZLON,SWIGGY,SYNGENE,"
    "TATACONSUM,TATAELXSI,TATAPOWER,TATASTEEL,TATATECH,TCS,TECHM,TIINDIA,TITAGARH,TITAN,TMPV,TORNTPHARM,TORNTPOWER,TRENT,TVSMOTOR,"
    "ULTRACEMCO,UNIONBANK,UNITDSPR,UNOMINDA,UPL,VBL,VEDL,VOLTAS,WIPRO,YESBANK,ZYDUSLIFE").split(",")

# tf -> (fyers resolution, history days, minutes per bar; None = daily)
TF = {
    "1m":  ("1",   3,   1),
    "5m":  ("5",   10,  5),
    "15m": ("15",  20,  15),
    "30m": ("30",  40,  30),
    "1h":  ("60",  100, 60),
    "1d":  ("D",   365, None),
}
DEFAULT_TF = ["5m", "15m", "30m", "1h", "1d"]   # 1m rate-limit ki wajah se default mein nahi

ATR_LEN, ATR_MULT, VWMA_LEN, MIN_BARS = 21, 6.3, 100, 130
THROTTLE = 0.35   # Fyers limit ~200 calls/min


class TokenExpired(Exception):
    pass


# ----------------------------------------------------------------------------
# Indicator engine (Pine tpEngine ka port)
# ----------------------------------------------------------------------------
def engine(d, lb):
    c = d["Close"].to_numpy(float)
    h = d["High"].to_numpy(float)
    l = d["Low"].to_numpy(float)
    v = d["Volume"].to_numpy(float)
    n = len(c)

    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(alpha=1 / ATR_LEN, adjust=False, min_periods=ATR_LEN).mean().to_numpy()
    vw = (pd.Series(c * v).rolling(VWMA_LEN).sum() / pd.Series(v).rolling(VWMA_LEN).sum()).to_numpy()

    ts = np.full(n, np.nan)
    pos = np.zeros(n, dtype=int)
    valid = np.where(~np.isnan(atr))[0]
    if len(valid) == 0:
        z = np.zeros(n, bool)
        return z, z, np.full(n, np.nan), np.full(n, np.nan)
    start = int(valid[0])

    for i in range(start, n):
        prev = ts[i - 1] if (i > start and not np.isnan(ts[i - 1])) else 0.0
        loss = ATR_MULT * atr[i]
        cp = c[i - 1]
        if c[i] > prev and cp > prev:
            ts[i] = max(prev, c[i] - loss)
        elif c[i] < prev and cp < prev:
            ts[i] = min(prev, c[i] + loss)
        elif c[i] > prev:
            ts[i] = c[i] - loss
        else:
            ts[i] = c[i] + loss

        if cp < prev and c[i] > prev:
            pos[i] = 1
        elif cp > prev and c[i] < prev:
            pos[i] = -1
        else:
            pos[i] = pos[i - 1]

    aqua = (vw + ts) / 2.0

    buy = np.zeros(n, bool)
    sell = np.zeros(n, bool)
    trig = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    bH = bA = sL = sA = np.nan
    bB = sB = -10 ** 9

    for i in range(start, n):
        if pos[i] == 1 and pos[i - 1] != 1:
            bH, bA, bB = h[i], aqua[i], i
        if pos[i] == -1 and pos[i - 1] != -1:
            sL, sA, sB = l[i], aqua[i], i

        if pos[i] == 1 and not np.isnan(bH) and not np.isnan(bA) and i - bB <= lb and c[i] > bH:
            buy[i], trig[i], sl[i] = True, bH, bA
        if pos[i] == -1 and not np.isnan(sL) and not np.isnan(sA) and i - sB <= lb and c[i] < sL:
            sell[i], trig[i], sl[i] = True, sL, sA

    return buy, sell, trig, sl


# ----------------------------------------------------------------------------
# Fyers data
# ----------------------------------------------------------------------------
def make_fyers():
    from fyers_apiv3 import fyersModel
    cid = os.environ.get("FYERS_CLIENT_ID")
    tok = os.environ.get("FYERS_ACCESS_TOKEN")
    if not cid or not tok:
        sys.exit("FYERS_CLIENT_ID / FYERS_ACCESS_TOKEN environment variable set karo")
    return fyersModel.FyersModel(client_id=cid, token=tok, is_async=False, log_path="")


def fetch_one(fy, sym, res, days, now):
    data = {
        "symbol": f"NSE:{sym}-EQ",
        "resolution": res,
        "date_format": "1",
        "range_from": (now - dt.timedelta(days=days)).strftime("%Y-%m-%d"),
        "range_to": now.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }
    for attempt in range(3):
        r = fy.history(data=data)
        if r.get("s") == "ok":
            break
        msg = str(r.get("message", "")).lower()
        code = r.get("code")
        if code in (-8, -15, -16, -17, -300) or "token" in msg or "authenticate" in msg:
            raise TokenExpired(msg)
        if code == 429 or "limit" in msg:
            time.sleep(3)
            continue
        return None   # no_data / invalid symbol
    else:
        return None

    candles = r.get("candles") or []
    if len(candles) < MIN_BARS:
        return None
    d = pd.DataFrame(candles, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    d.index = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_convert(IST)
    d = d.drop(columns="ts")
    return d[~d.index.duplicated(keep="last")].sort_index()


def fetch(fy, symbols, tf, now):
    res, days, _ = TF[tf]
    out = {}
    for s in symbols:
        try:
            d = fetch_one(fy, s, res, days, now)
        except TokenExpired:
            raise
        except Exception as e:
            print(f"  {s}: {e}")
            d = None
        if d is not None:
            out[s] = d
        time.sleep(THROTTLE)
    return out


def drop_forming(d, minutes, now):
    """Abhi chal rahi (incomplete) candle hata do."""
    idx = d.index
    if minutes:
        end = idx + pd.Timedelta(minutes=minutes)
        sess_close = idx.normalize() + pd.Timedelta(hours=15, minutes=30)
        end = end.where(end <= sess_close, sess_close)
        return d[end <= now]
    if idx[-1].date() == now.date() and (now.hour, now.minute) < (15, 35):
        return d.iloc[:-1]
    return d


def is_due(tf, now, window):
    """Timeframe tabhi scan karo jab uski candle abhi-abhi close hui ho (API calls bachane ke liye)."""
    mins = TF[tf][2]
    t = now.hour * 60 + now.minute
    if mins is None:
        return t >= 15 * 60 + 35
    return (t - (9 * 60 + 15)) % mins < window


# ----------------------------------------------------------------------------
# Scan
# ----------------------------------------------------------------------------
def scan_tf(fy, tf, symbols, lb, catchup, now):
    mins = TF[tf][2]
    data = fetch(fy, symbols, tf, now)
    nbars = 1 if mins is None else max(1, math.ceil(catchup / mins))
    events = []
    for s, d in data.items():
        d = drop_forming(d, mins, now)
        n = len(d)
        if n < MIN_BARS:
            continue
        buy, sell, trig, sl = engine(d, lb)
        for side, arr in (("BUY", buy), ("SELL", sell)):
            fresh = arr & ~np.r_[False, arr[:-1]]   # sirf pehla breakout candle
            for k in range(max(1, n - nbars), n):
                if fresh[k]:
                    gap = abs(trig[k] - sl[k])
                    dr = 1 if side == "BUY" else -1
                    events.append(dict(sym=s, tf=tf, side=side, ts=str(d.index[k]),
                                       close=float(d["Close"].iloc[k]), trig=float(trig[k]),
                                       sl=float(sl[k]), t1=float(trig[k] + dr * gap),
                                       t2=float(trig[k] + dr * gap * 2)))
    print(f"[{tf}] {len(data)} stocks scanned, {len(events)} signal(s)")
    return events


# ----------------------------------------------------------------------------
# State & Telegram
# ----------------------------------------------------------------------------
def load_state():
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
    except Exception:
        st = {}
    cutoff = time.time() - 3 * 86400
    return {k: t for k, t in st.items() if t > cutoff}


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)


def r05(x):
    return f"{round(x / 0.05) * 0.05:.2f}"


def build_messages(events):
    lines = []
    for side, icon in (("BUY", "🟢"), ("SELL", "🔴")):
        for tf in TF:
            evs = [e for e in events if e["side"] == side and e["tf"] == tf]
            if not evs:
                continue
            lines.append(f"{icon} {side} BREAKOUT | {tf}")
            for e in evs:
                lines.append(f"{e['sym']}  LTP {r05(e['close'])} | Trig {r05(e['trig'])} | "
                             f"SL {r05(e['sl'])} | T1 {r05(e['t1'])} | T2 {r05(e['t2'])}")
            lines.append("")
    msgs, cur = [], ""
    for ln in lines:
        if len(cur) + len(ln) + 1 > 3800:
            msgs.append(cur)
            cur = ""
        cur += ln + "\n"
    if cur.strip():
        msgs.append(cur)
    return msgs


def send_telegram(text):
    token, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT_ID")
    if not token or not chat:
        print("TG_TOKEN / TG_CHAT_ID set nahi hai")
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=20)
    if not r.ok:
        print("Telegram error:", r.text)
    return r.ok


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", nargs="+", default=DEFAULT_TF, choices=list(TF))
    ap.add_argument("--lookback", type=int, default=15, help="trigger candle kitne bars tak valid")
    ap.add_argument("--catchup", type=int, default=10, help="pichhle kitne minute ke signals dekhne hain")
    ap.add_argument("--due-window", type=int, default=8, help="candle close ke kitne min baad tak scan karna")
    ap.add_argument("--force", action="store_true", help="market hours / due check ignore karo")
    ap.add_argument("--dry", action="store_true", help="Telegram par mat bhejo, print karo")
    a = ap.parse_args()

    now = dt.datetime.now(IST)
    open_ = now.weekday() < 5 and (9, 10) <= (now.hour, now.minute) <= (15, 50)
    if not open_ and not a.force:
        print("Market band hai, skip (test ke liye --force use karo)")
        return

    tfs = a.tf if a.force else [t for t in a.tf if is_due(t, now, a.due_window)]
    print("Scanning:", tfs or "kuch due nahi")
    if not tfs:
        return

    fy = make_fyers()
    state = load_state()
    new_events = []
    try:
        for tf in tfs:
            for e in scan_tf(fy, tf, UNIVERSE, a.lookback, a.catchup, now):
                key = f"{e['sym']}|{e['tf']}|{e['side']}|{e['ts']}"
                if key not in state:
                    new_events.append(e)
                    state[key] = time.time()
    except TokenExpired as ex:
        print("Fyers token expire/invalid:", ex)
        if time.time() - state.get("_token_warn", 0) > 6 * 3600 and not a.dry:
            send_telegram("⚠️ Fyers access token expire ho gaya. Naya token generate karke update karein.")
            state["_token_warn"] = time.time()
        save_state(state)
        return

    if new_events:
        for m in build_messages(new_events):
            print(m) if a.dry else send_telegram(m)
    else:
        print("Koi naya signal nahi")
    save_state(state)


if __name__ == "__main__":
    main()
