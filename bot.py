import os
import time
import logging
import urllib.parse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from google import genai

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
except ImportError:
    ed25519 = None

AI_API_KEY = (os.environ.get("AI_API_KEY") or "").strip()
TELEGRAM_TOKEN = (os.environ.get("TELEGRAM_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
CS_API_KEY = (os.environ.get("COINSWITCH_API_KEY") or "").strip()
CS_SECRET_KEY = (os.environ.get("COINSWITCH_SECRET_KEY") or "").strip()

IST = ZoneInfo("Asia/Kolkata")
CS_BASE = "https://coinswitch.co"
CS_EXCHANGE = "c2c1"
CS_INTERVAL_MIN = 5
CANDLE_LIMIT_MINUTES = 5 * 80

FULL_NAME = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "BNB": "BNB",
    "XRP": "XRP", "ADA": "Cardano", "DOGE": "Dogecoin", "AVAX": "Avalanche",
    "LINK": "Chainlink", "DOT": "Polkadot", "LTC": "Litecoin",
    "NEAR": "NEAR Protocol", "SUI": "Sui", "ARB": "Arbitrum", "OP": "Optimism",
    "SHIB": "Shiba Inu", "PEPE": "Pepe", "WIF": "dogwifhat", "BONK": "Bonk",
    "FLOKI": "FLOKI", "CHILLGUY": "Chill Guy", "MOG": "Mog Coin",
}

PORTFOLIO = {
    "BTC": 0.00002404,
    "DOGE": 10.7122,
    "CHILLGUY": 10.8236,
    "MOG": 527837.4,
}

SCAN_BASES = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
    "LTC", "NEAR", "SUI", "ARB", "OP", "SHIB", "PEPE", "WIF", "BONK", "FLOKI",
]
for b in PORTFOLIO:
    if b not in SCAN_BASES:
        SCAN_BASES.insert(0, b)

# YOUR PLAN: every coin trade
STOP_LOSS_PCT = 0.30   # max acceptable loss = -30%
TAKE_PROFIT_PCT = 0.40  # target profit = +40%

RSI_BUY_MAX = 38
RSI_SELL_MIN = 62
EMA_FAST = 9
EMA_SLOW = 21
VOLUME_LOOKBACK = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

if not AI_API_KEY:
    raise SystemExit("Missing AI_API_KEY secret")

client = genai.Client(api_key=AI_API_KEY)
CS_ENABLED = bool(CS_API_KEY and CS_SECRET_KEY and ed25519 is not None)


def coin_name(base: str) -> str:
    return FULL_NAME.get(base, base)


def format_inr(amount: float) -> str:
    if amount >= 1000:
        return f"₹{amount:,.2f}"
    if amount >= 1:
        return f"₹{amount:.4f}"
    if amount >= 0.01:
        return f"₹{amount:.6f}"
    return f"₹{amount:.8f}"


def get_usdt_inr_rate() -> float:
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "tether", "vs_currencies": "inr"},
            timeout=10,
        )
        if r.status_code == 200:
            rate = float(r.json()["tether"]["inr"])
            if rate > 0:
                return rate
    except Exception as e:
        logger.warning("INR rate failed: %s", e)
    return 84.0


def now_ist() -> datetime:
    return datetime.now(IST)


def fmt_ist(dt: datetime) -> str:
    return dt.strftime("%d %b %Y, %I:%M %p IST")


def trade_window_ist(side, rsi):
    n = now_ist()
    if side == "BUY":
        return "BUY NOW", f"Buy between {fmt_ist(n)} and {fmt_ist(n + timedelta(minutes=15))}"
    if side == "SELL":
        return "SELL NOW", f"Sell between {fmt_ist(n)} and {fmt_ist(n + timedelta(minutes=15))}"
    return "WAIT", f"Do not trade now. Next check around {fmt_ist(n + timedelta(minutes=5))}"


def send_telegram(message: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise SystemExit("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID")
    if len(message) > 4000:
        message = message[:3990] + "\n...(truncated)"
    resp = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "disable_web_page_preview": True},
        timeout=15,
    )
    if resp.status_code != 200:
        raise SystemExit(f"Telegram API error {resp.status_code}: {resp.text[:500]}")
    logger.info("Telegram delivered")


def cs_sign_request(method, path, params=None):
    method = method.upper()
    if params:
        sep = "&" if "?" in path else "?"
        path = path + sep + urllib.parse.urlencode(params)
    decoded_path = urllib.parse.unquote_plus(path)
    epoch = str(int(time.time() * 1000))
    message = method + decoded_path + epoch
    secret = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(CS_SECRET_KEY))
    signature = secret.sign(message.encode("utf-8")).hex()
    return {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": CS_API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "X-AUTH-EPOCH": epoch,
    }, decoded_path


def cs_get(path, params=None):
    headers, full_path = cs_sign_request("GET", path, params)
    return requests.get(CS_BASE + full_path, headers=headers, timeout=15)


def validate_coinswitch_keys() -> bool:
    if not CS_ENABLED:
        return False
    try:
        r = cs_get("/trade/api/v2/validate/keys")
        return r.status_code == 200
    except Exception:
        return False


def get_klines_coinswitch(base):
    symbol = f"{base}/USDT"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - CANDLE_LIMIT_MINUTES * 60 * 1000
    try:
        r = cs_get(
            "/trade/api/v2/candles",
            params={
                "exchange": CS_EXCHANGE,
                "symbol": symbol,
                "interval": str(CS_INTERVAL_MIN),
                "start_time": str(start_ms),
                "end_time": str(end_ms),
            },
        )
        if r.status_code != 200:
            return None
        data = r.json().get("data") if isinstance(r.json(), dict) else None
        if not data:
            return None
        rows = [{
            "timestamp": int(float(c.get("start_time") or c.get("close_time") or 0)),
            "open": float(c["o"]), "high": float(c["h"]),
            "low": float(c["l"]), "close": float(c["c"]),
            "volume": float(c.get("volume") or 0),
        } for c in data]
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df if len(df) >= 30 else None
    except Exception:
        return None


def get_klines_binance(base):
    try:
        r = requests.get(
            f"https://api.binance.com/api/v3/klines?symbol={base}USDT&interval=5m&limit=100",
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or isinstance(data, dict):
            return None
        df = pd.DataFrame(data)[[0, 1, 2, 3, 4, 5]]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df
    except Exception:
        return None


def get_klines(base, prefer_cs):
    if prefer_cs:
        df = get_klines_coinswitch(base)
        if df is not None:
            return df, "CoinSwitch"
    df = get_klines_binance(base)
    return (df, "Binance") if df is not None else (None, "none")


def calculate_indicators(df):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["vol_avg"] = df["volume"].rolling(VOLUME_LOOKBACK).mean()
    return df


def check_signals(df):
    if len(df) < max(EMA_SLOW, VOLUME_LOOKBACK) + 5:
        return None
    latest, prev = df.iloc[-1], df.iloc[-2]
    ema_fast, ema_slow, rsi = latest["ema_fast"], latest["ema_slow"], latest["rsi"]
    vol, vol_avg = latest["volume"], latest["vol_avg"]
    volume_ok = vol > (vol_avg * 0.7) if pd.notna(vol_avg) else True
    bullish = prev["ema_fast"] <= prev["ema_slow"] and ema_fast > ema_slow
    bearish = prev["ema_fast"] >= prev["ema_slow"] and ema_fast < ema_slow
    if (ema_fast > ema_slow and rsi < RSI_BUY_MAX and volume_ok) or (bullish and rsi < 48 and volume_ok):
        return "BUY"
    if (ema_fast < ema_slow and rsi > RSI_SELL_MIN and volume_ok) or (bearish and rsi > 52 and volume_ok):
        return "SELL"
    return None


def ai_note(name, side, price_inr, rsi):
    prompt = (
        f"CoinSwitch India intraday. Plan: max loss -30%, target +40%. "
        f"{name} {side}, price ~{price_inr:.2f} INR, RSI {rsi:.1f}. "
        f"2 short sentences. Fees can wipe small capital."
    )
    try:
        return client.models.generate_content(model="gemini-3.7-flash", contents=prompt).text.strip()
    except Exception:
        return "AI note unavailable."


def scan_base(base, prefer_cs, usdt_inr):
    df, source = get_klines(base, prefer_cs)
    if df is None or len(df) < 30:
        return None
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    signal = check_signals(df)
    price_inr = float(latest["close"]) * usdt_inr
    rsi = float(latest["rsi"])
    action, window = trade_window_ist(signal, rsi)
    row = {
        "base": base,
        "name": coin_name(base),
        "side": signal,
        "price_inr": price_inr,
        "rsi": rsi,
        "source": source,
        "action": action,
        "window": window,
        "in_portfolio": base in PORTFOLIO,
        "qty": PORTFOLIO.get(base),
    }
    if signal:
        if signal == "BUY":
            row["stop_inr"] = price_inr * (1 - STOP_LOSS_PCT)
            row["target_inr"] = price_inr * (1 + TAKE_PROFIT_PCT)
        else:
            row["stop_inr"] = price_inr * (1 + STOP_LOSS_PCT)
            row["target_inr"] = price_inr * (1 - TAKE_PROFIT_PCT)
        row["reasoning"] = ai_note(row["name"], signal, price_inr, rsi)
    return row


def main():
    n = now_ist()
    prefer_cs = validate_coinswitch_keys()
    data_label = "CoinSwitch" if prefer_cs else "Binance (CS unavailable)"
    usdt_inr = get_usdt_inr_rate()

    results = []
    for base in SCAN_BASES:
        try:
            row = scan_base(base, prefer_cs, usdt_inr)
            if row:
                results.append(row)
            time.sleep(0.2)
        except Exception as e:
            logger.error("%s: %s", base, e)

    sells = [r for r in results if r.get("in_portfolio") and r.get("side") == "SELL"]
    holds = [r for r in results if r.get("in_portfolio") and r.get("side") != "SELL"]
    buys = [r for r in results if r.get("side") == "BUY"]
    buys = sorted(
        buys,
        key=lambda x: (0 if x["base"] in {"BTC", "ETH", "SOL", "XRP", "DOGE"} else 1, x["rsi"]),
    )[:5]

    lines = [
        f"INTRADAY PLAN — {fmt_ist(n)}",
        f"YOUR RULE: Max loss -30% | Target profit +40% (every coin)",
        f"Data: {data_label} | 1 USDT ≈ ₹{usdt_inr:.2f}",
        "Trade manually on CoinSwitch app.",
        "",
        "WALLET: Bitcoin, Dogecoin, Chill Guy, Mog Coin",
        "",
        "========== CLEAR ACTIONS ==========",
        "",
    ]

    if not sells and not buys:
        lines += [
            "NO TRADE RIGHT NOW",
            f"Time: {fmt_ist(n)}",
            f"Next check: ~{fmt_ist(n + timedelta(minutes=5))}",
            "Reason: no strong 5-minute setup.",
            "",
        ]
    else:
        if sells:
            lines.append("SELL (from wallet)")
            for r in sells:
                lines += [
                    f"Coin: {r['name']} ({r['base']})",
                    f"Action: {r['action']}",
                    f"Time window: {r['window']}",
                    f"Qty: {r.get('qty')}",
                    f"Price now: {format_inr(r['price_inr'])}",
                    f"Stop loss (-30%): {format_inr(r['stop_inr'])}",
                    f"Take profit (+40% path): {format_inr(r['target_inr'])}",
                    f"RSI: {r['rsi']:.0f}",
                    f"Note: {r.get('reasoning', '')[:140]}",
                    "",
                ]
        if buys:
            lines.append("BUY")
            for r in buys:
                lines += [
                    f"Coin: {r['name']} ({r['base']})",
                    f"Action: {r['action']}",
                    f"Time window: {r['window']}",
                    f"Price now: {format_inr(r['price_inr'])}",
                    f"Stop loss (-30%): {format_inr(r['stop_inr'])}",
                    f"Take profit (+40%): {format_inr(r['target_inr'])}",
                    f"RSI: {r['rsi']:.0f}",
                    f"Note: {r.get('reasoning', '')[:140]}",
                    "",
                ]

    if holds:
        lines.append("HOLDINGS WATCH")
        for r in holds:
            lines.append(f"{r['name']}: {r['action']} | {format_inr(r['price_inr'])} | RSI {r['rsi']:.0f}")
            lines.append(f"  {r['window']}")
        lines.append("")

    lines.append("Not financial advice. -30% risk is very high for intraday.")

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
