import os
import time
import logging
import urllib.parse
from datetime import datetime, timezone

import requests
import pandas as pd
from google import genai

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
except ImportError:
    ed25519 = None

# ====================== CONFIG ======================
AI_API_KEY = (os.environ.get("AI_API_KEY") or "").strip()
TELEGRAM_TOKEN = (os.environ.get("TELEGRAM_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
CS_API_KEY = (os.environ.get("COINSWITCH_API_KEY") or "").strip()
CS_SECRET_KEY = (os.environ.get("COINSWITCH_SECRET_KEY") or "").strip()

CS_BASE = "https://coinswitch.co"
CS_EXCHANGE = "c2c1"
CS_INTERVAL_MIN = 5
CANDLE_LIMIT_MINUTES = 5 * 80

PORTFOLIO = {
    "BTC": 0.00002404,
    "DOGE": 10.7122,
    "CHILLGUY": 10.8236,
    "MOG": 527837.4,
}

SCAN_BASES = [
    "BTC", "ETH", "SOL", "BNB", "XRP",
    "ADA", "DOGE", "AVAX", "LINK", "DOT",
    "LTC", "NEAR", "SUI", "ARB", "OP",
    "SHIB", "PEPE", "WIF", "BONK", "FLOKI",
]
for b in PORTFOLIO:
    if b not in SCAN_BASES:
        SCAN_BASES.insert(0, b)

STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
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


def send_telegram(message: str) -> None:
    """Send Telegram message or raise so the GitHub Action fails visibly."""
    logger.info(
        "Telegram secrets present? TOKEN=%s CHAT_ID=%s",
        "yes" if TELEGRAM_TOKEN else "NO",
        "yes" if TELEGRAM_CHAT_ID else "NO",
    )
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise SystemExit(
            "Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID in GitHub Actions secrets. "
            "Add them at: Settings → Secrets and variables → Actions"
        )

    if len(message) > 4000:
        message = message[:3990] + "\n...(truncated)"

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    resp = requests.post(url, json=payload, timeout=15)
    if resp.status_code != 200:
        # Retry plain text (HTML parse errors sometimes block delivery)
        logger.warning("HTML send failed (%s): %s", resp.status_code, resp.text[:300])
        payload.pop("parse_mode", None)
        resp = requests.post(url, json=payload, timeout=15)

    if resp.status_code != 200:
        raise SystemExit(
            f"Telegram API error {resp.status_code}: {resp.text[:500]}. "
            "Check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID secrets."
        )
    logger.info("Telegram message delivered")


def cs_sign_request(method: str, path: str, params: dict | None = None):
    method = method.upper()
    if params:
        sep = "&" if "?" in path else "?"
        path = path + sep + urllib.parse.urlencode(params)
    decoded_path = urllib.parse.unquote_plus(path)
    epoch = str(int(time.time() * 1000))
    message = method + decoded_path + epoch
    secret = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(CS_SECRET_KEY))
    signature = secret.sign(message.encode("utf-8")).hex()
    headers = {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": CS_API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "X-AUTH-EPOCH": epoch,
    }
    return headers, decoded_path


def cs_get(path: str, params: dict | None = None):
    headers, full_path = cs_sign_request("GET", path, params)
    return requests.get(CS_BASE + full_path, headers=headers, timeout=15)


def validate_coinswitch_keys() -> bool:
    if not CS_ENABLED:
        logger.info("CoinSwitch keys not fully configured")
        return False
    try:
        r = cs_get("/trade/api/v2/validate/keys")
        if r.status_code == 200:
            logger.info("CoinSwitch API keys validated")
            return True
        logger.warning("CoinSwitch key validation failed: %s %s", r.status_code, r.text[:200])
        return False
    except Exception as e:
        logger.warning("CoinSwitch validation error: %s", e)
        return False


def get_klines_coinswitch(base: str) -> pd.DataFrame | None:
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
            logger.info("CS candles %s: HTTP %s", symbol, r.status_code)
            return None
        payload = r.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not data:
            return None
        rows = []
        for c in data:
            rows.append(
                {
                    "timestamp": int(float(c.get("start_time") or c.get("close_time") or 0)),
                    "open": float(c["o"]),
                    "high": float(c["h"]),
                    "low": float(c["l"]),
                    "close": float(c["c"]),
                    "volume": float(c.get("volume") or 0),
                }
            )
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df if len(df) >= 30 else None
    except Exception as e:
        logger.warning("CS candles error %s: %s", base, e)
        return None


def get_klines_binance(base: str) -> pd.DataFrame | None:
    symbol = f"{base}USDT"
    try:
        r = requests.get(
            f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&limit=100",
            timeout=10,
        )
        if r.status_code != 200:
            r = requests.get(
                f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=5m&limit=100",
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


def get_klines(base: str, prefer_cs: bool):
    if prefer_cs:
        df = get_klines_coinswitch(base)
        if df is not None:
            return df, "CoinSwitch"
    df = get_klines_binance(base)
    if df is not None:
        return df, "Binance"
    return None, "none"


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
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


def check_signals(df: pd.DataFrame):
    if len(df) < max(EMA_SLOW, VOLUME_LOOKBACK) + 5:
        return None
    latest, prev = df.iloc[-1], df.iloc[-2]
    ema_fast, ema_slow, rsi = latest["ema_fast"], latest["ema_slow"], latest["rsi"]
    vol, vol_avg = latest["volume"], latest["vol_avg"]
    volume_ok = vol > (vol_avg * 0.7) if pd.notna(vol_avg) else True
    bullish_cross = prev["ema_fast"] <= prev["ema_slow"] and ema_fast > ema_slow
    bearish_cross = prev["ema_fast"] >= prev["ema_slow"] and ema_fast < ema_slow
    if (ema_fast > ema_slow and rsi < RSI_BUY_MAX and volume_ok) or (
        bullish_cross and rsi < 48 and volume_ok
    ):
        return "BUY"
    if (ema_fast < ema_slow and rsi > RSI_SELL_MIN and volume_ok) or (
        bearish_cross and rsi > 52 and volume_ok
    ):
        return "SELL"
    return None


def when_to_trade(side, rsi):
    if side == "BUY":
        return "NOW (oversold bounce setup)" if rsi < 30 else "NOW / next 5–15 min if price holds"
    if side == "SELL":
        return "NOW (overbought pressure)" if rsi > 70 else "NOW / scale out on next push up"
    if rsi < 35:
        return "WAIT – watch for buy confirmation"
    if rsi > 65:
        return "WAIT – watch for sell confirmation"
    return "WAIT – no clear intraday edge"


def ai_note(base, side, price, rsi):
    prompt = (
        f"Intraday advisor for CoinSwitch (India). Small capital. "
        f"{base}/USDT {side} on 5m. Price {price:.6f}, RSI {rsi:.1f}. "
        f"2 short sentences: what to do on the app and main risk (fees)."
    )
    try:
        return client.models.generate_content(
            model="gemini-3.7-flash", contents=prompt
        ).text.strip()
    except Exception as e:
        logger.error("AI failed: %s", e)
        return "AI note unavailable."


def scan_base(base: str, prefer_cs: bool):
    df, source = get_klines(base, prefer_cs)
    if df is None or len(df) < 30:
        return None
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    signal = check_signals(df)
    price = float(latest["close"])
    rsi = float(latest["rsi"])
    row = {
        "base": base,
        "pair": f"{base}/USDT",
        "side": signal,
        "price": price,
        "rsi": rsi,
        "source": source,
        "when": when_to_trade(signal, rsi),
        "in_portfolio": base in PORTFOLIO,
        "qty": PORTFOLIO.get(base),
    }
    if signal:
        row["stop"] = price * (1 - STOP_LOSS_PCT) if signal == "BUY" else price * (1 + STOP_LOSS_PCT)
        row["target"] = price * (1 + TAKE_PROFIT_PCT) if signal == "BUY" else price * (1 - TAKE_PROFIT_PCT)
        row["reasoning"] = ai_note(base, signal, price, rsi)
    return row


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logger.info("=== CoinSwitch Intraday Advisor (manual trades) ===")

    prefer_cs = validate_coinswitch_keys()
    data_label = "CoinSwitch API" if prefer_cs else "Binance fallback (CS keys missing/invalid)"
    logger.info("Data source preference: %s", data_label)

    results = []
    for base in SCAN_BASES:
        try:
            row = scan_base(base, prefer_cs)
            if row:
                results.append(row)
                logger.info(
                    "%s src=%s side=%s RSI=%.1f when=%s",
                    row["pair"], row["source"], row["side"], row["rsi"], row["when"],
                )
            time.sleep(0.2)
        except Exception as e:
            logger.error("%s error: %s", base, e)

    sells = [r for r in results if r.get("in_portfolio") and r.get("side") == "SELL"]
    holds = [r for r in results if r.get("in_portfolio") and r.get("side") != "SELL"]
    buys = [r for r in results if r.get("side") == "BUY"]
    buys = sorted(
        buys,
        key=lambda x: (0 if x["base"] in {"BTC", "ETH", "SOL", "XRP", "DOGE"} else 1, x["rsi"]),
    )[:5]

    lines = [
        f"📌 INTRADAY ADVICE ({now})",
        f"Data: {data_label}",
        "Action: open CoinSwitch app and trade manually.",
        "",
        "You hold: BTC, DOGE, CHILLGUY, MOG",
        "",
    ]

    action_lines = []
    for r in sells:
        action_lines.append(
            f"🔴 SELL {r['pair']}\n"
            f"When: {r['when']}\n"
            f"Qty: {r.get('qty')} @ {r['price']:.6f} | RSI {r['rsi']:.0f}\n"
            f"{r.get('reasoning', '')[:160]}"
        )
    for r in buys:
        action_lines.append(
            f"🟢 BUY {r['pair']}\n"
            f"When: {r['when']}\n"
            f"Price {r['price']:.6f} | SL {r['stop']:.6f} | TP {r['target']:.6f} | RSI {r['rsi']:.0f}\n"
            f"{r.get('reasoning', '')[:160]}"
        )

    if action_lines:
        lines.append("WHAT / WHEN TO TRADE NOW")
        lines.extend(action_lines)
        lines.append("")
    else:
        lines.append("WHAT / WHEN TO TRADE NOW")
        lines.append("No strong BUY/SELL right now → WAIT (do not force a trade).")
        lines.append("")

    if holds:
        lines.append("Holdings watch:")
        for r in holds:
            lines.append(f"• {r['pair']}: {r['when']} (RSI {r['rsi']:.0f})")
        lines.append("")

    lines.append("Not financial advice. Fees can erase small-capital profits.")

    # Must succeed or job fails
    send_telegram("\n".join(lines))
    logger.info("Intraday advice sent")


if __name__ == "__main__":
    main()
