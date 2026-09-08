import os
import time
import logging
import requests
import pandas as pd
from google import genai

# ====================== CONFIG ======================
AI_API_KEY = os.environ.get("AI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# ~50 coins: majors + alts + popular meme coins (Binance.US pairs)
# Missing pairs are skipped automatically
COINS = [
    # Majors
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "ATOMUSDT", "NEARUSDT", "UNIUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "SUIUSDT", "SEIUSDT", "TIAUSDT",
    # More alts
    "FILUSDT", "ICPUSDT", "HBARUSDT", "VETUSDT", "ALGOUSDT",
    "AAVEUSDT", "MKRUSDT", "GRTUSDT", "SANDUSDT", "MANAUSDT",
    "AXSUSDT", "FTMUSDT", "EGLDUSDT", "XTZUSDT", "EOSUSDT",
    "XLMUSDT", "TRXUSDT", "BCHUSDT", "ETCUSDT", "COMPUSDT",
    # Meme / high volatility
    "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT", "WIFUSDT",
    "MEMEUSDT", "BABYDOGEUSDT", "1000SATSUSDT", "ORDIUSDT", "RATSUSDT",
]

# Risk settings (suggestions only)
STOP_LOSS_PCT = 0.02      # 2%
TAKE_PROFIT_PCT = 0.04    # 4%

# Strategy settings
RSI_BUY_MAX = 38
RSI_SELL_MIN = 62
EMA_FAST = 9
EMA_SLOW = 21
VOLUME_LOOKBACK = 20

BACKTEST_MODE = False
BACKTEST_LIMIT = 200

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ====================== CLIENTS ======================
if not AI_API_KEY and not BACKTEST_MODE:
    logger.error("AI_API_KEY is missing")
    raise SystemExit("Missing AI_API_KEY")

client = genai.Client(api_key=AI_API_KEY) if AI_API_KEY else None

# ====================== HELPERS ======================
def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing – alert not sent")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        # Telegram has a 4096 char limit – split if needed
        if len(message) > 4000:
            message = message[:3990] + "\n...(truncated)"
        resp = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            },
            timeout=12
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def get_klines(symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame | None:
    url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        if not data or isinstance(data, dict):
            return None

        df = pd.DataFrame(data)
        df = df[[0, 1, 2, 3, 4, 5]]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df
    except Exception:
        return None


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


def get_ai_reasoning(symbol: str, side: str, price: float, ema_fast: float, ema_slow: float, rsi: float) -> str:
    if client is None:
        return "AI disabled."

    prompt = (
        f"Act as a quantitative crypto trader. "
        f"{symbol} shows a {side} recommendation on the 1-hour chart. "
        f"Price: ${price:.6f}, EMA{EMA_FAST}: ${ema_fast:.6f}, EMA{EMA_SLOW}: ${ema_slow:.6f}, RSI: {rsi:.1f}. "
        f"Give a concise 2-sentence reasoning. Be realistic and mention key risks."
    )
    try:
        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"AI reasoning failed for {symbol}: {e}")
        return "AI reasoning unavailable."


def check_signals(df: pd.DataFrame) -> str | None:
    if len(df) < max(EMA_SLOW, VOLUME_LOOKBACK) + 5:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    ema_fast = latest["ema_fast"]
    ema_slow = latest["ema_slow"]
    rsi = latest["rsi"]
    vol = latest["volume"]
    vol_avg = latest["vol_avg"]

    volume_ok = vol > (vol_avg * 0.75) if pd.notna(vol_avg) else True

    bullish_cross = prev["ema_fast"] <= prev["ema_slow"] and ema_fast > ema_slow
    buy_condition = (
        (ema_fast > ema_slow and rsi < RSI_BUY_MAX and volume_ok) or
        (bullish_cross and rsi < 48 and volume_ok)
    )

    bearish_cross = prev["ema_fast"] >= prev["ema_slow"] and ema_fast < ema_slow
    sell_condition = (
        (ema_fast < ema_slow and rsi > RSI_SELL_MIN and volume_ok) or
        (bearish_cross and rsi > 52 and volume_ok)
    )

    if buy_condition:
        return "BUY"
    if sell_condition:
        return "SELL"
    return None


def scan_coin(symbol: str, recommendations: list):
    """Scan one coin and collect buy/sell recommendations."""
    logger.info(f"Scanning {symbol}...")

    df = get_klines(symbol, limit=100)
    if df is None or len(df) < 40:
        logger.info(f"{symbol} → skipped (no data / not listed)")
        return

    df = calculate_indicators(df)
    latest = df.iloc[-1]

    price = latest["close"]
    ema_fast = latest["ema_fast"]
    ema_slow = latest["ema_slow"]
    rsi = latest["rsi"]

    signal = check_signals(df)

    if signal is None:
        logger.info(f"{symbol} → no clear setup (RSI {rsi:.1f})")
        return

    stop_loss = price * (1 - STOP_LOSS_PCT) if signal == "BUY" else price * (1 + STOP_LOSS_PCT)
    take_profit = price * (1 + TAKE_PROFIT_PCT) if signal == "BUY" else price * (1 - TAKE_PROFIT_PCT)

    reasoning = get_ai_reasoning(symbol, signal, price, ema_fast, ema_slow, rsi)

    rec = {
        "symbol": symbol,
        "side": signal,
        "price": price,
        "rsi": rsi,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasoning": reasoning,
    }
    recommendations.append(rec)

    # Individual alert
    emoji = "🟢" if signal == "BUY" else "🔴"
    action = "BUY" if signal == "BUY" else "SELL"

    message = (
        f"{emoji} <b>RECOMMENDATION: {action} {symbol}</b>\n\n"
        f"Current Price: <b>${price:.6f}</b>\n"
        f"Suggested Stop: ${stop_loss:.6f}\n"
        f"Suggested Target: ${take_profit:.6f}\n"
        f"RSI: {rsi:.1f} | EMA{EMA_FAST}/{EMA_SLOW}: ${ema_fast:.6f} / ${ema_slow:.6f}\n\n"
        f"🤖 <b>AI Reasoning:</b>\n{reasoning}\n\n"
        f"⚠️ This is only a signal – not financial advice. Do your own research."
    )

    if send_telegram(message):
        logger.info(f"✅ {action} recommendation sent for {symbol}")
    else:
        logger.warning(f"{action} recommendation generated but Telegram failed for {symbol}")


# ====================== MAIN ======================
def main():
    logger.info("=== Crypto Buy/Sell Recommendation Bot ===")
    logger.info(f"Scanning up to {len(COINS)} coins (majors + alts + meme)")

    recommendations = []

    for coin in COINS:
        try:
            scan_coin(coin, recommendations)
            time.sleep(0.3)
        except Exception as e:
            logger.error(f"Unexpected error on {coin}: {e}")

    # Final summary recommendation list
    buys = [r for r in recommendations if r["side"] == "BUY"]
    sells = [r for r in recommendations if r["side"] == "SELL"]

    logger.info(f"=== DONE | BUY: {len(buys)} | SELL: {len(sells)} ===")

    if not buys and not sells:
        send_telegram(
            "📊 <b>Scan Complete</b>\n\n"
            f"Scanned {len(COINS)} coins.\n"
            "No clear BUY or SELL recommendations right now."
        )
        return

    summary_lines = ["📋 <b>BUY / SELL RECOMMENDATIONS</b>\n"]

    if buys:
        summary_lines.append("<b>🟢 CONSIDER BUYING:</b>")
        for r in buys:
            summary_lines.append(
                f"• <b>{r['symbol']}</b> @ ${r['price']:.6f} "
                f"(RSI {r['rsi']:.0f}) → SL ${r['stop_loss']:.6f} | TP ${r['take_profit']:.6f}"
            )
        summary_lines.append("")

    if sells:
        summary_lines.append("<b>🔴 CONSIDER SELLING:</b>")
        for r in sells:
            summary_lines.append(
                f"• <b>{r['symbol']}</b> @ ${r['price']:.6f} "
                f"(RSI {r['rsi']:.0f})"
            )
        summary_lines.append("")

    summary_lines.append(
        f"Scanned: {len(COINS)} pairs | "
        f"Signals: {len(buys)} buy, {len(sells)} sell\n"
        "⚠️ Not financial advice."
    )

    send_telegram("\n".join(summary_lines))


if __name__ == "__main__":
    main()
