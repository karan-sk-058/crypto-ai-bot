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

# Expanded coin list
COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "MATICUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT", "UNIUSDT"
]

# Risk settings
STOP_LOSS_PCT = 0.015     # 1.5%
TAKE_PROFIT_PCT = 0.03    # 3%

# Strategy settings
RSI_BUY_MAX = 35
RSI_SELL_MIN = 65
EMA_FAST = 9
EMA_SLOW = 21
VOLUME_LOOKBACK = 20

# Set to True to run a simple historical backtest instead of live scan
BACKTEST_MODE = True
BACKTEST_LIMIT = 200      # number of candles for backtest

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
    """Send message to Telegram. Returns True if successful."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing – alert not sent")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            },
            timeout=10
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def get_klines(symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame | None:
    """Fetch klines from Binance.US and return a clean DataFrame."""
    url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data or isinstance(data, dict):
            logger.warning(f"No valid data for {symbol}")
            return None

        df = pd.DataFrame(data)
        df = df[[0, 1, 2, 3, 4, 5]]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA fast/slow, RSI, and volume average."""
    df = df.copy()

    # EMAs
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume filter
    df["vol_avg"] = df["volume"].rolling(VOLUME_LOOKBACK).mean()

    return df


def get_ai_reasoning(symbol: str, side: str, price: float, ema_fast: float, ema_slow: float, rsi: float) -> str:
    """Ask Gemini for a short trade reasoning."""
    if client is None:
        return "AI disabled."

    prompt = (
        f"Act as a quantitative crypto trader. "
        f"{symbol} shows a {side} signal on the 1-hour chart. "
        f"Price: ${price:.4f}, EMA{EMA_FAST}: ${ema_fast:.4f}, EMA{EMA_SLOW}: ${ema_slow:.4f}, RSI: {rsi:.1f}. "
        f"Give a concise 2-sentence reasoning. Be realistic and mention key risks."
    )
    try:
        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"AI reasoning failed: {e}")
        return "AI reasoning unavailable."


def check_signals(df: pd.DataFrame) -> str | None:
    """
    Returns 'BUY', 'SELL', or None based on improved strategy:
    - BUY: Fast EMA > Slow EMA + RSI oversold + volume above average
    - SELL: Fast EMA < Slow EMA + RSI overbought + volume above average
    """
    if len(df) < max(EMA_SLOW, VOLUME_LOOKBACK) + 5:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = latest["close"]
    ema_fast = latest["ema_fast"]
    ema_slow = latest["ema_slow"]
    rsi = latest["rsi"]
    vol = latest["volume"]
    vol_avg = latest["vol_avg"]

    # Volume confirmation
    volume_ok = vol > (vol_avg * 0.8) if pd.notna(vol_avg) else True

    # Bullish: fast EMA crossed above slow + RSI not overbought
    bullish_cross = prev["ema_fast"] <= prev["ema_slow"] and ema_fast > ema_slow
    buy_condition = (
        (ema_fast > ema_slow) and
        (rsi < RSI_BUY_MAX) and
        volume_ok
    ) or (bullish_cross and rsi < 50 and volume_ok)

    # Bearish: fast EMA crossed below slow + RSI not oversold
    bearish_cross = prev["ema_fast"] >= prev["ema_slow"] and ema_fast < ema_slow
    sell_condition = (
        (ema_fast < ema_slow) and
        (rsi > RSI_SELL_MIN) and
        volume_ok
    ) or (bearish_cross and rsi > 50 and volume_ok)

    if buy_condition:
        return "BUY"
    if sell_condition:
        return "SELL"
    return None


def simple_backtest(symbol: str, df: pd.DataFrame) -> dict:
    """Very simple backtest: count signals and rough win rate assumption."""
    signals = []
    for i in range(max(EMA_SLOW, VOLUME_LOOKBACK) + 5, len(df)):
        window = df.iloc[:i+1]
        signal = check_signals(window)
        if signal:
            signals.append({
                "index": i,
                "signal": signal,
                "price": df.iloc[i]["close"]
            })

    buys = [s for s in signals if s["signal"] == "BUY"]
    sells = [s for s in signals if s["signal"] == "SELL"]

    return {
        "symbol": symbol,
        "total_signals": len(signals),
        "buys": len(buys),
        "sells": len(sells),
        "last_price": df.iloc[-1]["close"]
    }


def scan_coin(symbol: str, stats: dict):
    """Scan one coin and send alert if conditions are met."""
    logger.info(f"Scanning {symbol}...")

    limit = BACKTEST_LIMIT if BACKTEST_MODE else 100
    df = get_klines(symbol, limit=limit)
    if df is None or len(df) < 40:
        logger.warning(f"Not enough data for {symbol}")
        return

    df = calculate_indicators(df)

    if BACKTEST_MODE:
        result = simple_backtest(symbol, df)
        logger.info(
            f"[BACKTEST] {symbol} → Signals: {result['total_signals']} "
            f"(Buys: {result['buys']}, Sells: {result['sells']})"
        )
        stats["backtest"].append(result)
        return

    latest = df.iloc[-1]
    price = latest["close"]
    ema_fast = latest["ema_fast"]
    ema_slow = latest["ema_slow"]
    rsi = latest["rsi"]

    logger.info(
        f"{symbol} → Price: ${price:.4f} | EMA{EMA_FAST}: ${ema_fast:.4f} | "
        f"EMA{EMA_SLOW}: ${ema_slow:.4f} | RSI: {rsi:.1f}"
    )

    signal = check_signals(df)

    if signal == "BUY":
        stats["buys"] += 1
        stop_loss = price * (1 - STOP_LOSS_PCT)
        take_profit = price * (1 + TAKE_PROFIT_PCT)

        reasoning = get_ai_reasoning(symbol, "BUY", price, ema_fast, ema_slow, rsi)

        message = (
            f"🟢 <b>{symbol} BUY SIGNAL</b>\n\n"
            f"Entry: <b>${price:.4f}</b>\n"
            f"Stop Loss ({STOP_LOSS_PCT*100:.1f}%): ${stop_loss:.4f}\n"
            f"Take Profit ({TAKE_PROFIT_PCT*100:.1f}%): ${take_profit:.4f}\n"
            f"RSI: {rsi:.1f} | EMA{EMA_FAST}/{EMA_SLOW}: ${ema_fast:.4f} / ${ema_slow:.4f}\n\n"
            f"🤖 <b>AI Reasoning:</b>\n{reasoning}\n\n"
            f"⚠️ Signal only – not financial advice."
        )

        if send_telegram(message):
            logger.info(f"✅ BUY alert sent for {symbol}")
        else:
            logger.warning(f"BUY signal generated but Telegram failed for {symbol}")

    elif signal == "SELL":
        stats["sells"] += 1
        reasoning = get_ai_reasoning(symbol, "SELL", price, ema_fast, ema_slow, rsi)

        message = (
            f"🔴 <b>{symbol} SELL SIGNAL</b>\n\n"
            f"Price: <b>${price:.4f}</b>\n"
            f"RSI: {rsi:.1f} | EMA{EMA_FAST}/{EMA_SLOW}: ${ema_fast:.4f} / ${ema_slow:.4f}\n\n"
            f"🤖 <b>AI Reasoning:</b>\n{reasoning}\n\n"
            f"⚠️ Signal only – not financial advice."
        )

        if send_telegram(message):
            logger.info(f"✅ SELL alert sent for {symbol}")
        else:
            logger.warning(f"SELL signal generated but Telegram failed for {symbol}")

    else:
        logger.info(f"{symbol} → No clear setup")


# ====================== MAIN ======================
def main():
    mode = "BACKTEST" if BACKTEST_MODE else "LIVE SCAN"
    logger.info(f"=== Gemini AI Crypto Bot ({mode}) ===")
    logger.info(f"Coins: {len(COINS)} | Strategy: EMA{EMA_FAST}/{EMA_SLOW} + RSI + Volume")

    stats = {"buys": 0, "sells": 0, "backtest": []}

    for coin in COINS:
        try:
            scan_coin(coin, stats)
            time.sleep(0.35)
        except Exception as e:
            logger.error(f"Unexpected error on {coin}: {e}")

    # Summary
    if BACKTEST_MODE:
        total_signals = sum(r["total_signals"] for r in stats["backtest"])
        logger.info(f"=== BACKTEST COMPLETE | Total signals found: {total_signals} ===")
    else:
        logger.info(f"=== SCAN COMPLETE | Buys: {stats['buys']} | Sells: {stats['sells']} ===")

        if stats["buys"] or stats["sells"]:
            summary = (
                f"📊 <b>Scan Summary</b>\n"
                f"Buys: {stats['buys']} | Sells: {stats['sells']}\n"
                f"Coins scanned: {len(COINS)}"
            )
            send_telegram(summary)


if __name__ == "__main__":
    main()
