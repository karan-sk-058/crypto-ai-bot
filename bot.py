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

# Coins to scan
COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"
]

# Risk settings
STOP_LOSS_PCT = 0.01      # 1%
TAKE_PROFIT_PCT = 0.02    # 2%

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ====================== CLIENTS ======================
if not AI_API_KEY:
    logger.error("AI_API_KEY is missing")
    raise SystemExit("Missing AI_API_KEY")

client = genai.Client(api_key=AI_API_KEY)

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
        df = df[[0, 1, 2, 3, 4, 5]]  # time, open, high, low, close, volume
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA and RSI to the dataframe."""
    df = df.copy()
    df["ema_14"] = df["close"].ewm(span=14, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def get_ai_reasoning(symbol: str, price: float, ema: float, rsi: float) -> str:
    """Ask Gemini for a short trade reasoning."""
    prompt = (
        f"Act as a quantitative crypto trader. "
        f"{symbol} shows a BUY signal on the 1-hour chart. "
        f"Price: ${price:.4f}, EMA14: ${ema:.4f}, RSI14: {rsi:.1f}. "
        f"Give a concise 2-sentence reasoning for this potential long setup. "
        f"Be realistic and mention key risks."
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


def scan_coin(symbol: str):
    """Scan one coin and send alert if buy conditions are met."""
    logger.info(f"Scanning {symbol}...")

    df = get_klines(symbol)
    if df is None or len(df) < 30:
        logger.warning(f"Not enough data for {symbol}")
        return

    df = calculate_indicators(df)
    latest = df.iloc[-1]

    price = latest["close"]
    ema = latest["ema_14"]
    rsi = latest["rsi_14"]
    volume = latest["volume"]

    logger.info(f"{symbol} → Price: ${price:.4f} | EMA: ${ema:.4f} | RSI: {rsi:.1f}")

    # === BUY CONDITIONS ===
    # Price above EMA + RSI oversold + some volume
    if price > ema and rsi < 32 and volume > 0:
        stop_loss = price * (1 - STOP_LOSS_PCT)
        take_profit = price * (1 + TAKE_PROFIT_PCT)

        reasoning = get_ai_reasoning(symbol, price, ema, rsi)

        message = (
            f"🟢 <b>{symbol} BUY SIGNAL</b>\n\n"
            f"Entry: <b>${price:.4f}</b>\n"
            f"Stop Loss (1%): ${stop_loss:.4f}\n"
            f"Take Profit (2%): ${take_profit:.4f}\n"
            f"RSI: {rsi:.1f} | EMA14: ${ema:.4f}\n\n"
            f"🤖 <b>AI Reasoning:</b>\n{reasoning}\n\n"
            f"⚠️ This is only a signal – not financial advice."
        )

        if send_telegram(message):
            logger.info(f"✅ Alert sent for {symbol}")
        else:
            logger.warning(f"Alert generated but Telegram failed for {symbol}")

    elif price < ema and rsi > 68:
        logger.info(f"{symbol} → SELL zone (not alerting)")
    else:
        logger.info(f"{symbol} → No clear setup")


# ====================== MAIN ======================
def main():
    logger.info("=== Gemini AI Crypto Scanner Started ===")
    logger.info(f"Scanning {len(COINS)} coins...")

    for coin in COINS:
        try:
            scan_coin(coin)
            time.sleep(0.4)  # small delay to be nice to the API
        except Exception as e:
            logger.error(f"Unexpected error on {coin}: {e}")

    logger.info("=== Scan complete ===")


if __name__ == "__main__":
    main()
