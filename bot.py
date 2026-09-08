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

# YOUR CURRENT HOLDINGS (update when you trade)
# Used so the bot prioritizes advice for coins you already own
PORTFOLIO = {
    "BTCUSDT": 0.00002404,
    "DOGEUSDT": 10.7122,
    # Meme coins (may not be on all exchanges – bot will skip if no data)
    "CHILLGUYUSDT": 10.8236,
    "MOGUSDT": 527837.4,
}

# Liquid coins to consider rotating INTO (small capital = prefer liquid pairs)
SCAN_COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "NEARUSDT", "SUIUSDT", "ARBUSDT", "OPUSDT",
    "SHIBUSDT", "PEPEUSDT", "WIFUSDT", "BONKUSDT", "FLOKIUSDT",
]

# Merge portfolio symbols into scan list
for sym in PORTFOLIO:
    if sym not in SCAN_COINS:
        SCAN_COINS.insert(0, sym)

STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
RSI_BUY_MAX = 38
RSI_SELL_MIN = 62
EMA_FAST = 9
EMA_SLOW = 21
VOLUME_LOOKBACK = 20
CANDLE_INTERVAL = "15m"  # more responsive than 1h for frequent checks

# Only spam Telegram when there is a real recommendation
ALWAYS_SEND_SUMMARY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

if not AI_API_KEY:
    logger.error("AI_API_KEY is missing")
    raise SystemExit("Missing AI_API_KEY")

client = genai.Client(api_key=AI_API_KEY)


def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing")
        return False
    if len(message) > 4000:
        message = message[:3990] + "\n...(truncated)"
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=12,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Telegram failed: {e}")
        return False


def get_klines(symbol: str, interval: str = CANDLE_INTERVAL, limit: int = 100):
    # Public market data (signals). CoinSwitch execution stays manual on the app.
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            # fallback Binance.US
            r = requests.get(
                f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
                timeout=10,
            )
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or isinstance(data, dict):
            return None
        df = pd.DataFrame(data)[[0, 1, 2, 3, 4, 5]]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for c in ["close", "high", "low", "volume"]:
            df[c] = df[c].astype(float)
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


def ai_note(symbol, side, price, ema_fast, ema_slow, rsi):
    prompt = (
        f"Act as a cautious crypto advisor for a trader with VERY SMALL capital. "
        f"{symbol} has a {side} signal on 15m. Price ${price:.6f}, "
        f"EMA9 ${ema_fast:.6f}, EMA21 ${ema_slow:.6f}, RSI {rsi:.1f}. "
        f"Give 2 short sentences. Mention fees risk for small size."
    )
    try:
        return client.models.generate_content(
            model="gemini-3.7-flash", contents=prompt
        ).text.strip()
    except Exception as e:
        logger.error(f"AI failed: {e}")
        return "AI note unavailable."


def scan_symbol(symbol: str):
    df = get_klines(symbol)
    if df is None or len(df) < 40:
        return None
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    signal = check_signals(df)
    if not signal:
        return {
            "symbol": symbol,
            "side": None,
            "price": float(latest["close"]),
            "rsi": float(latest["rsi"]),
            "in_portfolio": symbol in PORTFOLIO,
        }
    price = float(latest["close"])
    ema_fast = float(latest["ema_fast"])
    ema_slow = float(latest["ema_slow"])
    rsi = float(latest["rsi"])
    return {
        "symbol": symbol,
        "side": signal,
        "price": price,
        "rsi": rsi,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "stop": price * (1 - STOP_LOSS_PCT) if signal == "BUY" else price * (1 + STOP_LOSS_PCT),
        "target": price * (1 + TAKE_PROFIT_PCT) if signal == "BUY" else price * (1 - TAKE_PROFIT_PCT),
        "reasoning": ai_note(symbol, signal, price, ema_fast, ema_slow, rsi),
        "in_portfolio": symbol in PORTFOLIO,
        "qty": PORTFOLIO.get(symbol),
    }


def main():
    logger.info("=== Portfolio Buy/Sell Advisor (manual CoinSwitch) ===")
    results = []
    for sym in SCAN_COINS:
        try:
            row = scan_symbol(sym)
            if row:
                results.append(row)
                logger.info(
                    f"{sym}: side={row.get('side')} RSI={row.get('rsi', 0):.1f} "
                    f"portfolio={row.get('in_portfolio')}"
                )
            time.sleep(0.25)
        except Exception as e:
            logger.error(f"{sym} error: {e}")

    # Portfolio-focused advice
    portfolio_sells = [
        r for r in results if r.get("in_portfolio") and r.get("side") == "SELL"
    ]
    portfolio_holds = [
        r for r in results if r.get("in_portfolio") and r.get("side") != "SELL"
    ]
    buy_ideas = [r for r in results if r.get("side") == "BUY" and not r.get("in_portfolio")]
    # Prefer liquid majors for tiny capital
    buy_ideas = sorted(
        buy_ideas,
        key=lambda x: (
            0 if x["symbol"] in {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"} else 1,
            x["rsi"],
        ),
    )[:5]

    has_action = bool(portfolio_sells or buy_ideas)

    if not has_action and not ALWAYS_SEND_SUMMARY:
        logger.info("No actionable advice – silent (avoids spam every 5 min)")
        return

    lines = [
        "📌 <b>YOUR PORTFOLIO ADVICE</b>",
        "<i>Capital is small – fees matter. Trade manually on CoinSwitch.</i>",
        "",
        "<b>You hold:</b> BTC, DOGE, CHILLGUY, MOG",
        "",
    ]

    if portfolio_sells:
        lines.append("<b>🔴 CONSIDER SELLING (from your wallet):</b>")
        for r in portfolio_sells:
            lines.append(
                f"• <b>{r['symbol']}</b> qty {r.get('qty')} @ ${r['price']:.6f} "
                f"(RSI {r['rsi']:.0f})\n  {r.get('reasoning', '')[:180]}"
            )
        lines.append("")
    else:
        lines.append("<b>🔴 Sell from wallet:</b> No strong sell signal on tracked holdings right now.")
        lines.append("")

    if buy_ideas:
        lines.append("<b>🟢 CONSIDER BUYING (rotation ideas):</b>")
        for r in buy_ideas:
            lines.append(
                f"• <b>{r['symbol']}</b> @ ${r['price']:.6f} "
                f"| SL ${r['stop']:.6f} | TP ${r['target']:.6f} "
                f"(RSI {r['rsi']:.0f})"
            )
        lines.append("")
    else:
        lines.append("<b>🟢 Buy ideas:</b> No strong buy setups in liquid list right now.")
        lines.append("")

    if portfolio_holds:
        lines.append("<b>⏸ Holdings without sell signal:</b>")
        for r in portfolio_holds:
            if r.get("side") is None:
                lines.append(f"• {r['symbol']}: hold/watch (RSI {r['rsi']:.0f})")

    lines.append("")
    lines.append("⚠️ Not financial advice. Check CoinSwitch fees before every trade.")

    send_telegram("\n".join(lines))
    logger.info("Advice sent to Telegram")


if __name__ == "__main__":
    main()
