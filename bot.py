import os
import requests
import pandas as pd
from google import genai

# Python will now securely read the keys from GitHub's hidden environment storage
AI_API_KEY = os.environ.get("AI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

client = genai.Client(api_key=AI_API_KEY)

# 3. EXPANDED 10-COIN PORTFOLIO
coins = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT'
]

# 4. TELEGRAM ALERT FUNCTION
def send_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # We use a standard web request to push the text directly to your phone
    requests.get(url, params={"chat_id": TELEGRAM_CHAT_ID, "text": message})

print("--- 🤖 GEMINI AI 10-COIN SCANNER ---")

for coin in coins:
    print(f"\nScanning {coin}...")
    url = f"https://api.binance.us/api/v3/klines?symbol={coin}&interval=1h&limit=100"
    
    try:
        response = requests.get(url)
        candles = response.json()
        
        # Digital Spreadsheet
        df = pd.DataFrame(candles)[[0, 4]]
        df.columns = ['Date', 'Close']
        df['Close'] = df['Close'].astype(float)
        
        # Trend & Momentum Math
        df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        df['RSI_14'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        
        latest = df.iloc[-1]
        close_price = latest['Close']
        ema = latest['EMA_14']
        rsi = latest['RSI_14']
        
        print(f"Metrics -> Price: ${close_price:.2f} | EMA: ${ema:.2f} | RSI: {rsi:.2f}")
        
        # OPPORTUNITY FILTER
        if close_price > ema and rsi < 30:
            print("⚠️ BUY SIGNAL DETECTED! Calculating Risk & Notifying AI...")
            
            # THE COMMERCE LOGIC: 1% Max Loss, 2% Target Profit
            stop_loss = close_price * 0.99
            take_profit = close_price * 1.02
            
            # Querying the AI for context
            prompt = (f"Act as a quantitative trader. {coin} shows a BUY signal on the 1-hour chart. "
                      f"Price: ${close_price:.2f}, EMA: ${ema:.2f}, RSI: {rsi:.2f}. "
                      f"Provide a concise 2-sentence trade reasoning.")
            
            ai_response = client.models.generate_content(
                model='gemini-3.7-flash',
                contents=prompt
            )
            
            # FORMAT AND SEND THE NOTIFICATION TO YOUR PHONE
            alert_message = (
                f"🟢 {coin} BUY OPPORTUNITY!\n\n"
                f"Entry Price: ${close_price:.2f}\n"
                f"Stop Loss (1%): ${stop_loss:.2f}\n"
                f"Take Profit (2%): ${take_profit:.2f}\n\n"
                f"🤖 AI Reasoning:\n{ai_response.text}"
            )
            print("--- 🤖 GEMINI AI 10-COIN SCANNER ---")

# --- ADD THIS TEMPORARY LINE FOR TESTING ---
send_alert("🚨 TEST ALERT: Your cloud trading bot is live and scanning markets!")
# -------------------------------------------

for coin in coins:
    # (rest of your code...)
            send_alert(alert_message)
            print("✅ Alert sent directly to your phone!")
            
        elif close_price < ema and rsi > 70:
            print("⚪ SELL SIGNAL DETECTED! (Skipping alerts as we focus on buy setups)")
            
        else:
            print("⚪ NO TRADE. Metrics are neutral.")
            
    except Exception as e:
        print(f"❌ Error scanning {coin}. It may be temporarily unavailable.")

print("\n--- 10-COIN SCAN COMPLETE ---")
