# main.py
import requests
import time
import datetime
import numpy as np
import pandas as pd
from statistics import mean

TOKEN = "<TELEGRAM_BOT_TOKEN>"
CHAT_ID = "<TELEGRAM_CHAT_ID>"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "4h"
LIMIT = 100
API_URL = "https://api.binance.com/api/v3/klines"


def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg}
    try:
        requests.post(url, data=data)
    except:
        pass


def fetch_ohlcv(symbol):
    url = f"{API_URL}?symbol={symbol}&interval={INTERVAL}&limit={LIMIT}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "qv", "trades", "taker_base_vol", "taker_quote_vol", "ignore"])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
        df["close"] = df["close"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    else:
        return None


def detect_peaks(df):
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    peaks = []
    bottoms = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
            peaks.append((df["timestamp"].iloc[i], highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
            bottoms.append((df["timestamp"].iloc[i], lows[i]))
    return peaks, bottoms


def compute_rsi(closes, period=14):
    deltas = np.diff(closes)
    ups = deltas.clip(min=0)
    downs = -deltas.clip(max=0)
    ma_up = pd.Series(ups).rolling(window=period).mean()
    ma_down = pd.Series(downs).rolling(window=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def check_rsi_divergence(df):
    closes = df["close"].values
    rsi = compute_rsi(closes)
    if len(rsi.dropna()) < 5:
        return None
    recent_price = closes[-1]
    recent_rsi = rsi.iloc[-1]
    past_price = closes[-5]
    past_rsi = rsi.iloc[-5]

    if recent_price > past_price and recent_rsi < past_rsi:
        return "Bearish RSI Divergence detected"
    elif recent_price < past_price and recent_rsi > past_rsi:
        return "Bullish RSI Divergence detected"
    return None


def analyze(symbol):
    df = fetch_ohlcv(symbol)
    if df is None:
        send_telegram(f"{symbol} verisi alınamadı.")
        return

    peaks, bottoms = detect_peaks(df)
    message = f"\n\n📈 {symbol} 4H Teknik Analiz"
    if peaks:
        message += f"\nSon tepe: {peaks[-1][1]:.2f} ({peaks[-1][0]})"
    if bottoms:
        message += f"\nSon dip: {bottoms[-1][1]:.2f} ({bottoms[-1][0]})"

    rsi_msg = check_rsi_divergence(df)
    if rsi_msg:
        message += f"\n⚠️ {rsi_msg}"

    send_telegram(message)


if __name__ == "__main__":
    for sym in SYMBOLS:
        print(f"[DEBUG] {sym} işleniyor...")
        analyze(sym)
        time.sleep(3)
