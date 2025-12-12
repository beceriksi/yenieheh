import requests
import time
import numpy as np
import pandas as pd
from datetime import datetime

# OKX API endpoint
BASE_URL = "https://www.okx.com"
CANDLES_ENDPOINT = "/api/v5/market/candles"

# Telegram
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

COINS = ["BTC-USDT", "ETH-USDT"]  # genişletilebilir
INTERVAL = "4H"


def get_klines(symbol):
    url = BASE_URL + CANDLES_ENDPOINT
    params = {
        "instId": symbol,
        "bar": INTERVAL,
        "limit": 100
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        df = pd.DataFrame(data['data'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', '_1', '_2', '_3', '_4', '_5'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['rsi'] = compute_rsi(df['close'], 14)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"[HATA] {symbol} veri çekilemedi: {e}")
        return None


def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def detect_peaks(df):
    peaks, troughs = [], []
    for i in range(2, len(df) - 2):
        if df['high'][i] > df['high'][i - 1] and df['high'][i] > df['high'][i + 1]:
            peaks.append((df['datetime'][i], df['high'][i]))
        if df['low'][i] < df['low'][i - 1] and df['low'][i] < df['low'][i + 1]:
            troughs.append((df['datetime'][i], df['low'][i]))
    return peaks, troughs


def detect_rsi_divergence(df):
    latest_rsi = df['rsi'].iloc[-1]
    prev_rsi = df['rsi'].iloc[-5]
    if latest_rsi > 70 and latest_rsi < prev_rsi:
        return "Negative RSI divergence (olası düşüş)"
    elif latest_rsi < 30 and latest_rsi > prev_rsi:
        return "Positive RSI divergence (olası yükseliş)"
    return None


def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg
    }
    try:
        r = requests.post(url, data=payload)
        if not r.ok:
            print("[Telegram] Mesaj gönderilemedi.")
    except Exception as e:
        print(f"[Telegram] Hata: {e}")


def main():
    for coin in COINS:
        df = get_klines(coin)
        if df is None or df.empty:
            continue
        peaks, troughs = detect_peaks(df)
        signal = detect_rsi_divergence(df)

        msg = f"\n[{coin}]\nSon 4 saatlik analiz:"
        if peaks:
            msg += f"\nSon Tepe: {peaks[-1][1]} @ {peaks[-1][0]}"
        if troughs:
            msg += f"\nSon Dip: {troughs[-1][1]} @ {troughs[-1][0]}"
        if signal:
            msg += f"\nRSI Sinyali: {signal}"

        send_telegram(msg)


if __name__ == "__main__":
    main()
