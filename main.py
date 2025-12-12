import requests
import pandas as pd
import time
import datetime
import os
import numpy as np
import ta

# Telegram
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"[HATA] Telegram mesaj gönderilemedi: {e}")

# OKX API
BASE_URL = "https://www.okx.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_ohlcv(symbol, interval="4H", limit=100):
    url = f"{BASE_URL}/api/v5/market/candles"
    params = {"instId": symbol, "bar": interval, "limit": str(limit)}
    try:
        response = requests.get(url, params=params, headers=HEADERS)
        data = response.json()
        if "data" in data:
            df = pd.DataFrame(data["data"], columns=["timestamp", "open", "high", "low", "close", "volume", "volCcy"])
            df = df.iloc[::-1].reset_index(drop=True)
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df
        else:
            print(f"[HATA] API yanıtı: {data}")
            return None
    except Exception as e:
        print(f"[HATA] Veri çekilemedi: {e}")
        return None

def detect_divergence(df):
    rsi = ta.momentum.RSIIndicator(close=df['close']).rsi()
    df['rsi'] = rsi
    df['peak'] = (df['high'].shift(1) < df['high']) & (df['high'] > df['high'].shift(-1))
    df['bottom'] = (df['low'].shift(1) > df['low']) & (df['low'] < df['low'].shift(-1))

    alerts = []

    for i in range(2, len(df)):
        if df['peak'][i]:
            price_diff = df['high'][i] - df['high'][i-2]
            rsi_diff = df['rsi'][i] - df['rsi'][i-2]
            if price_diff > 0 and rsi_diff < 0:
                alerts.append((df['timestamp'][i], "Negatif RSI uyumsuzluğu: POTANSİYEL TEPE"))

        if df['bottom'][i]:
            price_diff = df['low'][i] - df['low'][i-2]
            rsi_diff = df['rsi'][i] - df['rsi'][i-2]
            if price_diff < 0 and rsi_diff > 0:
                alerts.append((df['timestamp'][i], "Pozitif RSI uyumsuzluğu: POTANSİYEL DİP"))

    return alerts

def main():
    symbols = ["BTC-USDT", "ETH-USDT"]

    for sym in symbols:
        print(f"[DEBUG] {sym} verisi çekiliyor...")
        df = get_ohlcv(sym)
        if df is not None and not df.empty:
            alerts = detect_divergence(df)
            for t, msg in alerts[-2:]:
                dt = datetime.datetime.utcfromtimestamp(int(t.split("T")[0]))
                send_telegram_message(f"{sym}: {msg} | Tarih: {dt}")
        else:
            print(f"[HATA] {sym} için veri alınamadı")

if __name__ == "__main__":
    main()
