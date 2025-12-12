import requests
import pandas as pd
import numpy as np
import time
import datetime
import os
import telegram
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator

# Telegram
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = telegram.Bot(token=TOKEN)

# Ayarlar
SYMBOLS = ["BTC-USDT", "ETH-USDT", "BNB-USDT"]
INTERVAL = "4H"
API_URL = "https://www.okx.com"
LIMIT = 200


def fetch_ohlcv(symbol):
    url = f"{API_URL}/api/v5/market/candles?instId={symbol}&bar={INTERVAL}&limit={LIMIT}"
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.DataFrame(response.json()["data"], columns=[
            "timestamp", "open", "high", "low", "close", "volume", "quoteVol"])
        df = df.iloc[::-1]  # ters çevir
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    else:
        print(f"[HATA] {symbol} verisi alınamadı")
        return None


def detect_local_extrema(df, order=2):
    df["is_top"] = (df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))
    df["is_bottom"] = (df["low"] < df["low"].shift(1)) & (df["low"] < df["low"].shift(-1))
    return df


def analyze(df, symbol):
    # Göstergeler
    macd = MACD(df["close"])
    rsi = RSIIndicator(df["close"])
    ema200 = EMAIndicator(df["close"], window=200)

    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["rsi"] = rsi.rsi()
    df["ema200"] = ema200.ema_indicator()

    df = detect_local_extrema(df)

    son = df.iloc[-1]
    onceki = df.iloc[-2]

    mesajlar = []

    if son["macd"] > son["macd_signal"] and son["close"] > son["ema200"] and son["rsi"] < 70:
        mesajlar.append(f"{symbol} ↗ LONG sinyali (MACD kesişimi, EMA200 üstü, RSI={son['rsi']:.2f})")

    if son["macd"] < son["macd_signal"] and son["close"] < son["ema200"] and son["rsi"] > 30:
        mesajlar.append(f"{symbol} ↘ SHORT sinyali (MACD kesişimi, EMA200 altı, RSI={son['rsi']:.2f})")

    if onceki["is_top"] and son["close"] < onceki["low"]:
        mesajlar.append(f"{symbol} potansiyel TEPE dönüşü tespit edildi.")

    if onceki["is_bottom"] and son["close"] > onceki["high"]:
        mesajlar.append(f"{symbol} potansiyel DİP dönüşü tespit edildi.")

    return mesajlar


def main():
    for symbol in SYMBOLS:
        print(f"[DEBUG] {symbol} verisi çekiliyor...")
        df = fetch_ohlcv(symbol)
        if df is not None:
            mesajlar = analyze(df, symbol)
            for mesaj in mesajlar:
                print("[MESAJ]", mesaj)
                bot.send_message(chat_id=CHAT_ID, text=mesaj)
        time.sleep(3)


if __name__ == "__main__":
    main()
