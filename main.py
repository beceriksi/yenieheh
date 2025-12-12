import requests
import time
import datetime
import numpy as np
import talib
import os

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

SYMBOLS = ["BTC-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT", "SOL-USDT"]
INTERVAL = 14400  # 4 saat
LIMIT = 200

OKX_ENDPOINT = "https://www.okx.com"


def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=payload)


def fetch_ohlcv(symbol):
    url = f"{OKX_ENDPOINT}/api/v5/market/candles"
    params = {"instId": symbol, "bar": "4H", "limit": str(LIMIT)}
    response = requests.get(url, params=params)
    try:
        data = response.json()["data"]
        closes = [float(c[4]) for c in reversed(data)]
        highs = [float(c[2]) for c in reversed(data)]
        lows = [float(c[3]) for c in reversed(data)]
        volumes = [float(c[5]) for c in reversed(data)]
        return closes, highs, lows, volumes
    except:
        print(f"[HATA] {symbol} verisi çekilemedi.")
        return [], [], [], []


def detect_trend_reversals(closes):
    pivots = []
    for i in range(2, len(closes)-2):
        if closes[i] > closes[i-1] and closes[i] > closes[i+1]:
            pivots.append((i, 'tepe'))
        elif closes[i] < closes[i-1] and closes[i] < closes[i+1]:
            pivots.append((i, 'dip'))
    return pivots


def detect_rsi_divergence(closes):
    rsi = talib.RSI(np.array(closes), timeperiod=14)
    if len(rsi) < 20:
        return None
    if closes[-1] > closes[-2] and rsi[-1] < rsi[-2]:
        return "Negatif RSI uyumsuzluğu (Düşüş sinyali)"
    elif closes[-1] < closes[-2] and rsi[-1] > rsi[-2]:
        return "Pozitif RSI uyumsuzluğu (Yükseliş sinyali)"
    return None


def detect_macd_signal(closes):
    macd, signal, _ = talib.MACD(np.array(closes), fastperiod=12, slowperiod=26, signalperiod=9)
    if macd[-1] > signal[-1] and macd[-2] < signal[-2]:
        return "MACD Golden Cross (Yükseliş sinyali)"
    elif macd[-1] < signal[-1] and macd[-2] > signal[-2]:
        return "MACD Death Cross (Düşüş sinyali)"
    return None


def is_volume_supporting(volumes):
    recent = np.mean(volumes[-5:])
    past = np.mean(volumes[-15:-5])
    return recent > past * 1.2


def analyze_symbol(symbol):
    closes, highs, lows, volumes = fetch_ohlcv(symbol)
    if len(closes) < 20:
        return

    trend_points = detect_trend_reversals(closes)
    rsi_divergence = detect_rsi_divergence(closes)
    macd_signal = detect_macd_signal(closes)
    volume_ok = is_volume_supporting(volumes)

    message = f"\n\n--- {symbol} 4H Analiz ---"
    last_pivot = trend_points[-1] if trend_points else None

    if last_pivot:
        tip = "Tepe" if last_pivot[1] == 'tepe' else "Dip"
        message += f"\nTrend dönüş noktası: {tip}"

    if rsi_divergence:
        message += f"\n{rsi_divergence}"

    if macd_signal:
        message += f"\n{macd_signal}"

    if volume_ok:
        message += f"\nHacim desteği mevcut."
    else:
        message += f"\nHacim düşük."

    if rsi_divergence or macd_signal:
        send_telegram_message(message)


def main():
    while True:
        for symbol in SYMBOLS:
            analyze_symbol(symbol)
        time.sleep(14400)  # 4 saat bekle


if __name__ == "__main__":
    main()
