import os
import time
import math
import requests
from datetime import datetime, timezone

import pandas as pd

OKX_BASE = "https://www.okx.com"
SYMBOLS = ["BTC-USDT", "ETH-USDT"]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# ----------------- Yardımcılar ----------------- #

def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def jget_okx(path, params=None, retries=3, timeout=10):
    url = f"{OKX_BASE}{path}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == "0":
                    return data.get("data", [])
            time.sleep(1)
        except Exception as e:
            print(f"[HATA] OKX isteğinde hata: {e}")
            time.sleep(1)
    return []


def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("[UYARI] TELEGRAM_TOKEN veya CHAT_ID tanımlı değil. Mesaj gönderilmeyecek.")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print("[HATA] Telegram mesajı gönderilemedi:", r.text)
    except Exception as e:
        print("[HATA] Telegram isteğinde hata:", e)


# ----------------- Veri Çekme ----------------- #

def get_candles(inst_id: str, bar: str, limit: int = 200) -> pd.DataFrame:
    """
    OKX candles:
    [ ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm ]
    ts: ms, newest first
    """
    raw = jget_okx("/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": limit})
    if not raw:
        raise RuntimeError(f"{inst_id} için {bar} mum verisi alınamadı.")

    raw = list(reversed(raw))  # eskiden yeniye
    rows = []
    for row in raw:
        # row: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        ts_ms = int(row[0])
        rows.append({
            "ts": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })
    df = pd.DataFrame(rows)
    return df


def get_trades_whale(inst_id: str, limit: int = 200):
    """
    OKX trades:
    [ instId, tradeId, px, sz, side, ts ]
    """
    data = jget_okx("/api/v5/market/trades", {"instId": inst_id, "limit": limit})
    if not data:
        return {
            "net_delta": 0.0,
            "whale_category": "-",
            "whale_side_dir": None,
        }

    buy_val = 0.0
    sell_val = 0.0
    max_trade_val = 0.0
    max_trade_side = None

    for t in data:
        px = float(t[2])
        sz = float(t[3])
        side = t[4]  # buy / sell
        value = px * sz  # yaklaşık USDT

        if side == "buy":
            buy_val += value
        else:
            sell_val += value

        if value > max_trade_val:
            max_trade_val = value
            max_trade_side = side

    net_delta = buy_val - sell_val

    # Whale kategorisi
    if max_trade_val >= 1_000_000:
        cat = "XXL"
    elif max_trade_val >= 500_000:
        cat = "XL"
    elif max_trade_val >= 150_000:
        cat = "L"
    elif max_trade_val >= 50_000:
        cat = "M"
    else:
        cat = "-"

    if max_trade_side == "buy":
        whale_side_dir = "UP"
    elif max_trade_side == "sell":
        whale_side_dir = "DOWN"
    else:
        whale_side_dir = None

    return {
        "net_delta": net_delta,
        "whale_category": cat,
        "whale_side_dir": whale_side_dir,
    }


# ----------------- İndikatörler ----------------- #

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    # EMA'lar
    df["ema_fast"] = close.ewm(span=14, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=28, adjust=False).mean()

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Hacim oranı
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["v_ratio"] = df["volume"] / df["vol_sma20"]

    return df


def detect_swings(df: pd.DataFrame, lookback: int = 2) -> pd.DataFrame:
    df["swing_high"] = False
    df["swing_low"] = False

    for i in range(lookback, len(df) - lookback):
        hi = df.at[i, "high"]
        lo = df.at[i, "low"]

        is_high = True
        is_low = True

        for k in range(1, lookback + 1):
            if not (hi > df.at[i - k, "high"] and hi > df.at[i + k, "high"]):
                is_high = False
            if not (lo < df.at[i - k, "low"] and lo < df.at[i + k, "low"]):
                is_low = False

        if is_high:
            df.at[i, "swing_high"] = True
        if is_low:
            df.at[i, "swing_low"] = True

    return df


def get_structure_state(df: pd.DataFrame, idx: int):
    """Son yapıyı HH / HL / LH / LL olarak özetler."""
    highs = [i for i in range(idx + 1) if df.at[i, "swing_high"]]
    lows = [i for i in range(idx + 1) if df.at[i, "swing_low"]]

    last_high_idx = prev_high_idx = None
    last_low_idx = prev_low_idx = None

    high_type = None
    low_type = None

    if len(highs) >= 2:
        prev_high_idx = highs[-2]
        last_high_idx = highs[-1]
        if df.at[last_high_idx, "high"] > df.at[prev_high_idx, "high"]:
            high_type = "HH"
        else:
            high_type = "LH"

    if len(lows) >= 2:
        prev_low_idx = lows[-2]
        last_low_idx = lows[-1]
        if df.at[last_low_idx, "low"] > df.at[prev_low_idx, "low"]:
            low_type = "HL"
        else:
            low_type = "LL"

    # Yapı yönü
    struct_dir = "NEUTRAL"
    if high_type in ("HH",) or low_type in ("HL",):
        struct_dir = "UP"
    if high_type in ("LH",) or low_type in ("LL",):
        struct_dir = "DOWN"

    return {
        "struct_dir": struct_dir,
        "high_type": high_type,
        "low_type": low_type,
        "last_high_idx": last_high_idx,
        "last_low_idx": last_low_idx,
    }


def compute_trend_decision(df: pd.DataFrame, idx: int, whale_dir: str | None):
    """
    C seçeneği:
    - Market structure + EMA zorunlu
    - MACD veya whale onaylayıcı
    """
    st = get_structure_state(df, idx)
    struct_dir = st["struct_dir"]

    ema_dir = "UP" if df.at[idx, "ema_fast"] > df.at[idx, "ema_slow"] else "DOWN"
    macd_dir = "UP" if df.at[idx, "macd"] > 0 else "DOWN"

    confirmed_dir = None

    if struct_dir != "NEUTRAL" and struct_dir == ema_dir:
        match_count = 2  # struct + ema

        if macd_dir == struct_dir:
            match_count += 1
        if whale_dir == struct_dir:
            match_count += 1

        if match_count >= 3:
            confirmed_dir = struct_dir

    raw_dir = ema_dir  # always-in-market için yedek
    return {
        "raw_dir": raw_dir,
        "confirmed_dir": confirmed_dir,
        "struct": st,
        "ema_dir": ema_dir,
        "macd_dir": macd_dir,
    }


def analyze_symbol(inst_id: str):
    # 4H ve 1D verileri al
    df4 = get_candles(inst_id, "4H", limit=200)
    df4 = add_indicators(df4)
    df4 = detect_swings(df4)

    df1d = get_candles(inst_id, "1D", limit=120)
    df1d = add_indicators(df1d)
    df1d = detect_swings(df1d)

    # Whale ve hacim
    whale_info = get_trades_whale(inst_id, limit=200)
    net_delta = whale_info["net_delta"]
    whale_category = whale_info["whale_category"]
    whale_side_dir = whale_info["whale_side_dir"]

    # Whale yönünü biraz eşikleyelim
    whale_dir = None
    if abs(net_delta) > 100_000 and whale_side_dir is not None:
        whale_dir = whale_side_dir

    last_idx = len(df4) - 1
    prev_idx = len(df4) - 2

    trend_now = compute_trend_decision(df4, last_idx, whale_dir)
    trend_prev = compute_trend_decision(df4, prev_idx, None)

    # 1D trend
    last_1d_idx = len(df1d) - 1
    st_1d = get_structure_state(df1d, last_1d_idx)
    ema_dir_1d = "UP" if df1d.at[last_1d_idx, "ema_fast"] > df1d.at[last_1d_idx, "ema_slow"] else "DOWN"

    day_dir = "NEUTRAL"
    if st_1d["struct_dir"] == "UP" and ema_dir_1d == "UP":
        day_dir = "UP"
    elif st_1d["struct_dir"] == "DOWN" and ema_dir_1d == "DOWN":
        day_dir = "DOWN"

    # Hacim spike
    v_ratio_now = df4.at[last_idx, "v_ratio"] if not math.isnan(df4.at[last_idx, "v_ratio"]) else 1.0

    # Fiyatlar
    close_now = df4.at[last_idx, "close"]

    struct_now = trend_now["struct"]
    struct_prev = trend_prev["struct"]

    # HH/HL/LH/LL gösterimi
    high_type = struct_now["high_type"]
    low_type = struct_now["low_type"]

    # SL & TP hesapları için son swing
    last_low_idx = struct_now["last_low_idx"]
    last_high_idx = struct_now["last_high_idx"]

    swing_range = None
    last_low_price = None
    last_high_price = None

    if last_low_idx is not None and last_high_idx is not None:
        last_low_price = df4.at[last_low_idx, "low"]
        last_high_price = df4.at[last_high_idx, "high"]
        swing_range = abs(last_high_price - last_low_price)
    else:
        # yedek olsun
        swing_range = df4["high"].iloc[-20:].max() - df4["low"].iloc[-20:].min()

    return {
        "inst_id": inst_id,
        "df4": df4,
        "df1d": df1d,
        "trend_now": trend_now,
        "trend_prev": trend_prev,
        "day_dir": day_dir,
        "v_ratio_now": v_ratio_now,
        "net_delta": net_delta,
        "whale_category": whale_category,
        "whale_dir": whale_dir,
        "close_now": close_now,
        "last_low_price": last_low_price,
        "last_high_price": last_high_price,
        "swing_range": swing_range,
        "high_type": high_type,
        "low_type": low_type,
    }


def dir_to_text(direction: str) -> str:
    return "LONG" if direction == "UP" else "SHORT"


def dir_to_arrow(direction: str) -> str:
    return "🟢" if direction == "UP" else "🔴"


def signal_strength_text(trend_dir: str, day_dir: str) -> str:
    if day_dir == "NEUTRAL":
        return "Nötr Sinyal"
    if trend_dir == day_dir:
        return "Güçlü Sinyal"
    else:
        return "Zayıf Sinyal (Karşı Trend)"


def format_price(p: float, inst_id: str) -> str:
    # BTC/ETH için 2 ondalık yeterli
    if "BTC" in inst_id:
        return f"{p:,.2f}"
    if "ETH" in inst_id:
        return f"{p:,.2f}"
    return f"{p:,.4f}"


# ----------------- Ana Akış ----------------- #

def main():
    print(f"[INFO] Başladı: {ts()}")

    analyses = {}
    for inst in SYMBOLS:
        try:
            analyses[inst] = analyze_symbol(inst)
        except Exception as e:
            print(f"[HATA] {inst} analizinde hata: {e}")

    if not analyses:
        print("[HATA] Hiç enstrüman analiz edilemedi.")
        return

    # Trend değişimi var mı?
    any_trend_change = False
    cmd_lines = []
    detail_lines = []

    for inst in SYMBOLS:
        data = analyses.get(inst)
        if not data:
            continue

        inst_id = data["inst_id"]
        symbol_short = inst_id.split("-")[0]  # BTC / ETH

        now_dir = data["trend_now"]["confirmed_dir"]
        prev_dir = data["trend_prev"]["confirmed_dir"]

        day_dir = data["day_dir"]

        # Eğer confirmed yoksa, raw_dir kullanarak piyasayı yine yorumlayacağız ama komut vermeyeceğiz
        if now_dir is None:
            # Ayrı olarak uyarı için kullanabiliriz
            continue

        # Trend değişimi: önceki confirmed farklıysa
        trend_changed = (prev_dir is not None and prev_dir != now_dir) or (prev_dir is None)

        if trend_changed:
            any_trend_change = True
            side_text = dir_to_text(now_dir)
            arrow = dir_to_arrow(now_dir)
            strength = signal_strength_text(now_dir, day_dir)

            # SL & TP hesapla
            close_now = data["close_now"]
            swing_range = data["swing_range"] or 0.0
            last_low = data["last_low_price"]
            last_high = data["last_high_price"]

            if now_dir == "UP":
                # Long
                if last_low is not None:
                    sl = last_low
                else:
                    sl = close_now * 0.97  # yedek

                tp1 = close_now + swing_range * 0.5
                tp2 = close_now + swing_range * 1.0
                tp3 = close_now + swing_range * 1.5
            else:
                # Short
                if last_high is not None:
                    sl = last_high
                else:
                    sl = close_now * 1.03  # yedek

                tp1 = close_now - swing_range * 0.5
                tp2 = close_now - swing_range * 1.0
                tp3 = close_now - swing_range * 1.5

            cmd_lines.append(f"{arrow} {symbol_short} {side_text} AÇ ({strength})")

            # Detaylı bölüm
            tn = data["trend_now"]
            st = tn["struct"]
            v_ratio = data["v_ratio_now"]
            net_delta = data["net_delta"]
            whale_cat = data["whale_category"]
            whale_dir = data["whale_dir"]

            high_type = data["high_type"]
            low_type = data["low_type"]

            hhhl_text = []
            if high_type:
                color = "🟢" if high_type == "HH" else "🔴"
                hhhl_text.append(f"{color} {high_type}")
            if low_type:
                color = "🟢" if low_type == "HL" else "🔴"
                hhhl_text.append(f"{color} {low_type}")
            ms_line = " | ".join(hhhl_text) if hhhl_text else "-"

            whale_line = f"{whale_cat} / {net_delta:,.0f} USDT"
            if whale_dir == "UP":
                whale_line += " (Alım baskısı)"
            elif whale_dir == "DOWN":
                whale_line += " (Satış baskısı)"

            detail_lines.append(
                f"\n{symbol_short}:\n"
                f"- 4H Trend: {dir_to_text(now_dir)}\n"
                f"- Yapı: {ms_line}\n"
                f"- 4H Hacim Oranı (vRatio): {v_ratio:.2f}\n"
                f"- Whale: {whale_line}\n"
                f"- 1D Trend: {'LONG' if day_dir=='UP' else 'SHORT' if day_dir=='DOWN' else 'Nötr'}\n"
                f"- SL: {format_price(sl, inst_id)}\n"
                f"- TP1: {format_price(tp1, inst_id)}\n"
                f"- TP2: {format_price(tp2, inst_id)}\n"
                f"- TP3: {format_price(tp3, inst_id)}\n"
            )

    if any_trend_change:
        header = "⚠️ TREND DEĞİŞİMİ — 4H KAPANIŞI\n\n"
        text = header + "\n".join(cmd_lines) + "\n" + "".join(detail_lines)
        send_telegram(text)
        print("[INFO] Trend değişimi mesajı gönderildi.")
        return

    # Trend değişimi yoksa: Önemli uyarı var mı?
    # (Yeni HH/LL + yüksek hacim veya büyük whale)
    warning_lines = []
    for inst in SYMBOLS:
        data = analyses.get(inst)
        if not data:
            continue

        inst_id = data["inst_id"]
        symbol_short = inst_id.split("-")[0]

        v_ratio = data["v_ratio_now"]
        net_delta = data["net_delta"]
        whale_cat = data["whale_category"]
        whale_dir = data["whale_dir"]
        high_type = data["high_type"]
        low_type = data["low_type"]

        important_struct = high_type in ("HH", "LL") or low_type in ("HL", "LL")
        big_volume = v_ratio >= 3.0
        big_whale = whale_cat in ("L", "XL", "XXL")

        if important_struct and (big_volume or big_whale):
            struct_parts = []
            if high_type:
                color = "🟢" if high_type == "HH" else "🔴"
                struct_parts.append(f"{color} {high_type}")
            if low_type:
                color = "🟢" if low_type == "HL" else "🔴"
                struct_parts.append(f"{color} {low_type}")
            struct_line = " | ".join(struct_parts)

            whale_line = f"{whale_cat} / {net_delta:,.0f} USDT"
            if whale_dir == "UP":
                whale_line += " (Alım baskısı)"
            elif whale_dir == "DOWN":
                whale_line += " (Satış baskısı)"

            warning_lines.append(
                f"{symbol_short}:\n"
                f"- Yapı: {struct_line}\n"
                f"- vRatio: {v_ratio:.2f}\n"
                f"- Whale: {whale_line}\n"
            )

    if warning_lines:
        text = "❗ ÖNEMLİ UYARI — 4H\n\n" + "\n".join(warning_lines)
        send_telegram(text)
        print("[INFO] Uyarı mesajı gönderildi.")
    else:
        print("[INFO] Gönderilecek trend değişimi veya önemli uyarı yok.")


if __name__ == "__main__":
    main()
