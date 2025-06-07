from fastapi import APIRouter
import gspread
import pandas as pd
import json
import os
import time
import yfinance as yf  # Required for fetching live prices

router = APIRouter()



SHEET_ID = "1GLw524fGrKtsVByxV-wZ9NwuBNxeDBTzBqH3nx6xNtA"

# In-memory cache
_cached_result = None
_cache_expiry = 0

@router.get("/trending-tickers")
def get_trending_tickers():
    global _cached_result, _cache_expiry

    if time.time() < _cache_expiry and _cached_result:
        return _cached_result

    try:
        creds_dict = json.loads(os.environ["GOOGLE_SHEETS_CREDS_JSON_APP"])
        gc = gspread.service_account_from_dict(creds_dict)
        sh = gc.open_by_key(SHEET_ID)
        worksheet = sh.sheet1
        data = worksheet.get_all_records()

        df = pd.DataFrame(data)

        if "Timestamp" not in df.columns or "Event" not in df.columns or "Details" not in df.columns:
            return {"error": "Missing required columns"}

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
        df = df[df["Timestamp"] >= pd.Timestamp.now() - pd.Timedelta(days=14)]
        calc_df = df[df["Event"] == "Calculator Submitted"]

        parsed = []
        for row in calc_df["Details"]:
            try:
                parsed_row = json.loads(row)
                if "ticker" in parsed_row and "targetPrice" in parsed_row:
                    parsed.append(parsed_row)
            except Exception:
                continue

        if not parsed:
            return {"error": "No valid rows with ticker and targetPrice"}

        parsed_df = pd.DataFrame(parsed)

        trending = (
            parsed_df.groupby("ticker")["targetPrice"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values(by="count", ascending=False)
            .head(10)
        )

        # Fetch current prices for the top 10 tickers
        current_prices = {}
        tickers = trending["ticker"].tolist()
        for ticker in tickers:
            try:
                cleaned = ticker.strip().upper()
                current = yf.Ticker(cleaned).fast_info.get("lastPrice", None)
                if current is not None:
                    current_prices[ticker] = current
            except Exception:
                current_prices[ticker] = None

        result = []
        for _, row in trending.iterrows():
            ticker = row["ticker"]
            avg_target = round(row["mean"], 2)
            count = int(row["count"])
            current_price = current_prices.get(ticker, None)

            result.append({
                "ticker": ticker,
                "avgTarget": avg_target,
                "count": count,
                "currentPrice": current_price
            })

        _cached_result = result
        _cache_expiry = time.time() + 900  # 15 minutes

        return result

    except Exception as e:
        return {"error": str(e)}
