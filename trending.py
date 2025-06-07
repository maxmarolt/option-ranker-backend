from fastapi import APIRouter
import gspread
import pandas as pd
import json
import os

router = APIRouter()

# Path to your service account credentials
CREDENTIALS_PATH = "google_creds.json"

# Google Sheet ID (from the sheet URL)
SHEET_ID = "1GLw524fGrKtsVByxV-wZ9NwuBNxeDBTzBqH3nx6xNtA"

@router.get("/trending-tickers")
def get_trending_tickers():
    try:
        creds_dict = json.loads(os.environ["GOOGLE_SHEETS_CREDS_JSON_APP"])
        gc = gspread.service_account_from_dict(creds_dict)
        sh = gc.open_by_key(SHEET_ID)
        worksheet = sh.sheet1  # first tab

        data = worksheet.get_all_records()

        # Create DataFrame
        df = pd.DataFrame(data)

        # Ensure required columns (case-sensitive)
        if "Event" not in df.columns or "Details" not in df.columns:
            return {"error": "Required columns not found in sheet"}

        # Filter only calculator submission rows
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

        # Filter out rows missing required fields
        parsed_df = parsed_df[parsed_df["ticker"].notnull() & parsed_df["targetPrice"].notnull()]

        if parsed_df.empty:
            return {"error": "No valid ticker data found"}


        # Compute trending stats
        trending = (
            parsed_df.groupby("ticker")["targetPrice"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values(by="count", ascending=False)
            .head(5)
        )

        result = []
        for _, row in trending.iterrows():
            result.append({
                "ticker": row["ticker"],
                "avgTarget": round(row["mean"], 2),
                "count": int(row["count"])
            })

        return result

    except Exception as e:
        return {"error": str(e)}
