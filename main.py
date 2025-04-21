from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.stats import norm
from fastapi.middleware.cors import CORSMiddleware
import math
import traceback
from yahoo_api import fetch_option_chain_from_yahoo

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptionRequest(BaseModel):
    target_price: float
    target_date: str
    decision_date: str
    budget: float
    ticker: str
    price_mode: str = "ask"

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def explain_reason(row):
    drivers = []
    cost_per_contract = row["buy_price"]
    payoff_per_contract = row["bs_estimated_value"]
    contracts = row["contracts_affordable"]
    iv = row["implied_volatility"]
    T = row["T"]
    strike_diff = abs(row["strike"] - row["bs_target_price"])

    if iv < 0.3:
        drivers.append("Low implied volatility makes this contract cheap for its potential payoff")
    if cost_per_contract < 5 and contracts > 1:
        drivers.append("Low cost allows multiple contracts, amplifying potential gains")
    if T < 0.1 and payoff_per_contract > cost_per_contract * 5:
        drivers.append("Short expiry reduces cost while retaining significant payoff potential")
    if strike_diff < 5:
        drivers.append("Strike is well-aligned with your price target, maximizing intrinsic value")

    if not drivers:
        return "ROI driven by a blend of cost efficiency, premium, and target alignment"

    return " • ".join(drivers)

def generate_badges(row, df_all):
    badges = []
    n_total = len(df_all)
    roi_rank_str = f"({int(row['roi_rank'])}/{n_total})"

    if row["predicted_return"] < 0.05:
        badges.insert(0, {
            "icon": "priority-high",
            "label": "Option ROI is low — buying shares might be better",
            "pack": "MaterialIcons",
            "color": "orange"
        })

    peer_roi_75 = df_all["predicted_return"].quantile(0.75)
    if row["predicted_return"] > peer_roi_75:
        badges.append({
            "icon": "trending-up",
            "label": f"High ROI compared to peers {roi_rank_str}",
            "pack": "MaterialIcons"
        })

    if row["contracts_affordable"] >= 2:
        badges.append({
            "icon": "layers",
            "label": "Multiple contracts → flexible exit",
            "pack": "MaterialIcons"
        })

    if row["contracts_affordable"] == 1:
        badges.append({
            "icon": "block",
            "label": "Only 1 contract — can't exit partially",
            "pack": "MaterialIcons",
            "color": "red"
        })

    if row["T"] < 0.05:
        badges.append({
            "icon": "access-time",
            "label": "Short expiry — time risk",
            "pack": "MaterialIcons",
            "color": "red"
        })

    peer_median_iv = df_all["implied_volatility"].median()
    if row["implied_volatility"] < peer_median_iv:
        badges.append({
            "icon": "attach-money",
            "label": "Low implied volatility → cheaper premium",
            "pack": "MaterialIcons"
        })

    payoff_ratio = row["bs_estimated_value"] / row["buy_price"] if row["buy_price"] > 0 else 0
    median_ratio = (df_all["bs_estimated_value"] / df_all["buy_price"].replace(0, np.nan)).median()
    if payoff_ratio < median_ratio:
        badges.append({
            "icon": "warning",
            "label": "Lower payoff ratio than similar options",
            "pack": "MaterialIcons"
        })

    return badges

@app.post("/predict-options")
def predict_options(req: OptionRequest):
    try:
        target_price = float(req.target_price)
        target_date = pd.to_datetime(req.target_date)
        decision_date = pd.to_datetime(req.decision_date)
        budget = float(req.budget)
        ticker = req.ticker.upper()
        price_mode = req.price_mode.lower()
        contract_multiplier = 100
        r = 0.05

        print(f"[INFO] target_date: {target_date}, decision_date: {decision_date}")
        df_all = fetch_option_chain_from_yahoo(ticker)
        print(f"[INFO] Fetched {len(df_all)} total options for {ticker}")

        current_price = df_all["active_underlying_price"].iloc[0]
        print(f"[INFO] Current price of {ticker}: {current_price}")

        option_type = "C" if target_price >= current_price else "P"
        df = df_all[
            (df_all["expiration"] >= target_date) &
            (df_all["option_type"] == option_type)
        ].copy()

        df = df[df["impliedVolatility"] >= 0.01]
        df["T"] = ((df["expiration"] - decision_date).dt.days.clip(lower=0)) / 365

        if option_type == "C":
            df["bs_estimated_value"] = df.apply(
                lambda row: black_scholes_call(target_price, row["strike"], row["T"], r, row["impliedVolatility"]), axis=1
            )
        else:
            df["bs_estimated_value"] = df.apply(
                lambda row: black_scholes_put(target_price, row["strike"], row["T"], r, row["impliedVolatility"]), axis=1
            )

        df["mid_price"] = (df["bid"] + df["ask"]) / 2

        if price_mode == "ask":
            df["entry_price"] = df["ask"]
        elif price_mode == "mid":
            df["entry_price"] = df["mid_price"]
        else:
            df["entry_price"] = df["ask"]

        # ✅ New: Check for missing or stale bid/ask data
        if df["bid"].isnull().all() or df["ask"].isnull().all():
            print("[INFO] All options missing bid/ask. Likely market closed or stale data from Yahoo.")
            return {"market_closed_or_no_data": True}
        if ((df["bid"] == 0) & (df["ask"] == 0)).all():
            print("[INFO] All options have 0 bid/ask. Likely market closed or stale data from Yahoo.")
            return {"market_closed_or_no_data": True}

        df = df[df["entry_price"] > 0].copy()
        df["contract_cost"] = df["entry_price"] * contract_multiplier
        df["contracts_affordable"] = (budget // df["contract_cost"]).astype(int)
        df = df[df["contracts_affordable"] >= 1].copy()
        if df.empty:
            print("[INFO] No affordable options found. Returning early.")
            return {"no_profitable_options": True}

        df["total_cost"] = df["contracts_affordable"] * df["contract_cost"]
        df["total_value_at_target"] = df["contracts_affordable"] * df["bs_estimated_value"] * contract_multiplier
        df["predicted_profit"] = df["total_value_at_target"] - df["total_cost"]
        df["predicted_return"] = df["predicted_profit"] / df["total_cost"]
        df["bs_target_price"] = target_price
        df["buy_price"] = df["entry_price"]
        df["implied_volatility"] = df["impliedVolatility"]

        df = df[df["predicted_profit"].apply(lambda x: math.isfinite(x))].copy()
        df["roi_rank"] = df["predicted_return"].rank(method="min", ascending=False)
        df["explanation"] = df.apply(lambda row: explain_reason(row), axis=1)

        df["badges"] = [generate_badges(row, df) for _, row in df.iterrows()]

        result = df.sort_values(by="predicted_profit", ascending=False).head(5)
        all_negative_roi = bool(df["predicted_return"].max() <= 0)

        return {
            "no_profitable_options": all_negative_roi,
            "results": result[[
                "expiration", "option_type", "strike", "buy_price", "ask",
                "contracts_affordable", "total_cost", "predicted_profit", "predicted_return",
                "explanation", "badges", "bs_target_price", "T", "bs_estimated_value", "implied_volatility"
            ]].to_dict(orient="records")
        }

    except Exception as e:
        traceback.print_exc()
        return {"message": f"Server error: {str(e)}"}
