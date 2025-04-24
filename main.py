from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.stats import norm
from fastapi.middleware.cors import CORSMiddleware
import math
import traceback
from yahoo_api import fetch_option_chain_from_yahoo
import yfinance as yf

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
def predict_options(req: OptionRequest, request: Request):

    try:
        mode = request.query_params.get("mode", "roi")

        # Early check for strategic mode (block short-term predictions)
        if mode == "profit":
            min_days_ahead = 14
            target_date_obj = pd.to_datetime(req.target_date).date()
            today_date_obj = pd.Timestamp.today().date()
            days_until_target = (target_date_obj - today_date_obj).days

            if days_until_target < min_days_ahead:
                print(f"[DEBUG] Strategic mode blocked: only {days_until_target} days ahead")
                return {
                    "no_profitable_options": True,
                    "message": f"Strategic mode requires a longer time horizon. Try a target date at least {min_days_ahead} days from today."
                }

        print(f"[DEBUG] Strategy mode received: {mode}")


        target_price = float(req.target_price)
        target_date = pd.to_datetime(req.target_date)
        decision_date = pd.to_datetime(req.decision_date)
        budget = float(req.budget)
        ticker = req.ticker.upper()
        price_mode = req.price_mode.lower()
        contract_multiplier = 100
        r = 0.05

        df_all = fetch_option_chain_from_yahoo(ticker)
        total_contracts_available = len(df_all)
        current_price = df_all["active_underlying_price"].iloc[0]

        option_type = "C" if target_price >= current_price else "P"
        df = df_all[
            (df_all["expiration"] >= target_date) &
            (df_all["option_type"] == option_type)
        ].copy()
        contracts_after_prediction_filter = len(df)


        df = df[df["impliedVolatility"] >= 0.01]
        df["T"] = ((df["expiration"] - target_date).dt.days.clip(lower=0)) / 365

        if option_type == "C":
            df["bs_estimated_value"] = df.apply(
                lambda row: black_scholes_call(target_price, row["strike"], row["T"], r, row["impliedVolatility"]), axis=1
            )
        else:
            df["bs_estimated_value"] = df.apply(
                lambda row: black_scholes_put(target_price, row["strike"], row["T"], r, row["impliedVolatility"]), axis=1
            )

        df["mid_price"] = (df["bid"] + df["ask"]) / 2
        df["entry_price"] = df["ask"] if price_mode == "ask" else df["mid_price"]

        if df["bid"].isnull().all() or df["ask"].isnull().all():
            return {"market_closed_or_no_data": True}
        if ((df["bid"] == 0) & (df["ask"] == 0)).all():
            return {"market_closed_or_no_data": True}

        df = df[df["entry_price"] > 0].copy()
        df["contract_cost"] = df["entry_price"] * contract_multiplier
        df["contracts_affordable"] = (budget // df["contract_cost"]).astype(int)
        df = df[df["contracts_affordable"] >= 1].copy()

        if df.empty:
            return {"no_profitable_options": True, "message": "No contracts found within budget."}

        if mode == "profit":

            print(f"[DEBUG] Starting strategic filter: {len(df)} contracts before filtering")

            df["days_to_expiration"] = df["T"] * 365

            # Find approximate ATM strike
            atm_strike = df.iloc[(df["strike"] - current_price).abs().argsort()].iloc[0]["strike"]
            atm_price = df[df["strike"] == atm_strike]["entry_price"].mean()

            # Tiered minimum price filter based on current price
            if current_price < 20:
                min_price = atm_price * 0.15
            elif current_price < 500:
                min_price = atm_price * 0.10
            else:
                min_price = atm_price * 0.05

            df = df[df["entry_price"] >= min_price]

            # 📉 Dynamically filter strike range using implied move estimate
            df["T_current"] = ((df["expiration"] - pd.Timestamp.today()).dt.days.clip(lower=0)) / 365
            df["expected_move"] = current_price * df["impliedVolatility"] * np.sqrt(df["T_current"])
            risk_factor = 1.5 if mode == "roi" else 1.2
            upper_bound = current_price + df["expected_move"] * risk_factor
            lower_bound = current_price - df["expected_move"]

            df["target_too_far"] = (target_price > upper_bound) | (target_price < lower_bound)


            print(f"[DEBUG] Current price: {current_price}")
            print(f"[DEBUG] Sample expected move range:")
            print(df[["strike", "expiration", "expected_move"]].head(10))

            df = df[
                (df["strike"] >= current_price - df["expected_move"]) &
                (df["strike"] <= current_price + df["expected_move"] * 1.5)  # allow extra upside
            ]


            # Expiry must be more than 14 days out
            df = df[df["days_to_expiration"] > 14]

            # Remove dead options (no movement potential)
            df = df[df["impliedVolatility"] * df["T"] >= 0.08]

            print(f"[DEBUG] After strategic filtering: {len(df)} contracts remain")

            if df.empty:
                return {
                    "no_profitable_options": True,
                    "message": "No strategically viable options found. Try increasing your budget or relaxing your target. Aggressive contracts may be necessary."
                }



        df["total_cost"] = df["contracts_affordable"] * df["contract_cost"]
        df["total_value_at_target"] = df["contracts_affordable"] * df["bs_estimated_value"] * contract_multiplier
        df["predicted_profit"] = df["total_value_at_target"] - df["total_cost"]
        df["predicted_return"] = df["predicted_profit"] / df["total_cost"]
        df["bs_target_price"] = target_price
        df["buy_price"] = df["entry_price"]
        df["implied_volatility"] = df["impliedVolatility"]

        df = df[df["predicted_profit"].apply(lambda x: math.isfinite(x))].copy()
        if df.empty:
            return {"no_profitable_options": True, "message": f"No profitable contracts met filtering criteria in '{mode}' mode."}

        df["roi_rank"] = df["predicted_return"].rank(method="min", ascending=False)
        df["explanation"] = df.apply(lambda row: explain_reason(row), axis=1)
        df["badges"] = [generate_badges(row, df) for _, row in df.iterrows()]

        sort_by = "predicted_return" if mode == "roi" else "predicted_profit"
        print(f"[DEBUG] Sorting by: {sort_by}")
        contracts_considered_final = len(df)

        result = df.sort_values(by=sort_by, ascending=False).head(5)

        print("Top results:\n", result[[
            "expiration", "strike", "buy_price", "bs_estimated_value",
            "implied_volatility", "predicted_return", "predicted_profit"
        ]])


        return {
            "no_profitable_options": False,
            "mode": mode,
            "results": result[[
                "expiration", "option_type", "strike", "buy_price", "ask",
                "contracts_affordable", "total_cost", "predicted_profit", "predicted_return",
                "explanation", "badges", "bs_target_price", "T", "bs_estimated_value", "implied_volatility"
            ]].to_dict(orient="records"),
            "stats": {
                "total_contracts_available": total_contracts_available,
                "contracts_after_prediction_filter": contracts_after_prediction_filter,
                "contracts_considered_final": contracts_considered_final
            }
        }


    except Exception as e:
        traceback.print_exc()
        return {"message": f"Server error: {str(e)}"}


@app.get("/vol-surface")
def get_vol_surface(ticker: str):
    try:
        # 🔹 Step 1: Get current stock price
        ticker_obj = yf.Ticker(ticker)
        price_df = ticker_obj.history(period="1d")
        if price_df.empty:
            raise ValueError("Unable to fetch current price for ticker.")
        current_price = price_df["Close"].iloc[-1]

        # 🔹 Step 2: Fetch full option chain
        df = fetch_option_chain_from_yahoo(ticker)
        df = df[df["impliedVolatility"].notnull()]

        # 🔹 Step 3: Compute days to expiry
        df["days_to_exp"] = (df["expiration"] - pd.Timestamp.today()).dt.days
        df = df[df["days_to_exp"] > 0]

        # 🔹 Step 4: Apply filters
        df = df[df["days_to_exp"] <= 90]  # Filter to 3 months out
        df = df[
            (df["strike"] >= current_price * 0.5) &
            (df["strike"] <= current_price * 1.5)
        ]

        if df.empty:
            return {"error": "No valid options found in specified range."}

        # 🔹 Step 5: Clean and bin
        df["strike"] = df["strike"].round(1)
        df["days_to_exp"] = (df["days_to_exp"] // 5) * 5  # bucket by 5-day intervals
        df["days_to_exp"] = df["days_to_exp"].astype(int)

        # 🔹 Step 6: Build the matrix
        strikes = sorted(df["strike"].unique())
        expiries = sorted(df["days_to_exp"].unique())

        iv_matrix = []
        for strike in strikes:
            row = []
            for expiry in expiries:
                match = df[
                    (df["strike"] == strike) &
                    (df["days_to_exp"] == expiry)
                ]
                if not match.empty:
                    iv_val = float(match["impliedVolatility"].mean()) * 100  # avg of matching rows
                    row.append(round(iv_val, 2))
                else:
                    row.append(None)
            iv_matrix.append(row)

        return {
            "strikes": [float(s) for s in strikes],
            "expiries": [int(e) for e in expiries],
            "ivs": iv_matrix
        }

    except Exception as e:
        print(f"[ERROR] Failed to generate vol surface: {e}")
        return {"error": str(e)}

@app.get("/current-price")
def get_current_price(ticker: str):
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if price:
            return {"price": price}
        else:
            return {"error": "Price not found"}
    except Exception as e:
        return {"error": str(e)}
