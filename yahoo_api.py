import yfinance as yf
import pandas as pd

def fetch_option_chain_from_yahoo(ticker: str):
    """
    Fetch the full option chain for all expiration dates for a given ticker using Yahoo Finance.
    """
    ticker_obj = yf.Ticker(ticker)
    expirations = ticker_obj.options

    if not expirations:
        raise ValueError("No option expirations found for this ticker.")

    all_data = []

    for exp in expirations:
        try:
            chain = ticker_obj.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()

            calls["option_type"] = "C"
            puts["option_type"] = "P"
            calls["expiration"] = pd.to_datetime(exp)
            puts["expiration"] = pd.to_datetime(exp)

            all_data.extend([calls, puts])
        except Exception as e:
            print(f"[WARNING] Failed to load data for expiration {exp}: {e}")

    df_all = pd.concat(all_data, ignore_index=True)

    # Add underlying price
    underlying_price = ticker_obj.info.get("regularMarketPrice", None)
    df_all["active_underlying_price"] = underlying_price

    return df_all
