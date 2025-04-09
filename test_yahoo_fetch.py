import yfinance as yf
import pandas as pd

ticker = "NVDA"
stock = yf.Ticker(ticker)

# Get all available expiration dates
expirations = stock.options
print(f"Available expiration dates for {ticker}:")
for date in expirations:
    print(date)

# Pick a far-dated one to inspect
target_expiry = "2026-01-16"
if target_expiry in expirations:
    calls = stock.option_chain(target_expiry).calls
    puts = stock.option_chain(target_expiry).puts

    print(f"\nFound {len(calls)} call options for {target_expiry}")
    print(calls.head())

    print(f"\nFound {len(puts)} put options for {target_expiry}")
    print(puts.head())
else:
    print(f"{target_expiry} not found in available expirations.")
