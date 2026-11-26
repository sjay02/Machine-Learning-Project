import yfinance as yt
import pandas as pd
import os

os.makedirs("data",exist_ok=True)

stocks={
    "RELIANCE_NS":"RELIANCE.NS",
    "TCS_NS":"TCS.NS",
    "INFOSYS_NS":"INFY.NS",
    "APPLE":"AAPL",
    "MICROSOFT":"MSFT",
    "NVIDIA":"NVDA"

}

for filename,symbol in stocks.items():
    print(f"dowwnloading{symbol}...")

    data=yt.download(
        symbol,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        print(f"no data found for{symbol}")
        continue

    data=data[["Open","High","Low","Close","Volume"]]

    filepath=f"data/{filename}.csv"
    data.to_csv(filepath,index=False)

    print(f"saved:{filepath}")

print("\nAll downloads completed!")
