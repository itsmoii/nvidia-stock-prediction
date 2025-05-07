import yfinance as yf

def fetch_stock_data(ticker="NVDA", start_date="2024-03-01", end_date="2025-03-01", output_file="data/nvidia_stock.csv"):
    # fetch stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # save to csv
    stock_data.to_csv(output_file)
    print(f"Stock data saved to {output_file}")

if __name__ == "__main__":
    ticker = "NVDA"
    start_date = "2024-03-01"
    end_date = "2025-03-31"
    
    # save the data
    fetch_stock_data(ticker, start_date, end_date)
