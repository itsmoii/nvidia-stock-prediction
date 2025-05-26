import pandas as pd

def get_trading_dates():
    df = pd.read_csv('data/nvidia_stock.csv', header=2)
    
    trading_dates = sorted(df['Date'].unique())
    
    with open('trading_dates.txt', 'w') as f:
        for date in trading_dates:
            f.write(date + '\n')
    
    print(f"Extracted {len(trading_dates)} trading dates from stock data.")
    return trading_dates

if __name__ == '__main__':
    trading_dates = get_trading_dates()
