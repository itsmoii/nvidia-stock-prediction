import pandas as pd

# Load your datasets
stock_df = pd.read_csv('data/nvidia_stock.csv')        # Has a 'Date' column
comments_df = pd.read_csv('cleaned_comments.csv')  # Has a 'Date' column too

# Make sure the date columns are datetime type
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
comments_df['date'] = pd.to_datetime(comments_df['date']).dt.date  # just date part, no time

# Count comments per date
comment_counts = comments_df.groupby('date').size().reset_index(name='Comment_Count')

# Convert stock_df 'Date' to date (without time) for matching
stock_df['Date'] = stock_df['Date'].dt.date

# Merge comment counts with stock dataset on Date
merged_df = pd.merge(stock_df, comment_counts, left_on='Date', right_on='date', how='left')

# Fill NaN (dates with no comments) with 0
merged_df['Comment_Count'] = merged_df['Comment_Count'].fillna(0).astype(int)

# Save merged dataset if you want
merged_df.to_csv('mergedCount.csv', index=False)

print(merged_df.head())

