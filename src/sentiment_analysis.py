import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk 

#nltk.download('vader_lexicon')

comments_df = pd.read_csv('data/cleaned_comments.csv', parse_dates=['date'])
comments_df = comments_df.dropna(subset=['cleaned_comment'])
comments_df['cleaned_comment'] = comments_df['cleaned_comment'].astype(str)

sia = SentimentIntensityAnalyzer()

comments_df['compound'] = comments_df['cleaned_comment'].apply(lambda x:sia.polarity_scores(x)['compound'])

# Calculate sentiment label based on compound score
def sentiment_label(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'
    
comments_df['sentiment_label'] = comments_df['compound'].apply(sentiment_label)

# get dominant sentiment label for each date
daily_label = comments_df.groupby('date')['sentiment_label'] \
    .agg(lambda x: x.value_counts().idxmax()) \
    .reset_index() \
    .rename(columns={'sentiment_label': 'dominant_sentiment'})

daily_sentiment = comments_df.groupby('date').agg(
    avg_sentiment=('compound', 'mean'),
    comment_count = ('compound', 'count'),

).reset_index()

# merge label with score
daily_sentiment = daily_sentiment.merge(daily_label, on='date')

daily_sentiment.to_csv('data/daily_sentiment.csv', index=False)
comments_df.to_csv('data/cleaned_comments_with_sentiment.csv', index=False)


print("Sentiment analysis completed")