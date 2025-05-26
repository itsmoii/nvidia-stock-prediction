import pandas as pd
import re
import regex
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

comments_df = pd.read_csv('comments.csv')

# remove duplicates
comments_df = comments_df.drop_duplicates(subset=['comment'])

# extract only the comment coloumn
comments_df = comments_df[['date','comment']].dropna()


def clean_text(text):
    # remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # remove special characters 
    text = regex.sub(r'[^\w\s\d\p{Emoji}]', '', text)
    # remove stopwords
    text = ' '.join([word.lower() for word in text.split() if word.lower() not in stop_words])
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\t\r\n]+', ' ', text).strip()
    # remove ellipsis
    text = re.sub(r'\.{2,}', ' ', text)
    # remove hashtags
    text = re.sub(r'#\w+', '', text)
    return text


comments_df['cleaned_comment'] = comments_df['comment'].apply(clean_text)

# save to csv
comments_df.to_csv('cleaned_comments.csv', index=False)

print("Comments cleaned and saved to 'cleaned_comments.csv'")
