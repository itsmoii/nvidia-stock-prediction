from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re
import os
import csv
import json

def get_comments_for_trading_dates(api_key, video_ids_file='video_IDs.txt', trading_dates_file='data/trading_dates.txt', checkpoint_file='checkpoint.txt', output_file='comments.csv', max_comments_per_date=200):
    # trading dates
    with open(trading_dates_file, 'r') as f:
        trading_dates = set(line.strip() for line in f.readlines())
    
    # video IDs
    with open(video_ids_file, 'r') as f:
        video_ids = [line.strip() for line in f.readlines()]

    # processed video IDs
    processed_ids = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_ids.update(line.strip() for line in f.readlines())
    
    # keywords for filtering comments
    keywords = ["nvidia", "nvda", "nvdia", "nvidia stock", "nvda stock"]
    keyword_pattern = re.compile(r'\b(?:' + '|'.join(keywords) + r')\b', re.IGNORECASE)
    
    # youtube API client
    youtube = build('youtube', 'v3', developerKey=api_key)

    # keep track of comment counts for each date
    if os.path.exists('date_comment_counts.json'):
        with open('date_comment_counts.json', 'r') as f:
            date_comment_counts = json.load(f)
    else:
        date_comment_counts = {date: 0 for date in trading_dates}

    # open the comments csv
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "video_id", "author", "comment"])  # header

        # fetching comments from each video id
        for video_id in video_ids:
            if video_id in processed_ids:
                print(f"Skipping {video_id}, already processed.")
                continue
            
            next_page_token = None
            
            while True:
                try:
                    request = youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        textFormat='plainText',
                        maxResults=100,
                        pageToken=next_page_token
                    )
                    response = request.execute()
                    
                    for item in response['items']:
                        comment = item['snippet']['topLevelComment']['snippet']
                        text = comment['textDisplay']
                        author = comment['authorDisplayName']
                        published_date = comment['publishedAt'].split('T')[0]
                        
                        # comment was made on trading data + contains keyword
                        if published_date in trading_dates and keyword_pattern.search(text):
                            if date_comment_counts[published_date] < max_comments_per_date:
                                writer.writerow([published_date, video_id, author.replace("\t", " "), text.replace("\t", " ")])
                                date_comment_counts[published_date] += 1
                            else:
                                print(f"Reached max comments ({max_comments_per_date}) for {published_date}. Skipping further comments for this date.")
                                continue
                    # move to next page of results
                    next_page_token = response.get('nextPageToken')
                    if not next_page_token:
                        break
                
                except HttpError as e:
                    error_reason = str(e)
                    
                    # stop if quota exceeded
                    if "quotaExceeded" in error_reason:
                        print("Quota exceeded. Stopping script.")
                        return
                    
                    # skip videos with disabled comments
                    if "commentsDisabled" in error_reason:
                        print(f"Skipping video {video_id} - comments are disabled.")
                        break
                    
                    print(f"Error fetching comments for video {video_id}: {error_reason}")
                    break
            
            # processed ids
            with open(checkpoint_file, 'a') as chk:
                chk.write(video_id + "\n")

            with open('date_comment_counts.json', 'w') as f:
                json.dump(date_comment_counts, f)
            
            print(f"Finished processing {video_id}")
    
    print("Finished collecting comments.")

if __name__ == '__main__':
    api_key = 'key'
    get_comments_for_trading_dates(api_key)
