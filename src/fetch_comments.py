from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re

def get_comments_for_trading_dates(api_key, video_ids_file='data/video_IDs.txt', trading_dates_file='data/trading_dates.txt', max_comments_per_date=500):
   
    with open(trading_dates_file, 'r') as f:
        trading_dates = [line.strip() for line in f.readlines()]
    
    with open(video_ids_file, 'r') as f:
        video_ids = [line.strip() for line in f.readlines()]
    
    keywords = ["nvidia", "nvda", "nvdia", "nvidia stock", "nvda stock"]
    keyword_pattern = re.compile(r'\b(?:' + '|'.join(keywords) + r')\b', re.IGNORECASE)
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments_by_date = {date: [] for date in trading_dates}
    
    # fetch comments
    for video_id in video_ids:
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
                    published_date = comment['publishedAt'].split('T')[0]  # split to only the date
                    
                    # get comments with keywords
                    if keyword_pattern.search(text):
                        # check if date exists in stock data + within limit of comments
                        if published_date in trading_dates and len(comments_by_date[published_date]) < max_comments_per_date:
                            comments_by_date[published_date].append((published_date, video_id, author, text))
                
               
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
            
            except HttpError as e:
                error_reason = e._get_reason()
                
                # skip videos with disabled comments
                if "commentsDisabled" in error_reason:
                    print(f"Skipping video {video_id} - comments are disabled.")
                    break
                
                print(f"Error fetching comments for video {video_id}: {error_reason}")
                break
    
    
    with open('comments_by_date.txt', 'a', encoding='utf-8') as f:
        f.write("date\tvideo_id\tauthor\tcomment\n")  # header row
        for date, comments in comments_by_date.items():
            for comment_data in comments:
                date, video_id, author, comment = comment_data
                
                author = author.replace("\t", " ")
                comment = comment.replace("\t", " ")
                f.write(f"{date}\t{video_id}\t{author}\t{comment}\n")
    
    print("Finished collecting comments.")
    return comments_by_date

if __name__ == '__main__':
    api_key = 'api key'
    comments = get_comments_for_trading_dates(api_key)
