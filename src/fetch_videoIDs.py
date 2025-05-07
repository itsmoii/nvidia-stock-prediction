from googleapiclient.discovery import build

def get_videoIDs(keywords, published_after, published_before, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_ids = []

    for keyword in keywords:
        next_page_token = None

        while True:
            request = youtube.search().list(
                part='snippet',
                q=keyword,
                type='video',
                publishedAfter=published_after,
                publishedBefore=published_before,
                maxResults=50,
                pageToken=next_page_token 
            )
            response = request.execute()
            
            for item in response['items']:
                video_id = item['id']['videoId']
                video_ids.append(video_id)
            
    
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break  

    return video_ids

def save_videoIDs(video_ids, filename="video_IDs.txt"):
    with open(filename, 'w') as f:
        for video_id in video_ids:
            f.write(f"{video_id}\n")

if __name__ == '__main__':
    api_key = 'api_key'
    keywords = ['NVIDIA stock', 'stock market', 'NVDA', 'investing', 'finance']
    published_after = '2024-03-01T00:00:00Z'
    published_before = '2025-03-31T00:00:00Z'
    
    # fetch the ids
    video_ids = get_videoIDs(keywords, published_after, published_before, api_key)
    
    # save ids to file
    save_videoIDs(video_ids)
    print(f"Saved {len(video_ids)} video IDs to 'video_IDs.txt'")
