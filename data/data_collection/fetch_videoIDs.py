from googleapiclient.discovery import build

def get_videoIDs(keywords, published_after, published_before, api_key, filename="video_IDs.txt"):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    with open(filename, 'a') as f:

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
                    f.write(f"{video_id}\n")
                
        
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break  

if __name__ == '__main__':
    api_key = 'key'
    #keywords = ['NVIDIA stock', 'stock market', 'NVDA', 'investing', 'finance']
    #keywords = ['NVIDIA stock analysis', 'NVIDIA stock earnings', 'NVIDIA stock news', 'NVIDIA stock prediction', 'NVIDIA stock price','tech stocks','buy nvda', 'sell nvda', 'NVDA earnings']
    #keywords = ['AI stock investing', 'NVIDIA investor', 'high growth stocks', 'stock market trends', 'NVIDIA shares']
    #keywords = ['NVIDIA vs AMD', 'NVIDIA contoversy', 'nvda stock bubble', 'nvidia investors', 'NVIDIA future']
    #keywords = ['NVIDIA vs intel', 'NVIDIA profit', 'nvda stock bull case', 'nvidia stock buy or sell', 'NVIDIA stock split']
    #keywords = ['NVIDIA vs competitors', 'NVIDIA long-term growth', 'nvda stock all-time high', 'nvidia stock risks', 'NVIDIA stock trends']
    keywords = ['NVDA stock', 'NVDA VS S&P', 'stock market prediction', 'nvidia AI revolution', 'NVIDIA investment strategy']
    published_after = '2024-03-01T00:00:00Z'
    published_before = '2024-08-31T00:00:00Z'
    
    # fetch the ids
    get_videoIDs(keywords, published_after, published_before, api_key)
    print("Video IDs have been saved to 'video_IDs.txt'")
