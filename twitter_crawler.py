import requests
import pandas as pd
from io import StringIO
from streamlit import secrets

class TwitterCrawler:
    def __init__(self):
        self.base_url = secrets["TWITTER_CRAWLER_API_URL"]

    def crawl_tweets(self, keyword, num_tweets=20, access_token=None) -> pd.DataFrame:
        response: requests.Response = requests.post(
            self.base_url,
            json={
                "keyword": keyword,
                "num_tweets": num_tweets,
                "access_token": access_token
            }
            )
        if not response.ok:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        data = response.content
        df = pd.read_csv(
            StringIO(data.decode('utf-8')),
            sep=',',
            header=0
        )
        return df
        