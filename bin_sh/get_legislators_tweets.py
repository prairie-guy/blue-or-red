#!/usr/bin/env ipython

import time
from datetime import date
from get_tweets_of import *
from get_legislators import *

def download_tweets():
    df = get_legislators()
    handles = df.handle
    path = f"tweets-{str(date.today())}"
    Path(path).mkdir(parents=True, exist_ok=True)
    for handle in handles:
        try:
            get_tweets_of(handle, path)
        except:
            for m in range(15):
                print(f"Minutes waiting: {m}")
                time.sleep(65)
            get_tweets_of(handle, path)

if __name__ == '__main__':
    print("Getting Legislator Tweets")
    print(f"Saving to: tweets-{str(date.today())}")
    download_tweets()
