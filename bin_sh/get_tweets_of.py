#!/usr/bin/env python
import tweepy, csv, sys, re
import pandas as pd
from pathlib import Path
import config  # Authentication Tokens

client = tweepy.Client(config.BEARER_TOKEN)

def get_tweets_of(handle, path='.', limit=1000):
    path = Path(path)/handle
    query = f'from:{handle} -is:retweet'
    res = []
    with open(path.with_suffix('.csv'), "w") as fn:
        writer = csv.writer(fn, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['handle','text'])
        for tweet in tweepy.Paginator(client.search_recent_tweets, query = query).flatten(limit=limit):
            text = clean_text(tweet.text)
            writer.writerow([handle, text])
    print(f'{handle}.csv')


def clean_text(text):
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub('[\\r,\\n]','', text)
    return text

if __name__ == '__main__':
    if len(sys.argv) == 2:
        _, handle = sys.argv
        path = Path()
    elif len(sys.argv) == 3:
        _, handle, path = sys.argv
        path = Path(path)
    else:
        print("usage: get_tweets_of handle [path]")
        exit()
    get_tweets_of(handle, path=path)
