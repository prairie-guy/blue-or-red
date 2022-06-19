#!/usr/bin/env ipython

import time
import pandas as pd
from datetime import date
from get_tweets_of import *

def download_tweets_of_group(csv_file, group_name, handle_name='handle'):
    """
Takes a `csv_file`, containing a twitter `handle`, creates a `dir` named `group_name`,
which will containt a `<handle>,csv` file with the tweets for that handle.
    """
    try:
        handles = pd.read_csv(csv_file)[handle_name]
    except:
        print("usage: get_tweets_of csv_file group_name [handle_name]")
    path = f"{group_name}-{str(date.today())}"
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
    if len(sys.argv) == 3:
        _, csv_file, group_name  = sys.argv
        handle_name = 'handle'
    elif len(sys.argv) == 4:
        _, csv_file, group_name, handle_name = sys.argv
    else:
        print("usage: get_tweets_of csv_file group_name [handle_name]")
        exit()
    print(f"Saving Tweets of {group_name} to: {group_name}-{str(date.today())}")
    download_tweets_of_group(csv_file, group_name, handle_name)
