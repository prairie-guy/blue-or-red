#!/usr/bin/env python
import csv, sys, re
import pandas as pd
from pathlib import Path

url = "https://theunitedstates.io/congress-legislators/legislators-current.csv"
def get_legislators(url=url):
    url = "https://theunitedstates.io/congress-legislators/legislators-current.csv"
    df = pd.read_csv(url)
    df = df[['first_name','last_name', 'state', 'party', 'twitter']]
    df = df[(df.party == 'Democrat') | (df.party == 'Republican')]
    df = df.rename(columns={'twitter':'handle'})
    df = df[df.handle.notnull()]
    return(df)

if __name__ == '__main__':
    df = get_legislators()
    print(df.to_csv(index=False))
