Congress Training Set

On 6/16/2022 and 6/17/2022, 1000 tweets were downloaded for each of the 529 current members of congress. Their twitter handels are all listed in `handels_congress_all.csv`. For downloading tweets, this file is broken into 6 files `handels_congress_100.csv`... `handels_congress_600.csv`.

Downloading the tweets was done using Colin Daniels's TwitterAPI repo: https://github.com/colindaniels/twitterAPI/tree/mr_daniels (Note that this has been branched)

To download tweets, use the following steps:
1. Use a mamba env containing `npm` for subsequent use of node to execute the program.
2. `npm install` will download the required dependencies identified in the `package.json` and `package-lock.json`.
3. mkdir `output`, as the program is looking for an empty directory by this name.
4. `node index.js handels_congress_100.csv 1000` will download the last 1000 tweets for each of the handles in the csv-file. A [handle].json file is created for each member of congress.
5. The program currently does not terminate, so after it looks to be done, it requires ctr-c to end the program. (This is a bug and may now be fixed)
6. After running each csv-file, move the `output` directory to the appropriate directory.
7. Here these directories are called: `tweets_congress-100` ... `tweets_congress-600`.
8. The individual directories were then merged into a single directory: `tweets-congress-2021-2022`.
9. `df = pd.concat(map(tweets2df, path.rglob("tweets-congress-2021-2022/*")))` will create a single dataframe containing all the tweets in the dataset.
10. `tweets2df` is defined in `ideology_utils.py`. It convert one or many `cvs` or `json` files into a single dataframe.
