import tweepy, csv, sys, re, time, warnings
from fastai.text.all import *
from datetime import date
from pathlib import Path
from emoji import demojize
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, DatasetDict, load_metric
from transformers import TrainingArguments, Trainer
 # Authentication Tokens
import config

def get_tweets_of(handle, path='tweets', proc_func=None, limit=1000):
    "Doownload and save tweets of `handle` to `path='.'`"
    client = tweepy.Client(config.BEARER_TOKEN)
    path = Path(path)/handle
    query = f'from:{handle} -is:retweet'
    res = []
    with open(path.with_suffix('.csv'), "w") as fn:
        writer = csv.writer(fn, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['handle','text'])
        for tweet in tweepy.Paginator(client.search_recent_tweets, query = query).flatten(limit=limit):
            text = proc_func(tweet.text) if proc_func else tweet.text
            writer.writerow([handle, text])
    print(f'{handle}.csv')

def get_tweets_of_group(df_or_csv, group_name, path='tweets', handle='handle', proc_func=None):
    """
Takes a `df_or_csv`, containing a twitter `handle`, creates a `dir` named `group_name`,
which will containt a `<handle>,csv` file with the tweets for that handle.
    """
    if type(df_or_csv) == pd.core.series.Series:
        handles = df_or_csv
    elif type(df_or_csv) == pd.core.frame.DataFrame:
        handles = df[handle]
    elif type(df_or_csv) == str and Path.is_file(Path(df_or_csv)):
        handles = pd.read_csv(df_or_csv)[handle]
    else:
        print("usage: get_tweets_of_group df_or_csv group_name [handle]")
        return
    path = Path(path)/f"tweets-{group_name}-{str(date.today())}"
    Path(path).mkdir(parents=True, exist_ok=True)
    for handle in handles:
        try:
            get_tweets_of(handle, path=path, proc_func=proc_func)
        except:
            for m in range(15):
                print(f"Minutes waiting: {m}")
                time.sleep(65)
            get_tweets_of(handle, path)
    return path

def tweets2df(path):
    "Takes a String or Path of either a file or dir, of types `csv` or `json`, and returns a dataframe"
    if type(path) != 'pathlib.PosixPath': path = Path(path)

    if   path.is_file() and path.suffix=='.csv':            func = pd.read_csv
    elif path.is_file() and path.suffix=='.json':           func = pd.read_json
    elif path.is_dir()  and path.ls()[0].suffix == '.csv':  func = pd.read_csv
    elif path.is_dir()  and path.ls()[0].suffix == '.json': func = pd.read_json
    else: return "usage: tweets2df(path), where `path` is a `file` or `dir` of `csv` or `json`"

    if path.is_file(): return func(path).dropna().reset_index(drop=True)
    if path.is_dir():  return pd.concat([func(f).dropna().reset_index(drop=True)
                                           for f in path.ls() if path.ls()[0].suffix == f.suffix], ignore_index=True)
    return("usage: tweets2df(path), where `path` is a `file` or `dir` of type `csv` or `json`")

def preprocess_tweet(text, max_len = 500):
    "Preprocess text in tweets, fixing @<name> -> @user, http://something -> http"
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        t = demojize(t) if len(t) == 1 else t
        t = re.sub('[\\r,\\n]','', t)
        t = " ".join(t.split())
        new_text.append(t)
    #new_text = new_text.str.encode("ascii", "ignore").str.decode('ascii') # Strip out pnon-ascii characters
    return " ".join(new_text)[:max_len]

def preprocess_tweets(df, max_len = 500, text_col='text', proc_func = preprocess_tweet):
    "Preprocess all tweets in a `df` of `max length` with col name of `text`"
    df = df.dropna().reset_index(drop=True)
    df[text_col] = df[text_col].apply(proc_func)
    df[text_col] = df[text_col].str[:max_len]
    #df[text_col] = df[text_col].str.encode("ascii", "ignore").str.decode('ascii')
    return df

def get_tweets_of_legislators(path='tweets', proc_func=None):
    "Convenience function to download most current members of congress"
    df = get_legislators()
    handles = df.handle
    get_tweets_of_group(handles, 'congress', path=path, proc_func=proc_func)

def label_tweets_of_legislators(df):
    dfl = get_legislators()[['handle','party']]
    df = df.merge(dfl)
    df = df[(df.party == 'Democrat') | (df.party == 'Republican')]
    df = df.reset_index(drop=True)
    return df

def get_legislators(url="https://theunitedstates.io/congress-legislators/legislators-current.csv"):
    "Get most recent US legislators provided in https://theunitedstates.io"
    url = "https://theunitedstates.io/congress-legislators/legislators-current.csv"
    df = pd.read_csv(url)
    df = df[['first_name','last_name', 'state', 'party', 'twitter']]
    df = df[(df.party == 'Democrat') | (df.party == 'Republican')]
    df = df.rename(columns={'twitter':'handle'})
    df = df[df.handle.notnull()]
    return(df)

## Fastai Models

def ideology_score(csv_file, model, text_col='text'):
    #warnings.simplefilter(action='ignore', category=FutureWarning)
    df = tweets2df(csv_file)
    if df.empty:
        print(f"empty: {csv_file}")
        return
    learn = load_learner(model)
    dl_test = learn.dls.test_dl(df)
    predicts, _ = learn.get_preds(dl = dl_test)
    d,r =  (sum(predicts)/len(predicts)).numpy().tolist()
    clas = np.argmax(predicts, axis=1)
    clas = int(sum(clas))/len(predicts)
    #return [round(clas,2), round(r,2), len(df[text_col])]
    return [df.handle[0], round(clas,2), round(r,2), len(df[text_col])]

def scores2df(scores, sortby='s1', columns=['s1','s2','n']):
    "Converts dict `scores` to `df` of form: scores = {handle:[s1,s2,n]}"
    df = pd.DataFrame(scores).transpose()
    df.columns = columns
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'handle'})
    df = df.sort_values(by=[sortby])
    df.dropna(inplace = True)
    return df

def load_fastai_model(path='ulmfit-2022.pkl'):
    return load_learner(model)

def df2dl(df, model):
    return model.dls.test(df)

def predict_fastai(dl_eval,model):
    preds, _ = learn.get_preds(dl = dl_test)
    return preds


## Transformer Models

def load_tfms_model(path = 'blue-or-red-roberta-2022', bs = 256):
    "Returns a `Transformer` based trainer and tokz , loading a saved tfms_model from dir `path`"
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokz  = AutoTokenizer.from_pretrained(path)
    args = TrainingArguments("tmp_trainer", per_device_eval_batch_size=bs)
    trainer = Trainer(model, args, tokenizer=tokz)
    return (trainer, tokz)

def df2ds(df, tokz, max_tokens = 128, text_col='text', label_col=None):
    "Takes df with `text_col`, `tokz` and optional `max_tokens` and 'label_col`. Returns a tokenized DataSet"
    if text_col not in df.columns: raise Exception(f'text_col: `{text_col}` not in df')
    df_eval = df.rename(columns={text_col:'input'})
    if label_col and label_col not in df_eval.columns: raise Exception(f'label_col: `{label_col}` not in df')
    df_eval = df_eval.rename(columns={label_col:'labels'})
    ds_eval = Dataset.from_pandas(df_eval).map(lambda x: tokz(x['input']), batched=True)
    ds_eval = ds_eval.filter(lambda x: len(x['input_ids']) < max_tokens)
    return ds_eval

def predict_tfms(ds_eval, trainer):
    "Takes `ds_eval` comprized to a data `trainer` and `tokz` then evaluates the trainer"
    preds = trainer.predict(ds_eval).predictions.astype(float)
    preds = torch.tensor(preds)
    preds = F.softmax(preds, dim = 1)
    return preds

def compute_metrics(preds_labels):
    metric = load_metric("accuracy")
    preds, labels = preds_labels
    preds = np.argmax(preds, axis=1)
    return metric.compute(predictions=preds, references=labels)

# Utils
def tok_func(x): return tokz(x["input"])
def party2num(x):return 0 if x=='Democrat' else 1
