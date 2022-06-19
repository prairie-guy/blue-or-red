#!/usr/bin/env python
from fastai.text.all import *
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def blue_red_score(csv_file, model, col_text='text'):
    df = pd.read_csv(csv_file)
    try:
        df.text[0]
    except:
        print(f"empty: {csv_file}")
        exit()
    learn = load_learner(model)
    #dl_test = TextDataLoaders.from_df(cleanup_text(df), valid_pct = 0, text_col='text', shuffle=False)[0]
    dl_test = learn.dls.test_dl(df.text.to_list())
    predicts, actuals = learn.get_preds(dl = dl_test)
    d,r =  (sum(predicts)/len(predicts)).numpy().tolist()
    clas = np.argmax(predicts, axis=1)
    clas = int(sum(clas))/len(predicts)
    return [df.handle[0], round(clas,2), round(r,2), len(df[col_text])]

def cleanup_text(df):
    #df.text = df.text.str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '')
    df = df[df.text.notnull()]
    df.text = df.text.str.replace('[\\r,\\n]','')
    df = df.dropna()
    df = df.reset_index()
    return df

if __name__ == '__main__':
    if len(sys.argv) == 3:
        _, csv_file, model = sys.argv
    else:
        print("usage: ideology_score [csv_file|csv_dir] model")
        exit()
    print(blue_red_score(csv_file, model))
