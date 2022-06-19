# blue-or-red
Fastai and Transformer Models trained on Congressional Tweets

## Nature of Project
This is an outgrowth of my fifth fastai course. The most notable changes between then and now is the large and rapid advancement with Natural Language Processing (NLP). The use of transfer learning and the emergence of the **Transformer** class of models, such as **GTP-3**, have opened up many interesting applications. 

In 2019, I started a project to use congressional tweets (labeled as Democrat or Republican) to develop an **Ideology Score** which might then be used to analyze sets of individual or group tweets to predict the relative **Ideological Leaning**. Arbitrarily, I choose `Democrat = 0` and `Republican = 1`. 

This project seeks to validate the results of that project as I still have the approximate 700,000 original 2018 - 2019 tweets. Moreover, I wanted to experiment with the new **Transformer** class of models to see what improvements in accuracy I might be able to achieve. As the **fastai** library does not yet access these models, I used the **HuggingFace** library of models and resources for this incarnation of **Ideology Scoring**.

Annoyingly, **Twitter** significantly restricted the use of their API to downloading tweets from only the previous 7-days. This limited the number of tweets I was able to get. Initially, I was downloading all congressional tweets on a weekly basis. As the projected evolved, I was able to use screen scraping techniques to gather a much larger number of tweets.

Though the project is ongoing, I have found that the **[cardiffnlp/twitter-roberta-base](https://huggingface.co/cardiffnlp/twitter-roberta-base)**, pre-trained on 58 million tweets, has significantly improved accuracy to approximately 86-87%.

Another change in this project is a more predicable pipeline for the downloading, pre-processing and using tweets. These are all python scripts found it `ideology_utils.py`. It also contains a variety of utility functions, including: downloading tweets for a single handle or a group of handles to csv files, loading these csv files to dataframes and prepossessing tweets. Additional functions are available to load both **fastai* and **transformer** type models to make predictions on test tweets. (These are still a work in process.)

## Preliminary Results
**WORK IN PROCESS**

A strategy of transfer learning was used. Both **fastai** model USMLFit and the **[twitter-roberta-base]** models were fine-tuned with tweets labeled as being either **Democratic** or **Republican**.

For classifying unknown test tweets, the **fastai** fine-tuned USMLFit model is achieving accuracy of **[~78-82]%**. The **twitter-roberta-base** fine-tuned model is achieving accuracy of **[84-88]%.

## Technical notes
Because of the size of the models (~1.9 Gig), these have been uploaded. Specifically, are the **fastai** `*.pkl` and `*.pth` and **Transformer** `*.bin` files were not uploaded. Initially I tried using `git-lfs`, but this was too awkward and expensive. Accordingly, these files have been added to the `.gitignore` file. 

The tweets themselves are available in csv file format in the folder `tweets`. Once the Jupyter notebooks are properly edited, the reader should be able to retrain the models themselves.

There are a variety of Jupyter notebooks that I have accumulated over the project. These need to be completely reviewed as some are no longer applicable and others need to be edited to ore consistently utilize the functions in in `ideology_utils.py`

<!--  LocalWords:  Fastai fastai NLP GTP HuggingFace cardiffnlp roberta pre py
 -->
<!--  LocalWords:  utils csv dataframes USMLFit pkl pth lfs gitignore Jupypter
 -->
<!--  LocalWords:  Jupyter
 -->
