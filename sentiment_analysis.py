from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np


# Softmax function for balancing sum of output sentiments
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Use NLTK's pretrained sentiment analysis tool to obtain "bullish/bearish" calculations
def pretrained_sentiment(tokenized_input):
    sia = SentimentIntensityAnalyzer()
    sentiments = {}
    for company in tokenized_input.keys():
        cs = {}
        for date in tokenized_input[company].keys():
            day_sentiments = []
            for tweet in tokenized_input[company][date]:
                untokenized_tweet = " ".join(tweet)
                day_sentiments.append(sia.polarity_scores(untokenized_tweet)['compound'])
            cs[date] = softmax(sum(day_sentiments))
        sentiments[company] = cs
    return sentiments
