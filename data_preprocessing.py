import os
from datetime import datetime
import numpy as np
from torch import tensor
from nltk.sentiment import SentimentIntensityAnalyzer
import json
from utils import daterange

# Generates a dictionary of lists containing all price data for a given security at each available date between 1/1/2014 and 12/31/2016
def load_stock_price_data(data_loc, start_dt, end_train, end_val, end_dt):

    train_pd = {}
    val_pd = {}
    test_pd = {}

    for company in os.listdir(data_loc):

        with open(os.path.join(data_loc, company)) as txt:
            lines = txt.readlines()
        lines.reverse()
        train_entries = []
        val_entries = []
        test_entries = []
        for line in lines:
            split_line = line.rstrip().split('\t')
            dt = datetime.strptime(split_line[0], '%Y-%m-%d')
            if dt >= start_dt and dt <= end_dt:

                split_line[0] = dt
                if dt >= start_dt and dt <= end_train:
                    train_entries.append(split_line)
                elif dt <= end_val:
                    val_entries.append(split_line)
                else:
                    test_entries.append(split_line)

        train_pd[company[:-4]] = train_entries
        val_pd[company[:-4]] = val_entries
        test_pd[company[:-4]] = test_entries

    return train_pd, val_pd, test_pd


# Generate dictionary of dictionary of lists of tweet data in json format, using keys of securities for first dict and dates for second
def load_tweet_data(data_loc, end_train, end_val):
    td_train = {}
    td_val = {}
    td_test = {}
    for company in os.listdir(data_loc):
        cd_train = {}
        cd_val = {}
        cd_test = {}
        for date in os.listdir(os.path.join(data_loc, company)):
            dt = datetime.strptime(date, '%Y-%m-%d')
            with open(os.path.join(data_loc, company, date)) as tweets_on_date:
                json_tweets = []
                for line in tweets_on_date.readlines():
                    json_tweets.append(json.loads(line)['text'])
            if dt <= end_train:
                cd_train[dt] = json_tweets
            elif dt <= end_val:
                cd_val[dt] = json_tweets
            else:
                cd_test[dt] = json_tweets
        td_train[company] = cd_train
        td_val[company] = cd_val
        td_test[company] = cd_test
    return td_train, td_val, td_test


def combine_price_and_sentiment(price_data, sentiment_data, date_start, date_end):
    p_and_s = {}
    for company in sentiment_data.keys():
        company_ps = {}
        for date in daterange(date_start, date_end):
            date_ps = np.zeros(7)
            # Currently inefficient, if this becomes bottleneck then fix
            for date_list in price_data[company]:
                if date_list[0] == date:
                    date_ps[:6] = date_list[1:]
            if date in sentiment_data[company]:
                date_ps[6] = sentiment_data[company][date]
            if not (date_ps == np.zeros(7)).all():
                company_ps[date] = date_ps
        p_and_s[company] = company_ps
    return p_and_s


def gen_timeseries_samples(data, step, lag):
    timeseries = []
    for company in data.keys():
        company_samples = [value for (key, value) in sorted(data[company].items())]
        company_samples = [item for item in company_samples if item[0] != 0]  # Remove values where percent change is zero
        for i in range(0, len(company_samples) - lag, step):
            timeseries.append(company_samples[i:i+lag+1])
    return np.array(timeseries)


def normalize_features(data1, data2, data3):
    d_min1 = np.amin(data1, axis=(0, 1))
    d_max1 = np.amax(data1, axis=(0, 1))

    d_min2 = np.amin(data2, axis=(0, 1))
    d_max2 = np.amax(data2, axis=(0, 1))

    d_min3 = np.amin(data1, axis=(0, 1))
    d_max3 = np.amax(data1, axis=(0, 1))

    d_min = np.minimum(d_min1, d_min2, d_min3)
    d_max = np.maximum(d_max1, d_max2, d_max3)

    norm_data1 = (data1 - d_min[None, None, :]) / (d_max[None, None, :] - d_min[None, None, :])
    norm_data2 = (data2 - d_min[None, None, :]) / (d_max[None, None, :] - d_min[None, None, :])
    norm_data3 = (data3 - d_min[None, None, :]) / (d_max[None, None, :] - d_min[None, None, :])
    return norm_data1, norm_data2, norm_data3, d_min, d_max
