import csv
import json
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

sys.path.append('/content/drive/My Drive/FYP')
import date_handler as d_hand
import pandas as pd
import config as c
from config import Running_Colab as colab
if colab:
    import sys
    sys.path.append('/content/drive/My Drive/FYP')

def target_sentiment():
    if colab:
        tweets_df = pd.read_csv(c.ROOT_DIR_Google + 'normalised_tweet.csv')
        price_data = pd.read_csv(c.ROOT_DIR_Google + 'Tdata_out.csv')
    else:
        tweets_df = pd.read_csv('Tweets/normalised_tweet.csv')
        price_data = pd.read_csv('Technical_data/Tdata_out.csv')
    vect_1= tweets_df[['date_time','tweet']].values.tolist()
    vect_2 = price_data[['formatted_date','c_change']].values.tolist()
    vect_1 = [x + [0] for x in vect_1]
    vect_1 = [x + [0] for x in vect_1]
    print("Building sentiment...")
    i=0
    for date_, change in vect_2:
        for j in range(i, len(vect_1)):
            if (j+1) % 50000 == 0:
                print("Progress %d / %d...." % (j+1, len(vect_1)))
            # print("tweet date: "+vect_1[j][0]+" closing: "+ date_)
            if vect_1[j][0] < date_ and vect_1[j][0] > d_hand.previous_trading_date(date_):
                if float(change)>0:
                    vect_1[j][2] = 1
                else:
                    vect_1[j][2] = 0
                vect_1[j][3] = change
            elif vect_1[j][0]>date_:
                i=j
                break
    tweets_df['target']=[row[2] for row in vect_1]
    tweets_df['change']=[row[3] for row in vect_1]
    tweets_df.drop(tweets_df.loc[tweets_df['change']==0.0].index, inplace=True)
    if colab:
        tweets_df.to_csv(c.ROOT_DIR_Google+"sentiment.csv", index=False)
    else:
        tweets_df.to_csv("sentiment.csv", index=False)

def get_BOW():
    if colab:
        tweets_df = pd.read_csv(c.ROOT_DIR_Google + 'sentiment.csv')
    else:
        tweets_df = pd.read_csv('Tweets/sentiment.csv')
    word_iter, word_dict, prob_dict = {}, {}, {}
    positive_tweets, negative_tweets,total_tweets=0,0,0
    indexNames = tweets_df[tweets_df['date_time'] > c.PARAMETERS['max_training_date_tweets']].index
    tweets_df.drop(indexNames, inplace=True)
    list_tweet = tweets_df[["tweet","target","change"]].values.tolist()
    total_tweets = len(list_tweet)
    i=0
    print("Building BOW model...")
    for item in list_tweet:
        i+=1
        if i%10000==0:
            print("Progress %d / %d...." % (i, total_tweets))
        if item[1] > 0:
            positive_tweets+=1
        else:
            negative_tweets+=1
        words = item[0].split()
        for word in words:
            if word not in list(word_dict):
                word_dict[word] = {
                    'word': word,
                    'num_total': 2,
                    'num_positive': 1,
                    'num_negative': 1,
                    'mean': 0.0,
                    'std': 0.0,
                    'prob_positive': 0.0,
                    'prob_negative': 0.0}
                word_iter[word] = []

            word_dict[word]['num_total'] += 1

            if item[1] > 0:
                word_dict[word]['num_positive'] += 1
            else:
                word_dict[word]['num_negative'] += 1
            word_iter[word].append(item[2])
    #
    #
    print("Calculating statistics...")
    word_dict, prob_dict = compute_weights(arr=[word_dict, prob_dict, word_iter])
    # total words in pos/neg class
    total_pos = sum([word_dict[word]['num_positive'] for word in word_dict.keys()])
    total_neg = sum([word_dict[word]['num_negative'] for word in word_dict.keys()])

    prob_dict['priori_prob_pos'] = positive_tweets / (negative_tweets+ positive_tweets)
    prob_dict['priori_prob_neg'] = negative_tweets / (negative_tweets + positive_tweets)
    prob_dict['total_pos'] = total_pos
    prob_dict['total_neg'] = total_neg
    prob_dict['tweet_pos'] = positive_tweets
    prob_dict['tweet_neg'] = negative_tweets

    for word in word_dict.keys():
        priori_prob_word_pos = (1 / total_pos) * word_dict[word]['num_positive']
        priori_prob_word_neg = (1 / total_neg) * word_dict[word]['num_negative']

        word_dict[word]['prob_positive'] = priori_prob_word_pos
        word_dict[word]['prob_negative'] = priori_prob_word_neg

    arr = [word_dict,prob_dict]
    with open("Sentiment_data/BOW.json", 'w', encoding='utf-8') as file:
        for line in arr:
            json.dump(line, file)
            file.write('\n')
    file.close()

    return word_dict, prob_dict

def load_BOW():
    data_object=[]
    with open("Sentiment_data/BOW.json", 'r', encoding='utf-8') as file:
        for object in file:
            d = json.loads(object)
            data_object.append(d)
    wordDict,probDict = {},{}
    [wordDict, probDict] = data_object

    return  wordDict, probDict

def compute_weights(arr):
    [word_dict, prob_dict, word_iter] = arr

    for word, array in word_iter.items():
        sq_diff = 0
        mean = sum(array) / len(array)
        for val in array:
            sq_diff += ((val - mean) ** 2)

        if len(array) > 1:
            N = len(array) - 1
        elif len(array) == 0:
            print('error {}'.format(word))
            N = 1
        else:
            N = 1
        variance = sq_diff / N
        std = np.sqrt(variance)

        word_dict[word]['mean'] = mean
        word_dict[word]['std'] = std

    return word_dict, prob_dict

def compute_probabilities(tweet, BayesWordDict, ProbDict):
    p_tweet_pos = ProbDict['priori_prob_pos']
    p_tweet_neg = ProbDict['priori_prob_neg']

    # check
    p_pos_prod = 1
    p_neg_prod = 1
    p_pos_sum = 0
    p_neg_sum = 0
    p_pos_list, p_neg_list, p_list = [], [], [0.01, 0.01]

    # print('{}'.format(tfidf_vec))
    tweet = tweet.split()
    # words = row.split()
    j=0
    for word in tweet:
        # if word in words:
            try:
                p_pos_prod *= (BayesWordDict[word]['prob_positive'])
                p_neg_prod *= (BayesWordDict[word]['prob_negative'])
                p_pos_sum += (BayesWordDict[word]['prob_positive'])
                p_neg_sum += (BayesWordDict[word]['prob_negative'])
                if((BayesWordDict[word]['prob_positive'])>(BayesWordDict[word]['prob_negative'])):
                    p_list[0] += 1
                else:
                    p_list[1] += 1
            except KeyError:


                p_pos_prod *= (1 / ProbDict['total_pos'])
                p_neg_prod *= (1 / ProbDict['total_neg'])

                p_pos_sum += (1 / ProbDict['total_pos'])
                p_neg_sum += (1 / ProbDict['total_neg'])

    p_pos = p_tweet_pos * p_pos_prod
    p_neg = p_tweet_neg * p_neg_prod

    p_pos_mean = (p_pos_sum / len(tweet))
    p_neg_mean = (p_neg_sum / len(tweet))

    p_pos_y = p_tweet_pos * ((p_pos_mean) / (p_pos_mean + p_neg_mean))
    p_neg_y = p_tweet_neg * ((p_neg_mean) / (p_pos_mean + p_neg_mean))

    p_pos_adjusted = p_pos / (p_pos + p_neg)
    viterbi_prob = p_list[0] / sum(p_list)
    # print(p_list)
    return p_pos_adjusted, viterbi_prob, p_pos_y, p_neg_y

def tfidf_fit_all(arr):
    doc = ""
    docs =[]
    tfidfvec = TfidfVectorizer()
    print("Start fitting tfidfvect...")
    i=0
    for key in arr:
        i+=1
        if i%100==0:
            print("Progress %d / %d...." % (i, len(arr)))
        for text in arr[key]:
            doc += (" " + text[1])
        docs.append(doc)
    x = tfidfvec.fit_transform(docs)

    return tfidfvec, x

def tfidf_blob(key, text_blob, tfidfvec):
    blob = []
    for text in text_blob[key]:
        blob.append(text[1])
    x = tfidfvec.transform(blob)

    return x

def vectorize_tweets(arr, min_range, max_range):
    date_key = min_range
    tweet_day = []
    date_vector = {}
    print("Start vectorizing tweets...")
    for item in arr:
        if date_key > max_range:
            break
        elif item[2] < date_key and item[2] > d_hand.previous_trading_date(date_key):
            tweet_day.append(item)
        elif item[2] > date_key:
            date_vector[date_key]=tweet_day
            tweet_day=[]
            tweet_day.append(item)
            date_key=d_hand.next_trading_date(date_key)
    if len(tweet_day) >1:
        date_vector[date_key]=tweet_day

    return date_vector


# target_sentiment()
# word, prob = get_BOW()


