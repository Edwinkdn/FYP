import re
import pandas as pd
import twint
from bs4 import BeautifulSoup
import config as c
from nltk.tokenize import WordPunctTokenizer
import Tweets as tw
from datetime import date
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import nltk
from config import Running_Colab as colab
if colab:
    import sys
    sys.path.append('/content/drive/My Drive/FYP')
nltk.download('wordnet')
nltk.download(['punkt','stopwords'])

t_list = c.TWITTER_ACC_LIST
filename = c.PARAMETERS['filename']
min_tweet_date = str(c.PARAMETERS['min_tweet_date'])
max_tweet_date = str(c.PARAMETERS['max_tweet_date'])
market_opening = c.PARAMETERS['market_opening']
market_close = c.PARAMETERS['market_close']
time_offset = c.PARAMETERS['timezone_market']


def buildTweetsArchive():
    for acc in t_list:
        con = twint.Config()
        con.Username = acc
        con.Since = max_tweet_date
        con.Until = "2020-02-14"
        con.Store_csv = True
        if colab:
            con.Output = (c.ROOT_DIR_Google + filename)
        else:
            con.Output = filename
        twint.run.Search(con)

def cleanTweetsArchive():
    if colab:
        tweet_df = pd.read_csv(c.ROOT_DIR_Google + filename)
    else:
        tweet_df = pd.read_csv(filename)
    print(tweet_df.shape)
    tweet_df.drop(tweet_df.loc[tweet_df['retweet']=="True"].index, inplace=True)
    print(tweet_df.shape)
    tweet_df.drop_duplicates(subset="id",keep=False, inplace=True)
    print(tweet_df.shape)
    tweet_df.to_csv(filename, encoding='utf-8', index=False)
    print('Dataset size:', tweet_df.shape)
    print('Columns are:', tweet_df.columns)
    tweet_df.drop(['id', 'conversation_id', 'created_at', 'timezone', 'user_id', 'name', 'place', 'mentions', 'urls',
                   'photos', 'hashtags', 'cashtags', 'link', 'quote_url', 'video', 'near', 'geo', 'source',
                   'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src',
                   'trans_dest'], axis=1, inplace=True)
    print ("Cleaning the tweets...\n")
    clean_tweet_texts = []
    for i in range(0, len(tweet_df)):
        if (i + 1) % 50000 == 0:
            print ("Tweets %d of %d has been processed" % (i + 1, len(tweet_df)))
        clean_tweet_texts.append(tweet_cleaner(tweet_df['tweet'][i]))
    clean_tweet_df = pd.DataFrame(clean_tweet_texts, columns=['tweet'])
    clean_tweet_df['date_time'] = tweet_df.date + " " + tweet_df.time
    clean_tweet_df['user'] = tweet_df.username
    # clean_tweet_df['rt_count'] = tweet_df.retweets_count
    # clean_tweet_df['l_count'] = tweet_df.likes_count
    # clean_tweet_df['re_count'] = tweet_df.replies_count
    if colab:
        clean_tweet_df.to_csv(c.ROOT_DIR_Google + 'clean_tweet.csv', encoding='utf-8', index=True, index_label='index')
    else:
        clean_tweet_df.to_csv('clean_tweet.csv', encoding='utf-8', index=True, index_label='index')
    # check and remove na entries due to cleaning, the tweets doest not contain relevant text
    if colab:
        final_df = pd.read_csv(c.ROOT_DIR_Google + 'clean_tweet.csv')
    else:
        final_df = pd.read_csv('Tweets/clean_tweet.csv')
    final_df.dropna(inplace=True)
    final_df.drop(['index'], axis=1, inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    if colab:
        final_df.to_csv(c.ROOT_DIR_Google + 'clean_tweet.csv', encoding='utf-8', index_label='index')
    else:
        final_df.to_csv('clean_tweet.csv', encoding='utf-8', index_label='index')

    #normalize the tweets words for used in model later
    sort_clean_tweets()
    normalization()

def remove_stopwords(word_tokens):
    stop_words = stopwords.words('english')

    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    # print(word_tokens)
    # print(filtered_sentence)
    return filtered_sentence

def tweet_cleaner(text):
    token = WordPunctTokenizer()
    negations_dic = {"isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
                     "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
                     "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
                     "can't": "can not", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
                     "mustn't": "must not"}
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    combined_regrex = cleaner_regrex()
    stripped = re.sub(combined_regrex, '', bom_removed)
    # stripped = re.sub(combined_regrex2, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # clean = remove_stopwords(letters_only)
    words = [x for x  in token.tokenize(letters_only) if len(x) > 3]
    filtered_words = remove_stopwords(words)
    return (" ".join(filtered_words)).strip()

def cleaner_regrex():
    # regrex to remove the links in the tweets
    regrex_http = r'https?://[^ ]+'
    regrex_www = r'www.[^ ]+'
    regrex_pic_link = r'pic.twitter.com/[^ ]+'
    regrex_links = r'|'.join((regrex_http,regrex_www,regrex_pic_link))

    #regrex to remove the hashtag and tagging of other user @
    regrex_at = r'@[A-Za-z0-9_]+'
    regrex_hashtag =r'#[A-Za-z0-9_]+'
    regrex_tags = r'|'.join((regrex_at,regrex_hashtag))

    return r'|'.join((regrex_tags,regrex_links))

def normalization():
    token = WordPunctTokenizer()
    porter = nltk.PorterStemmer()
    if colab:
        norm_df = pd.read_csv(c.ROOT_DIR_Google + 'sorted_tweet.csv')
    else:
        norm_df = pd.read_csv('Tweets/sorted_tweet.csv')

    print("Normalising the tweets...\n")
    tweets_list = norm_df['tweet'].tolist()
    i=0
    norm = []
    for tweet in tweets_list:
        i+=1
        if i % 50000 == 0:
            print ("Tweets %d of %d has been processed" % (i, len(tweets_list)))
        words =[x for x in token.tokenize(tweet) if len(x)>3]
        normalized_tweet = []
        for word in words:
            normalized_text = porter.stem(word)
            normalized_tweet.append(normalized_text)
        norm.append((" ".join(normalized_tweet)).strip())
    norm_df['tweet']= norm
    if colab:
        norm_df.to_csv(c.ROOT_DIR_Google + 'normalised_tweet.csv', encoding='utf-8', index=False)
    else:
        norm_df.to_csv('normalised_tweet.csv', encoding='utf-8', index=False)


def sort_clean_tweets():
    if colab:
        sort_df = pd.read_csv(c.ROOT_DIR_Google + 'clean_tweet.csv')
    else:
        sort_df = pd.read_csv('Tweets/clean_tweet.csv')
    print("Sorting the tweets...\n")
    sort_df = sort_df.sort_values(by=['date_time'])
    if colab:
        sort_df.to_csv(c.ROOT_DIR_Google+'sorted_tweet.csv', encoding='utf-8', index_label='index')
    else:
        sort_df.to_csv('sorted_tweet.csv', encoding='utf-8', index_label='index')
    sort_df.drop(['index'], axis=1, inplace=True)
    sort_df.reset_index(drop=True, inplace=True)
    if colab:
        sort_df.to_csv(c.ROOT_DIR_Google + 'sorted_tweet.csv', encoding='utf-8', index_label='index')
    else:
        sort_df.to_csv('sorted_tweet.csv', encoding='utf-8', index_label='index')

def getWeeklyTweets():
    weekly_tweet = tw.TwitterData(str(date.today()))
    for acc in t_list:
        tweets = weekly_tweet.getData(acc)
    print(tweets)

if __name__ == '__main__':
    # buildTweetsArchive()
    cleanTweetsArchive()
    # getWeeklyTweets()
    # sort_clean_tweets()
    # normalization()