Running_Colab = False
ROOT_DIR_Google = "/content/drive/My Drive/FYP/"
# parameters for the data harvesting
PARAMETERS = {
    'filename': 'tweets.csv',
    'min_tweet_date': '2013-12-10',
    'max_tweet_date': '2020-02-03',
    'market_opening': 9,
    'market_close': 4,
    'timezone_market': -5,
    'timezone_tweets': 8,
    # 'max_training_date_tweets': "2019-03-15 16:00:00",
    'min_training_date_tweets': "2013-12-30 16:00:00",
    'max_training_date_tweets': "2019-07-03 16:00:00",
    'min_val_date_tweets': "2019-07-05 16:00:00",
    'max_val_date_tweets': "2019-10-23 16:00:00",
    'min_test_date_tweets': "2019-10-24 16:00:00",
    'max_test_date_tweets': "2020-02-13 16:00:00",
    # 'max_training_date_tweets': "2018-04-12 16:00:00",
    'price_property': 'close',
    'price_movement_upper_limit': 0.015,
    'num_lda_topics': 20,
    'ACCESS_TOKEN': '1188688517645225985-AfeoQHAWtL6hYjsb6OLEWUJl2OBAXn',
    'ACCESS_TOKEN_SECRET': 'Ve9PWZX6R2n6AV6EPEjl1OoXXcO3HG3fwkWfbiUyQ7ypX',
    'CONSUMER_KEY': '9uZntxLnw5kWH5MZ7FmFzyooA',
    'CONSUMER_SECRET': 'B4uL3uNFtx8yQxRTTadmlQIbNMnpoihOdbG5PpbsbCwDRpfhrt'
}

# twitter accounts shortlisted for data harvesting
TWITTER_ACC_LIST = ['Bloomberg', 'realDonaldTrump', 'AP_Europe', 'Schuldensuehner', 'LizAnnSonders',
                    'CiovaccoCapital', 'StockTwits', 'bespokeinvest', 'BBCBreaking', 'FXCM', 'WSJmarkets', 'ReutersUS',
                    'BBCWorld', 'AP', 'AP_Politics', 'ReutersBiz', 'business', 'LiveSquawk','markets','ReutersWorld']
#