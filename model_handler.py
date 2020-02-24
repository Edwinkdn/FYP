import json

from keras.engine.saving import load_model
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config as c
from config import Running_Colab as colab
if colab:
    import sys
    sys.path.append('/content/drive/My Drive/FYP')
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import numpy as np
from keras import backend as K, regularizers, optimizers, initializers
from matplotlib import pyplot as plt
import nlp_handler as nlp_h

# params = {
#     "batch_size": 32,
#     "epochs": 100,
#     "lr": 0.00500,
#     "time_steps": 10
# }
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back-1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def create_model_tech(x_t_shape, BATCH_SIZE, TIME_STEPS, lr, units, shuffle):
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    initi=initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None)
    lstm_model.add(LSTM(units[0], batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t_shape),
                        dropout=0.0, recurrent_dropout=0.0, stateful=shuffle, return_sequences=True, name='lstm_1', kernel_initializer=initi))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(LSTM(units[1], dropout=0.0, name='lstm_2'))
    lstm_model.add(Dropout(0.1))
    # lstm_model.add(Dense(units[2],activation='exponential', name='dense_1'))
    lstm_model.add(Dense(1,activation='tanh', name='dense_2'))
    optimizer = optimizers.RMSprop(lr=lr)
    # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    return lstm_model

def create_model_senti(x_t_shape, BATCH_SIZE, TIME_STEPS, lr, units, shuffle):
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    initi=initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None)
    lstm_model.add(LSTM(units[0], batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t_shape),
                        dropout=0.0, recurrent_dropout=0.0, stateful=shuffle, return_sequences=True,
                        kernel_initializer=initi, name='lstm_1'))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(LSTM(units[1], dropout=0.0, name='lstm_2'))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(Dense(1,activation='tanh', name='dense_2'))
    optimizer = optimizers.RMSprop(lr=lr)
    # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)

    lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')
    return lstm_model

def trim_dataset(mat,batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if no_of_rows_drop > 0:
        # print(no_of_rows_drop)
        return mat[:-no_of_rows_drop]
    else:
        return mat

def classified_labels(arr):
    for row in arr:
        if row[0]>0:
            row[0] = 1
        else:
            row[0] = 0

    arr=np.asarray(arr)
    return arr

def run_technical_model(runs, BATCH_SIZE, TIME_STEPS, epochs, lr, units):
    acc_=[]
    rsme=[]
    if colab:
        dataset=pd.read_csv(c.ROOT_DIR_Google+"Tdata_out.csv",index_col="formatted_date")
    else:
        dataset=pd.read_csv("Technical_data/Tdata_out.csv", index_col="formatted_date")

    # load the dataset
    data = dataset.values
    # split into train and test sets
    train_size = int(len(data) * 0.70)

    val_size = int((len(data) - train_size)/2)
    train, val, test = data[0:train_size, :], data[train_size:(len(data)-val_size), :] , data[(len(data)-val_size):len(data), :]
    # print("train: {}, val: {}, test: {}".format(train.shape, val.shape, test.shape))

    scaler = MinMaxScaler(feature_range = (-1, 1))

    scaler_single = MinMaxScaler(feature_range = (-1, 1))

    train = np.concatenate([scaler_single.fit_transform(train[:,0].reshape(-1,1)),
                                      scaler.fit_transform(train[:,1:])], axis = 1)
    val = np.concatenate([scaler_single.fit_transform(val[:, 0].reshape(-1, 1)),
                           scaler.fit_transform(val[:, 1:])], axis=1)
    test = np.concatenate([scaler_single.fit_transform(test[:,0].reshape(-1,1)),
                                      scaler.fit_transform(test[:,1:])], axis = 1)
    # # print(train[0:15])
    # # reshape into X=t and Y=t+1
    look_back = TIME_STEPS
    trainX, trainY = create_dataset(train, look_back)

    # print(trainY[1])
    valX, valY = create_dataset(val, look_back)
    testX, testY = create_dataset(test, look_back)
    # print("train: {}, val: {}, test: {}".format(trainX.shape, valX.shape, testX.shape))
    x_t = trim_dataset(trainX, BATCH_SIZE)
    y_t = trim_dataset(trainY, BATCH_SIZE)
    x_v = trim_dataset(valX, BATCH_SIZE)
    y_v = trim_dataset(valY, BATCH_SIZE)
    x_test = trim_dataset(testX, BATCH_SIZE)
    y_test = trim_dataset(testY, BATCH_SIZE)
    # print("train: {}, val: {}, test: {}".format(y_t.shape, y_v.shape, y_test.shape))
    for i in range(runs):
        # model1 = create_model_senti(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, False)
        # history1 = model1.fit(x_t, y_t, epochs=epochs, verbose=0, batch_size=BATCH_SIZE,
        #                     shuffle=True, validation_data=(x_v, y_v))
        # model1.save_weights("weights_tech.h5")
        model = create_model_tech(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, True)
        # model.load_weights("weights_tech.h5", by_name=True, skip_mismatch=True)
        history = model.fit(x_t, y_t, epochs=100, verbose=0, batch_size=BATCH_SIZE,
                                shuffle=False, validation_data=(x_v,y_v))
        # model.save('lstm_model.h5')
        # Plot training
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['loss', 'validation'], loc='upper right')
        # plt.show()
        # # #
        # # make predictions
        # model = load_model('lstm_model.h5')
        y_pred = model.predict(trim_dataset(x_test,BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred = y_pred.flatten()
        y_test_t = trim_dataset(y_test, BATCH_SIZE)
        # print(y_test_t)
        # print(y_test_t.shape)
        # print(y_test_t[0:15])

        y_pred = scaler_single.inverse_transform(y_pred.reshape(-1,1))
        y_test_t = scaler_single.inverse_transform(y_test_t.reshape(-1,1))
        error = mean_squared_error(y_test_t, y_pred)
        # print(y_pred)
        # # print(y_test_t)
        true_positive=0
        true_negative=0
        for i in range (len(y_pred)):
            if y_pred[i]>0 and y_test_t[i]>0:
                true_positive +=1
            elif y_pred[i]<0 and y_test_t[i]<0:
                true_negative+=1
        accuracy = (true_negative+true_positive)/(len(y_pred))*100
        print(accuracy)
        acc_.append(accuracy)
        print(trainY.shape)
        # calculate root mean squared error
        # trainScore = np.math.sqrt(y_pred)
        # print('Train Score: %.6f RMSE' % (trainScore))
        testScore = np.math.sqrt(error)
        print('Test Score: %.6f RMSE' % (testScore))
        rsme.append(testScore)
        plt.plot(y_test_t)
        plt.plot(y_pred)
        plt.title('model predictions, lr= %0.4f, units[%d,%d,%d]' %(lr, units[0], units[1], units[2]))
        plt.ylabel('close change')
        plt.xlabel('dates')
        plt.legend(['real', 'predict'], loc='upper right')
        plt.grid(True)
        plt.show()
    print(max(acc_))
    print(np.mean(acc_))
    print(min(rsme))
    print(np.mean(rsme))

    return max(acc_), np.mean(acc_), min(rsme), np.mean(rsme)

def setup_sentiment_analyser(load=True):
    # _ = nlp_h.target_sentiment()
    if load:
        word_dict, prob_dict = nlp_h.load_BOW()
    else:
        word_dict, prob_dict = nlp_h.get_BOW()
    if colab:
        tweet_=pd.read_csv(c.ROOT_DIR_Google+"sentiment.csv").values.tolist()
    else:
        tweet_=pd.read_csv("Tweets/sentiment.csv").values.tolist()
    tweet_train = nlp_h.vectorize_tweets(tweet_, "2013-12-30 16:00:00", "2018-04-12 16:00:00")
    if load:
        xt_train = pd.read_csv("Sentiment_data/xt_train.csv")["day_string"].values.tolist()
    else:
        tfidfvec1, tweet_train1 = nlp_h.tfidf_fit_all(tweet_train)
        xt_train =[]
        for vec in tweet_train1:
            df = pd.DataFrame(vec.T.todense(),
                          columns=["tfidf"])
            df["words"] = tfidfvec1.get_feature_names()
            df = df[df.tfidf != 0.0]
            df.sort_values(by=["tfidf"], ascending=False)
            day_words = df["words"].values.tolist()
            day_string = (" ".join(day_words))
            xt_train.append(day_string)
        df = pd.DataFrame(xt_train, columns=["day_string"])
        df.to_csv('xt_train.csv')
        xt_train = df["day_string"].values.tolist
    # # print(len(xt_test[0]))
    tweet_val = nlp_h.vectorize_tweets(tweet_, "2018-04-13 16:00:00", "2019-03-15 16:00:00")

    if load:
        xt_val = pd.read_csv("Sentiment_data/xt_val.csv")["day_string"].values.tolist()
    else:
        tfidfvec2, tweet_val1 = nlp_h.tfidf_fit_all(tweet_val)
        xt_val =[]
        for vec in tweet_val1:
            df = pd.DataFrame(vec.T.todense(),
                          columns=["tfidf"])
            df["words"] = tfidfvec2.get_feature_names()
            df = df[df.tfidf != 0.0]
            df.sort_values(by=["tfidf"], ascending=False)
            day_words = df["words"].values.tolist()
            day_string = (" ".join(day_words))
            xt_val.append(day_string)
        df = pd.DataFrame(xt_val, columns=["day_string"])
        df.to_csv('xt_val.csv')
        xt_val = df["day_string"].values.tolist
    tweet_test = nlp_h.vectorize_tweets(tweet_, "2019-03-18 16:00:00", "2020-02-13 16:00:00")
    if load:
        xt_test = pd.read_csv("Sentiment_data/xt_test.csv")["day_string"].values.tolist()
    else:
        tfidfvec3, tweet_test1 = nlp_h.tfidf_fit_all(tweet_test)
        xt_test =[]
        for vec in tweet_test1:
            df = pd.DataFrame(vec.T.todense(), columns=["tfidf"])
            df["words"] = tfidfvec3.get_feature_names()
            df = df[df.tfidf != 0.0]
            df.sort_values(by=["tfidf"], ascending=False)
            day_words = df["words"].values.tolist()
            day_string = (" ".join(day_words))
            xt_test.append(day_string)
        df = pd.DataFrame(xt_test, columns=["day_string"])
        df.to_csv('xt_test.csv')
        xt_test = df["day_string"].values.tolist

    return word_dict, prob_dict,xt_train, xt_val, xt_test, tweet_train, tweet_val, tweet_test

# def run_mean_model(runs, word_dict, prob_dict, xt_train, xt_val, xt_test, tweet_train, tweet_val, tweet_test):
#     BATCH_SIZE = 10
#     TIME_STEPS = 5
#     epochs = 100
#     lr = 0.003
#     units = [140, 80, 30]
#     acc_ = []
#     rsme = []
#     if colab:
#         dataset = pd.read_csv(c.ROOT_DIR_Google + "Tdata_out.csv", index_col="formatted_date")
#     else:
#         dataset = pd.read_csv("Tdata_out.csv", index_col="formatted_date")
#
#     # load the dataset
#     data = dataset.values
#     # split into train and test sets
#     train_size = int(len(data) * 0.70)
#
#     val_size = int((len(data) - train_size) / 2)
#     train, val, test = data[0:train_size, :], data[train_size:(len(data) - val_size), :], data[
#                                                                                           (len(
#                                                                                               data) - val_size):len(
#                                                                                               data), :]
#     # print("train: {}, val: {}, test: {}".format(train.shape, val.shape, test.shape))
#
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaler_single = MinMaxScaler(feature_range=(-1, 1))
#
#     train = np.concatenate([scaler_single.fit_transform(train[:, 0].reshape(-1, 1)),
#                             scaler.fit_transform(train[:, 1:])], axis=1)
#     val = np.concatenate([scaler_single.fit_transform(val[:, 0].reshape(-1, 1)),
#                           scaler.fit_transform(val[:, 1:])], axis=1)
#     test = np.concatenate([scaler_single.fit_transform(test[:, 0].reshape(-1, 1)),
#                            scaler.fit_transform(test[:, 1:])], axis=1)
#     # # print(train[0:15])
#
#     look_back = TIME_STEPS
#     xt_train_prob = []
#     i = 0
#     for key in tweet_train:
#         if i % 100 == 0:
#             print("Progress %d / %d...." % (i, len(tweet_train)))
#         pos_prob,neg_prob = [],[]
#         for tweet in tweet_train[key]:
#             _, _, pos_mean, neg_mean = nlp_h.compute_probabilities(xt_train[i], tweet[1], word_dict, prob_dict)
#             pos_prob.append(pos_mean)
#             neg_prob.append(neg_mean)
#         pos_prob = np.mean(pos_prob)
#         neg_prob = np.mean(neg_prob)
#         xt_train_prob.append([pos_prob,neg_prob])
#         i += 1
#     xt_train_prob = np.asarray(xt_train_prob)
#     scaler_senti = MinMaxScaler(feature_range=(-1, 1))
#     xt_train_prob=scaler_senti.fit_transform(xt_train_prob)
#     train = np.concatenate([train, xt_train_prob], axis=1)
#     trainX, trainY = create_dataset(train, look_back)
#     xt_val_prob = []
#     i = 0
#     for key in tweet_val:
#         if i % 100 == 0:
#             print("Progress %d / %d...." % (i, len(tweet_val)))
#         pos_prob, neg_prob = [], []
#         for tweet in tweet_val[key]:
#             _, _, pos_mean, neg_mean = nlp_h.compute_probabilities(xt_val[i], tweet[1], word_dict, prob_dict)
#             pos_prob.append(pos_mean)
#             neg_prob.append(neg_mean)
#         pos_prob = np.mean(pos_prob)
#         neg_prob = np.mean(neg_prob)
#         xt_val_prob.append([pos_prob, neg_prob])
#         i += 1
#     xt_val_prob = np.asarray(xt_val_prob)
#     scaler_senti = MinMaxScaler(feature_range=(-1, 1))
#     xt_val_prob=scaler_senti.fit_transform(xt_val_prob)
#     val = np.concatenate([val, xt_val_prob], axis=1)
#     valX, valY = create_dataset(val, look_back)
#     xt_test_prob = []
#     i = 0
#     for key in tweet_test:
#         if i % 100 == 0:
#             print("Progress %d / %d...." % (i, len(tweet_test)))
#         pos_prob, neg_prob = [], []
#         for tweet in tweet_test[key]:
#             _, _, pos_mean, neg_mean = nlp_h.compute_probabilities(xt_test[i], tweet[1], word_dict, prob_dict)
#             pos_prob.append(pos_mean)
#             neg_prob.append(neg_mean)
#         pos_prob = np.mean(pos_prob)
#         neg_prob = np.mean(neg_prob)
#         xt_test_prob.append([pos_prob, neg_prob])
#         i += 1
#     xt_test_prob = np.asarray(xt_test_prob)
#     scaler_senti = MinMaxScaler(feature_range=(-1, 1))
#     xt_test_prob=scaler_senti.fit_transform(xt_test_prob)
#     test = np.concatenate([test, xt_test_prob], axis=1)
#     testX, testY = create_dataset(test, look_back)
#     x_t = trim_dataset(trainX, BATCH_SIZE)
#     y_t = trim_dataset(trainY, BATCH_SIZE)
#     x_v = trim_dataset(valX, BATCH_SIZE)
#     y_v = trim_dataset(valY, BATCH_SIZE)
#     x_test = trim_dataset(testX, BATCH_SIZE)
#     y_test = trim_dataset(testY, BATCH_SIZE)
#     for i in range(runs):
#         model = create_model_tech(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, True)
#         history = model.fit(x_t, y_t, epochs=epochs, verbose=0, batch_size=BATCH_SIZE,
#                             shuffle=False, validation_data=(x_v, y_v))
#         # model.save('lstm_model.h5')
#         # Plot training
#         plt.plot(history.history['loss'])
#         plt.plot(history.history['val_loss'])
#         plt.title('model loss')
#         plt.ylabel('loss')
#         plt.xlabel('epoch')
#         plt.legend(['loss', 'validation'], loc='upper right')
#         plt.show()
#         # # #
#         # # make predictions
#         # model = load_model('lstm_model.h5')
#         y_pred = model.predict(trim_dataset(x_test, BATCH_SIZE), batch_size=BATCH_SIZE)
#         y_pred = y_pred.flatten()
#         y_test_t = trim_dataset(y_test, BATCH_SIZE)
#         # print(y_test_t)
#         # print(y_test_t.shape)
#         # print(y_test_t[0:15])
#
#         y_pred = scaler_single.inverse_transform(y_pred.reshape(-1, 1))
#         y_test_t = scaler_single.inverse_transform(y_test_t.reshape(-1, 1))
#         error = mean_squared_error(y_test_t, y_pred)
#         # print(y_pred)
#         # # print(y_test_t)
#         true_positive = 0
#         true_negative = 0
#         for i in range(len(y_pred)):
#             if y_pred[i] > 0 and y_test_t[i] > 0:
#                 true_positive += 1
#             elif y_pred[i] < 0 and y_test_t[i] < 0:
#                 true_negative += 1
#         accuracy = (true_negative + true_positive) / (len(y_pred)) * 100
#         print(accuracy)
#         acc_.append(accuracy)
#         print(trainY.shape)
#         # calculate root mean squared error
#         # trainScore = np.math.sqrt(y_pred)
#         # print('Train Score: %.6f RMSE' % (trainScore))
#         testScore = np.math.sqrt(error)
#         print('Test Score: %.6f RMSE' % (testScore))
#         rsme.append(testScore)
#         plt.plot(y_test_t)
#         plt.plot(y_pred)
#         plt.title('model predictions, lr= %0.4f, units[%d,%d,%d]' % (lr, units[0], units[1], units[2]))
#         plt.ylabel('close change')
#         plt.xlabel('dates')
#         plt.legend(['real', 'predict'], loc='upper right')
#         plt.show()
#     print(max(acc_))
#     print(np.mean(acc_))
#     print(min(rsme))
#     print(np.mean(rsme))
#
# def run_hyp_vit_model(runs, word_dict, prob_dict, xt_train, xt_val, xt_test, tweet_train, tweet_val, tweet_test):
#     BATCH_SIZE = 10
#     TIME_STEPS = 5
#     epochs = 100
#     lr = 0.003
#     units = [140, 80, 30]
#     acc_ = []
#     rsme = []
#     if colab:
#         dataset = pd.read_csv(c.ROOT_DIR_Google + "Tdata_out.csv", index_col="formatted_date")
#     else:
#         dataset = pd.read_csv("Tdata_out.csv", index_col="formatted_date")
#
#     # load the dataset
#     data = dataset.values
#     # split into train and test sets
#     train_size = int(len(data) * 0.70)
#
#     val_size = int((len(data) - train_size) / 2)
#     train, val, test = data[0:train_size, :], data[train_size:(len(data) - val_size), :], data[
#                                                                                           (len(
#                                                                                               data) - val_size):len(
#                                                                                               data), :]
#     # print("train: {}, val: {}, test: {}".format(train.shape, val.shape, test.shape))
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaler_single = MinMaxScaler(feature_range=(-1, 1))
#
#     train = np.concatenate([scaler_single.fit_transform(train[:, 0].reshape(-1, 1)),
#                             scaler.fit_transform(train[:, 1:])], axis=1)
#     val = np.concatenate([scaler_single.fit_transform(val[:, 0].reshape(-1, 1)),
#                           scaler.fit_transform(val[:, 1:])], axis=1)
#     test = np.concatenate([scaler_single.fit_transform(test[:, 0].reshape(-1, 1)),
#                            scaler.fit_transform(test[:, 1:])], axis=1)
#     look_back = TIME_STEPS
#     xt_train_prob = []
#     i = 0
#     for key in tweet_train:
#         if i % 100 == 0:
#             print("Progress %d / %d...." % (i, len(tweet_train)))
#         prob_, vit_prob_ = [], []
#         for tweet in tweet_train[key]:
#             prob, vit_prob, pos_mean, neg_mean = nlp_h.compute_probabilities(xt_train[i], tweet[1], word_dict, prob_dict)
#             prob_.append(prob)
#             vit_prob_.append(vit_prob)
#         prob_ = np.mean(prob_)
#         vit_prob_ = np.mean(vit_prob_)
#         xt_train_prob.append([prob_, vit_prob_])
#         i += 1
#     xt_train_prob = np.asarray(xt_train_prob)
#     scaler_senti = MinMaxScaler(feature_range=(-1, 1))
#     xt_train_prob = scaler_senti.fit_transform(xt_train_prob)
#     train = np.concatenate([train, xt_train_prob], axis=1)
#     trainX, trainY = create_dataset(train, look_back)
#     xt_val_prob = []
#     i = 0
#     for key in tweet_val:
#         if i % 100 == 0:
#             print("Progress %d / %d...." % (i, len(tweet_val)))
#         prob_, vit_prob_ = [], []
#         for tweet in tweet_val[key]:
#             prob, vit_prob, pos_mean, neg_mean = nlp_h.compute_probabilities(xt_val[i], tweet[1], word_dict, prob_dict)
#             prob_.append(prob)
#             vit_prob_.append(vit_prob)
#         prob_ = np.mean(prob_)
#         vit_prob_ = np.mean(vit_prob_)
#         xt_val_prob.append([prob_, vit_prob_])
#         i += 1
#     xt_val_prob = np.asarray(xt_val_prob)
#     scaler_senti = MinMaxScaler(feature_range=(-1, 1))
#     xt_val_prob = scaler_senti.fit_transform(xt_val_prob)
#     val = np.concatenate([val, xt_val_prob], axis=1)
#     valX, valY = create_dataset(val, look_back)
#     xt_test_prob = []
#     i = 0
#     for key in tweet_test:
#         if i % 100 == 0:
#             print("Progress %d / %d...." % (i, len(tweet_test)))
#         prob_, vit_prob_ = [], []
#         for tweet in tweet_test[key]:
#             prob, vit_prob, pos_mean, neg_mean = nlp_h.compute_probabilities(xt_test[i], tweet[1], word_dict, prob_dict)
#             prob_.append(prob)
#             vit_prob_.append(vit_prob)
#         prob_ = np.mean(prob_)
#         vit_prob_ = np.mean(vit_prob_)
#         xt_test_prob.append([prob_, vit_prob_])
#         i += 1
#     xt_test_prob = np.asarray(xt_test_prob)
#     scaler_senti = MinMaxScaler(feature_range=(-1, 1))
#     xt_test_prob = scaler_senti.fit_transform(xt_test_prob)
#     test = np.concatenate([test, xt_test_prob], axis=1)
#     testX, testY = create_dataset(test, look_back)
#     x_t = trim_dataset(trainX, BATCH_SIZE)
#     y_t = trim_dataset(trainY, BATCH_SIZE)
#     x_v = trim_dataset(valX, BATCH_SIZE)
#     y_v = trim_dataset(valY, BATCH_SIZE)
#     x_test = trim_dataset(testX, BATCH_SIZE)
#     y_test = trim_dataset(testY, BATCH_SIZE)
#
#     for i in range(runs):
#         model = create_model_tech(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, True)
#         history = model.fit(x_t, y_t, epochs=epochs, verbose=0, batch_size=BATCH_SIZE,
#                             shuffle=False, validation_data=(x_v, y_v))
#         # model.save('lstm_model.h5')
#         # Plot training
#         plt.plot(history.history['loss'])
#         plt.plot(history.history['val_loss'])
#         plt.title('model loss')
#         plt.ylabel('loss')
#         plt.xlabel('epoch')
#         plt.legend(['loss', 'validation'], loc='upper right')
#         plt.show()
#         # # #
#         # # make predictions
#         # model = load_model('lstm_model.h5')
#         y_pred = model.predict(trim_dataset(x_test, BATCH_SIZE), batch_size=BATCH_SIZE)
#         y_pred = y_pred.flatten()
#         y_test_t = trim_dataset(y_test, BATCH_SIZE)
#         # print(y_test_t)
#         # print(y_test_t.shape)
#         # print(y_test_t[0:15])
#
#         y_pred = scaler_single.inverse_transform(y_pred.reshape(-1, 1))
#         y_test_t = scaler_single.inverse_transform(y_test_t.reshape(-1, 1))
#         error = mean_squared_error(y_test_t, y_pred)
#         # print(y_pred)
#         # # print(y_test_t)
#         true_positive = 0
#         true_negative = 0
#         for i in range(len(y_pred)):
#             if y_pred[i] > 0 and y_test_t[i] > 0:
#                 true_positive += 1
#             elif y_pred[i] < 0 and y_test_t[i] < 0:
#                 true_negative += 1
#         accuracy = (true_negative + true_positive) / (len(y_pred)) * 100
#         print(accuracy)
#         acc_.append(accuracy)
#         print(trainY.shape)
#         # calculate root mean squared error
#         # trainScore = np.math.sqrt(y_pred)
#         # print('Train Score: %.6f RMSE' % (trainScore))
#         testScore = np.math.sqrt(error)
#         print('Test Score: %.6f RMSE' % (testScore))
#         rsme.append(testScore)
#         plt.plot(y_test_t)
#         plt.plot(y_pred)
#         plt.title('model predictions, lr= %0.4f, units[%d,%d,%d]' % (lr, units[0], units[1], units[2]))
#         plt.ylabel('close change')
#         plt.xlabel('dates')
#         plt.legend(['real', 'predict'], loc='upper right')
#         plt.show()
#     print(max(acc_))
#     print(np.mean(acc_))
#     print(min(rsme))
#     print(np.mean(rsme))

def _f1collect(score,p,label):
    if p >= 0.5:
        if label > 0:
            score['tp']+=1
        else:
            score['fn']+=1
    else:
        if label > 0:
            score['fp']+=1
        else:
            score['tn']+=1
    return score

def get_probabilities(word_dict, prob_dict, xt_train, xt_val, xt_test, tweet_train, tweet_val, tweet_test):
    tr, val, test = [], [], []
    i = 0
    for key in tweet_train:
        if i % 100 == 0:
            print("Progress %d / %d...." % (i, len(tweet_train)))
        prob_, vit_prob_, pos_prob, neg_prob = [], [], [], []
        for tweet in tweet_train[key]:
            prob, vit_prob, pos_mean, neg_mean = nlp_h.compute_probabilities(xt_train[i], tweet[1], word_dict,
                                                                             prob_dict)
            prob_.append(prob)
            vit_prob_.append(vit_prob)
            pos_prob.append(pos_mean)
            neg_prob.append(neg_mean)
        prob_ = np.mean(prob_)
        vit_prob_ = np.mean(vit_prob_)
        pos_prob = np.mean(pos_prob)
        neg_prob = np.mean(pos_prob)
        tr.append([prob_,vit_prob_,pos_prob,neg_prob])
        i += 1
    i=0
    for key in tweet_val:
        if i % 100 == 0:
            print("Progress %d / %d...." % (i, len(tweet_val)))
        prob_, vit_prob_, pos_prob, neg_prob = [], [], [], []
        for tweet in tweet_val[key]:
            prob, vit_prob, pos_mean, neg_mean = nlp_h.compute_probabilities(xt_val[i], tweet[1], word_dict,
                                                                             prob_dict)
            prob_.append(prob)
            vit_prob_.append(vit_prob)
            pos_prob.append(pos_mean)
            neg_prob.append(neg_mean)
        prob_ = np.mean(prob_)
        vit_prob_ = np.mean(vit_prob_)
        pos_prob = np.mean(pos_prob)
        neg_prob = np.mean(pos_prob)
        val.append([prob_,vit_prob_,pos_prob,neg_prob])
        i += 1
    i=0
    for key in tweet_test:
        if i % 100 == 0:
            print("Progress %d / %d...." % (i, len(tweet_test)))
        prob_, vit_prob_, pos_prob, neg_prob = [], [], [], []
        for tweet in tweet_test[key]:
            prob, vit_prob, pos_mean, neg_mean = nlp_h.compute_probabilities(xt_test[i], tweet[1], word_dict,
                                                                             prob_dict)
            prob_.append(prob)
            vit_prob_.append(vit_prob)
            pos_prob.append(pos_mean)
            neg_prob.append(neg_mean)
        prob_ = np.mean(prob_)
        vit_prob_ = np.mean(vit_prob_)
        pos_prob = np.mean(pos_prob)
        neg_prob = np.mean(pos_prob)
        test.append([prob_,vit_prob_,pos_prob,neg_prob])
        i += 1

        df_train = pd.DataFrame(tr, columns=['hyper', 'viterbi', 'pos_mean', 'neg_mean'])
        df_train.to_csv("train_prob.csv", index=False)
        df_val = pd.DataFrame(val, columns=['hyper', 'viterbi', 'pos_mean', 'neg_mean'])
        df_val.to_csv("val_prob.csv", index=False)
        df_test = pd.DataFrame(test, columns=['hyper', 'viterbi', 'pos_mean', 'neg_mean'])
        df_test.to_csv("test_prob.csv", index=False)

def run_mean_model(runs):
    BATCH_SIZE = 10
    TIME_STEPS = 5
    epochs = 100
    lr = 0.003
    units = [80,160, 30]
    acc_ = []
    rsme = []
    if colab:
        dataset = pd.read_csv(c.ROOT_DIR_Google + "Tdata_out.csv", index_col="formatted_date")
    else:
        dataset = pd.read_csv("Technical_data/Tdata_out.csv", index_col="formatted_date")

    # load the dataset
    data = dataset.values
    # split into train and test sets
    train_size = int(len(data) * 0.70)

    val_size = int((len(data) - train_size) / 2)
    train, val, test = data[0:train_size, :], data[train_size:(len(data) - val_size), :], data[
                                                                                          (len(
                                                                                              data) - val_size):len(
                                                                                              data), :]
    # print("train: {}, val: {}, test: {}".format(train.shape, val.shape, test.shape))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_single = MinMaxScaler(feature_range=(-1, 1))

    train = np.concatenate([scaler_single.fit_transform(train[:, 0].reshape(-1, 1)),
                            scaler.fit_transform(train[:, 1:])], axis=1)
    val = np.concatenate([scaler_single.fit_transform(val[:, 0].reshape(-1, 1)),
                          scaler.fit_transform(val[:, 1:])], axis=1)
    test = np.concatenate([scaler_single.fit_transform(test[:, 0].reshape(-1, 1)),
                           scaler.fit_transform(test[:, 1:])], axis=1)
    # # print(train[0:15])

    look_back = TIME_STEPS
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "train_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/train_prob.csv")
    xt_train_prob = prob[['pos_mean','neg_mean']].values.tolist()
    scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_train_prob=scaler_senti.fit_transform(xt_train_prob)
    train = np.concatenate([train, xt_train_prob], axis=1)
    trainX, trainY = create_dataset(train, look_back)
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "val_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/val_prob.csv")
    xt_val_prob = prob[['pos_mean','neg_mean']].values.tolist()
    # scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_val_prob=scaler_senti.transform(xt_val_prob)
    val = np.concatenate([val, xt_val_prob], axis=1)
    valX, valY = create_dataset(val, look_back)
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "test_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/test_prob.csv")
    xt_test_prob = prob[['pos_mean','neg_mean']].values.tolist()
    # scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_test_prob=scaler_senti.transform(xt_test_prob)
    test = np.concatenate([test, xt_test_prob], axis=1)
    testX, testY = create_dataset(test, look_back)
    x_t = trim_dataset(trainX, BATCH_SIZE)
    y_t = trim_dataset(trainY, BATCH_SIZE)
    x_v = trim_dataset(valX, BATCH_SIZE)
    y_v = trim_dataset(valY, BATCH_SIZE)
    x_test = trim_dataset(testX, BATCH_SIZE)
    y_test = trim_dataset(testY, BATCH_SIZE)
    for i in range(runs):
        # model1 = create_model_senti(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, False)
        # history1 = model1.fit(x_t, y_t, epochs=100, verbose=0, batch_size=BATCH_SIZE,
        #                       shuffle=True, validation_data=(x_v, y_v))
        # model1.save_weights("weights.h5")
        model = create_model_tech(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, True)
        # model.load_weights("weights.h5", by_name=True, skip_mismatch=True)
        history = model.fit(x_t, y_t, epochs=100, verbose=0, batch_size=BATCH_SIZE,
                            shuffle=False, validation_data=(x_v, y_v))
        # model.save('lstm_model.h5')
        # Plot training
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['loss', 'validation'], loc='upper right')
        # plt.show()
        # # #
        # # make predictions
        # model = load_model('lstm_model.h5')
        y_pred = model.predict(trim_dataset(x_test, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred = y_pred.flatten()
        y_test_t = trim_dataset(y_test, BATCH_SIZE)
        # print(y_test_t)
        # print(y_test_t.shape)
        # print(y_test_t[0:15])

        y_pred = scaler_single.inverse_transform(y_pred.reshape(-1, 1))
        y_test_t = scaler_single.inverse_transform(y_test_t.reshape(-1, 1))
        error = mean_squared_error(y_test_t, y_pred)
        # print(y_pred)
        # # print(y_test_t)
        true_positive = 0
        true_negative = 0
        for i in range(len(y_pred)):
            if y_pred[i] > 0 and y_test_t[i] > 0:
                true_positive += 1
            elif y_pred[i] < 0 and y_test_t[i] < 0:
                true_negative += 1
        accuracy = (true_negative + true_positive) / (len(y_pred)) * 100
        print(accuracy)
        acc_.append(accuracy)
        print(trainY.shape)
        # calculate root mean squared error
        # trainScore = np.math.sqrt(y_pred)
        # print('Train Score: %.6f RMSE' % (trainScore))
        testScore = np.math.sqrt(error)
        print('Test Score: %.6f RMSE' % (testScore))
        rsme.append(testScore)
        plt.plot(y_test_t)
        plt.plot(y_pred)
        plt.title('model predictions, lr= %0.4f, units[%d,%d,%d]' % (lr, units[0], units[1], units[2]))
        plt.ylabel('close change')
        plt.xlabel('dates')
        plt.legend(['real', 'predict'], loc='upper right')
        plt.grid(True)
        plt.show()
    print(max(acc_))
    print(np.mean(acc_))
    print(min(rsme))
    print(np.mean(rsme))

    return max(acc_), np.mean(acc_), min(rsme), np.mean(rsme)

def run_hyp_model(runs):
    BATCH_SIZE = 10
    TIME_STEPS = 5
    epochs = 100
    lr = 0.003
    units = [80,160, 30]
    acc_ = []
    rsme = []
    if colab:
        dataset = pd.read_csv(c.ROOT_DIR_Google + "Tdata_out.csv", index_col="formatted_date")
    else:
        dataset = pd.read_csv("Technical_data/Tdata_out.csv", index_col="formatted_date")

    # load the dataset
    data = dataset.values
    # split into train and test sets
    train_size = int(len(data) * 0.70)

    val_size = int((len(data) - train_size) / 2)
    train, val, test = data[0:train_size, :], data[train_size:(len(data) - val_size), :], data[
                                                                                          (len(
                                                                                              data) - val_size):len(
                                                                                              data), :]
    # print("train: {}, val: {}, test: {}".format(train.shape, val.shape, test.shape))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_single = MinMaxScaler(feature_range=(-1, 1))

    train = np.concatenate([scaler_single.fit_transform(train[:, 0].reshape(-1, 1)),
                            scaler.fit_transform(train[:, 1:])], axis=1)
    val = np.concatenate([scaler_single.fit_transform(val[:, 0].reshape(-1, 1)),
                          scaler.fit_transform(val[:, 1:])], axis=1)
    test = np.concatenate([scaler_single.fit_transform(test[:, 0].reshape(-1, 1)),
                           scaler.fit_transform(test[:, 1:])], axis=1)
    # # print(train[0:15])

    look_back = TIME_STEPS
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "train_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/train_prob.csv")
    xt_train_prob = prob['hyper'].values
    scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_train_prob = scaler_senti.fit_transform(xt_train_prob.reshape(-1,1))
    train = np.concatenate([train, xt_train_prob], axis=1)
    trainX, trainY = create_dataset(train, look_back)
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "val_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/val_prob.csv")
    xt_val_prob = prob['hyper'].values
    # scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_val_prob = scaler_senti.transform(xt_val_prob.reshape(-1,1))
    val = np.concatenate([val, xt_val_prob], axis=1)
    valX, valY = create_dataset(val, look_back)
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "test_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/test_prob.csv")
    xt_test_prob = prob['hyper'].values
    # scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_test_prob = scaler_senti.transform(xt_test_prob.reshape(-1, 1))
    test = np.concatenate([test, xt_test_prob], axis=1)
    testX, testY = create_dataset(test, look_back)
    x_t = trim_dataset(trainX, BATCH_SIZE)
    y_t = trim_dataset(trainY, BATCH_SIZE)
    x_v = trim_dataset(valX, BATCH_SIZE)
    y_v = trim_dataset(valY, BATCH_SIZE)
    x_test = trim_dataset(testX, BATCH_SIZE)
    y_test = trim_dataset(testY, BATCH_SIZE)
    for i in range(runs):
        # model1 = create_model_senti(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, False)
        # history1 = model1.fit(x_t, y_t, epochs=100, verbose=0, batch_size=BATCH_SIZE,
        #                       shuffle=True, validation_data=(x_v, y_v))
        # model1.save_weights("weights.h5")
        model = create_model_tech(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, True)
        # model.load_weights("weights.h5", by_name=True, skip_mismatch=True)
        history = model.fit(x_t, y_t, epochs=100, verbose=0, batch_size=BATCH_SIZE,
                            shuffle=False, validation_data=(x_v, y_v))
        # model.save('lstm_model.h5')
        # Plot training
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['loss', 'validation'], loc='upper right')
        # plt.show()
        # # #
        # # make predictions
        # model = load_model('lstm_model.h5')
        y_pred = model.predict(trim_dataset(x_test, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred = y_pred.flatten()
        y_test_t = trim_dataset(y_test, BATCH_SIZE)
        # print(y_test_t)
        # print(y_test_t.shape)
        # print(y_test_t[0:15])

        y_pred = scaler_single.inverse_transform(y_pred.reshape(-1, 1))
        y_test_t = scaler_single.inverse_transform(y_test_t.reshape(-1, 1))
        error = mean_squared_error(y_test_t, y_pred)
        # print(y_pred)
        # # print(y_test_t)
        true_positive = 0
        true_negative = 0
        for i in range(len(y_pred)):
            if y_pred[i] > 0 and y_test_t[i] > 0:
                true_positive += 1
            elif y_pred[i] < 0 and y_test_t[i] < 0:
                true_negative += 1
        accuracy = (true_negative + true_positive) / (len(y_pred)) * 100
        print(accuracy)
        acc_.append(accuracy)
        print(trainY.shape)
        # calculate root mean squared error
        # trainScore = np.math.sqrt(y_pred)
        # print('Train Score: %.6f RMSE' % (trainScore))
        testScore = np.math.sqrt(error)
        print('Test Score: %.6f RMSE' % (testScore))
        rsme.append(testScore)
        plt.plot(y_test_t)
        plt.plot(y_pred)
        plt.title('model predictions, lr= %0.4f, units[%d,%d,%d]' % (lr, units[0], units[1], units[2]))
        plt.ylabel('close change')
        plt.xlabel('dates')
        plt.legend(['real', 'predict'], loc='upper right')
        plt.grid(True)
        plt.show()
    print(max(acc_))
    print(np.mean(acc_))
    print(min(rsme))
    print(np.mean(rsme))

    return max(acc_), np.mean(acc_), min(rsme), np.mean(rsme)

def run_viterbi_model(runs):
    BATCH_SIZE = 10
    TIME_STEPS = 5
    epochs = 100
    lr = 0.003
    units = [80,160, 30]
    acc_ = []
    rsme = []
    if colab:
        dataset = pd.read_csv(c.ROOT_DIR_Google + "Tdata_out.csv", index_col="formatted_date")
    else:
        dataset = pd.read_csv("Technical_data/Tdata_out.csv", index_col="formatted_date")

    # load the dataset
    data = dataset.values
    # split into train and test sets
    train_size = int(len(data) * 0.70)

    val_size = int((len(data) - train_size) / 2)
    train, val, test = data[0:train_size, :], data[train_size:(len(data) - val_size), :], data[
                                                                                          (len(
                                                                                              data) - val_size):len(
                                                                                              data), :]
    # print("train: {}, val: {}, test: {}".format(train.shape, val.shape, test.shape))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_single = MinMaxScaler(feature_range=(-1, 1))

    train = np.concatenate([scaler_single.fit_transform(train[:, 0].reshape(-1, 1)),
                            scaler.fit_transform(train[:, 1:])], axis=1)
    val = np.concatenate([scaler_single.fit_transform(val[:, 0].reshape(-1, 1)),
                          scaler.fit_transform(val[:, 1:])], axis=1)
    test = np.concatenate([scaler_single.fit_transform(test[:, 0].reshape(-1, 1)),
                           scaler.fit_transform(test[:, 1:])], axis=1)
    # # print(train[0:15])

    look_back = TIME_STEPS
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "train_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/train_prob.csv")
    xt_train_prob = prob['viterbi'].values
    scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_train_prob = scaler_senti.fit_transform(xt_train_prob.reshape(-1,1))
    train = np.concatenate([train, xt_train_prob], axis=1)
    trainX, trainY = create_dataset(train, look_back)
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "val_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/val_prob.csv")
    xt_val_prob = prob['viterbi'].values
    # scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_val_prob = scaler_senti.transform(xt_val_prob.reshape(-1,1))
    val = np.concatenate([val, xt_val_prob], axis=1)
    valX, valY = create_dataset(val, look_back)
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "test_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/test_prob.csv")
    xt_test_prob = prob['viterbi'].values
    # scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_test_prob = scaler_senti.transform(xt_test_prob.reshape(-1,1))
    test = np.concatenate([test, xt_test_prob], axis=1)
    testX, testY = create_dataset(test, look_back)
    x_t = trim_dataset(trainX, BATCH_SIZE)
    y_t = trim_dataset(trainY, BATCH_SIZE)
    x_v = trim_dataset(valX, BATCH_SIZE)
    y_v = trim_dataset(valY, BATCH_SIZE)
    x_test = trim_dataset(testX, BATCH_SIZE)
    y_test = trim_dataset(testY, BATCH_SIZE)
    for i in range(runs):
        # model1 = create_model_senti(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, False)
        # history1 = model1.fit(x_t, y_t, epochs=100, verbose=0, batch_size=BATCH_SIZE,
        #                       shuffle=True, validation_data=(x_v, y_v))
        # model1.save_weights("weights.h5")
        model = create_model_tech(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, True)
        # model.load_weights("weights.h5", by_name=True, skip_mismatch=True)
        history = model.fit(x_t, y_t, epochs=100, verbose=0, batch_size=BATCH_SIZE,
                            shuffle=False, validation_data=(x_v, y_v))
        # model.save('lstm_model.h5')
        # Plot training
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['loss', 'validation'], loc='upper right')
        # plt.show()
        # # #
        # # make predictions
        # model = load_model('lstm_model.h5')
        y_pred = model.predict(trim_dataset(x_test, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred = y_pred.flatten()
        y_test_t = trim_dataset(y_test, BATCH_SIZE)
        # print(y_test_t)
        # print(y_test_t.shape)
        # print(y_test_t[0:15])

        y_pred = scaler_single.inverse_transform(y_pred.reshape(-1, 1))
        y_test_t = scaler_single.inverse_transform(y_test_t.reshape(-1, 1))
        error = mean_squared_error(y_test_t, y_pred)
        # print(y_pred)
        # # print(y_test_t)
        true_positive = 0
        true_negative = 0
        for i in range(len(y_pred)):
            if y_pred[i] > 0 and y_test_t[i] > 0:
                true_positive += 1
            elif y_pred[i] < 0 and y_test_t[i] < 0:
                true_negative += 1
        accuracy = (true_negative + true_positive) / (len(y_pred)) * 100
        print(accuracy)
        acc_.append(accuracy)
        print(trainY.shape)
        # calculate root mean squared error
        # trainScore = np.math.sqrt(y_pred)
        # print('Train Score: %.6f RMSE' % (trainScore))
        testScore = np.math.sqrt(error)
        print('Test Score: %.6f RMSE' % (testScore))
        rsme.append(testScore)
        plt.plot(y_test_t)
        plt.plot(y_pred)
        plt.title('model predictions, lr= %0.4f, units[%d,%d,%d]' % (lr, units[0], units[1], units[2]))
        plt.ylabel('close change')
        plt.xlabel('dates')
        plt.legend(['real', 'predict'], loc='upper right')
        plt.grid(True)
        plt.show()
    print(max(acc_))
    print(np.mean(acc_))
    print(min(rsme))
    print(np.mean(rsme))

    return max(acc_), np.mean(acc_), min(rsme), np.mean(rsme)

def run_hyp_vit_model(runs):
    BATCH_SIZE = 10
    TIME_STEPS = 5
    epochs = 100
    lr = 0.003
    units = [80,160, 30]
    acc_ = []
    rsme = []
    if colab:
        dataset = pd.read_csv(c.ROOT_DIR_Google + "Tdata_out.csv", index_col="formatted_date")
    else:
        dataset = pd.read_csv("Technical_data/Tdata_out.csv", index_col="formatted_date")

    # load the dataset
    data = dataset.values
    # split into train and test sets
    train_size = int(len(data) * 0.70)

    val_size = int((len(data) - train_size) / 2)
    train, val, test = data[0:train_size, :], data[train_size:(len(data) - val_size), :], data[
                                                                                          (len(
                                                                                              data) - val_size):len(
                                                                                              data), :]
    # print("train: {}, val: {}, test: {}".format(train.shape, val.shape, test.shape))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_single = MinMaxScaler(feature_range=(-1, 1))

    train = np.concatenate([scaler_single.fit_transform(train[:, 0].reshape(-1, 1)),
                            scaler.fit_transform(train[:, 1:])], axis=1)
    val = np.concatenate([scaler_single.fit_transform(val[:, 0].reshape(-1, 1)),
                          scaler.fit_transform(val[:, 1:])], axis=1)
    test = np.concatenate([scaler_single.fit_transform(test[:, 0].reshape(-1, 1)),
                           scaler.fit_transform(test[:, 1:])], axis=1)
    # # print(train[0:15])

    look_back = TIME_STEPS
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "train_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/train_prob.csv")
    xt_train_prob = prob[['hyper','viterbi']].values
    scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_train_prob = scaler_senti.fit_transform(xt_train_prob)
    train = np.concatenate([train, xt_train_prob], axis=1)
    trainX, trainY = create_dataset(train, look_back)
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "val_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/val_prob.csv")
    xt_val_prob = prob[['hyper','viterbi']].values
    # scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_val_prob = scaler_senti.transform(xt_val_prob)
    val = np.concatenate([val, xt_val_prob], axis=1)
    valX, valY = create_dataset(val, look_back)
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "test_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/test_prob.csv")
    xt_test_prob = prob[['hyper','viterbi']].values
    # scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_test_prob = scaler_senti.transform(xt_test_prob)
    test = np.concatenate([test, xt_test_prob], axis=1)
    testX, testY = create_dataset(test, look_back)
    x_t = trim_dataset(trainX, BATCH_SIZE)
    y_t = trim_dataset(trainY, BATCH_SIZE)
    x_v = trim_dataset(valX, BATCH_SIZE)
    y_v = trim_dataset(valY, BATCH_SIZE)
    x_test = trim_dataset(testX, BATCH_SIZE)
    y_test = trim_dataset(testY, BATCH_SIZE)
    for i in range(runs):
        # model1 = create_model_senti(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, False)
        # history1 = model1.fit(x_t, y_t, epochs=100, verbose=0, batch_size=BATCH_SIZE,
        #                       shuffle=True, validation_data=(x_v, y_v))
        # model1.save_weights("weights.h5")
        model = create_model_tech(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, True)
        # model.load_weights("weights.h5", by_name=True, skip_mismatch=True)
        history = model.fit(x_t, y_t, epochs=epochs, verbose=0, batch_size=BATCH_SIZE,
                            shuffle=False, validation_data=(x_v, y_v))
        # model.save('lstm_model.h5')
        # Plot training
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['loss', 'validation'], loc='upper right')
        # plt.show()
        # # #
        # # make predictions
        # model = load_model('lstm_model.h5')
        y_pred = model.predict(trim_dataset(x_test, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred = y_pred.flatten()
        y_test_t = trim_dataset(y_test, BATCH_SIZE)
        # print(y_test_t)
        # print(y_test_t.shape)
        # print(y_test_t[0:15])

        y_pred = scaler_single.inverse_transform(y_pred.reshape(-1, 1))
        y_test_t = scaler_single.inverse_transform(y_test_t.reshape(-1, 1))
        error = mean_squared_error(y_test_t, y_pred)
        # print(y_pred)
        # # print(y_test_t)
        true_positive = 0
        true_negative = 0
        for i in range(len(y_pred)):
            if y_pred[i] > 0 and y_test_t[i] > 0:
                true_positive += 1
            elif y_pred[i] < 0 and y_test_t[i] < 0:
                true_negative += 1
        accuracy = (true_negative + true_positive) / (len(y_pred)) * 100
        print(accuracy)
        acc_.append(accuracy)
        print(trainY.shape)
        # calculate root mean squared error
        # trainScore = np.math.sqrt(y_pred)
        # print('Train Score: %.6f RMSE' % (trainScore))
        testScore = np.math.sqrt(error)
        print('Test Score: %.6f RMSE' % (testScore))
        rsme.append(testScore)
        plt.plot(y_test_t)
        plt.plot(y_pred)
        plt.title('model predictions, lr= %0.4f, units[%d,%d,%d]' % (lr, units[0], units[1], units[2]))
        plt.ylabel('close change')
        plt.xlabel('dates')
        plt.legend(['real', 'predict'], loc='upper right')
        plt.grid(True)
        plt.show()
    print(max(acc_))
    print(np.mean(acc_))
    print(min(rsme))
    print(np.mean(rsme))

    return max(acc_), np.mean(acc_), min(rsme), np.mean(rsme)

def run_hyp_vit_mean_model(runs):
    BATCH_SIZE = 10
    TIME_STEPS = 5
    epochs = 100
    lr = 0.003
    units = [80,160, 30]
    acc_ = []
    rsme = []
    if colab:
        dataset = pd.read_csv(c.ROOT_DIR_Google + "Tdata_out.csv", index_col="formatted_date")
    else:
        dataset = pd.read_csv("Technical_data/Tdata_out.csv", index_col="formatted_date")

    # load the dataset
    data = dataset.values
    # split into train and test sets
    train_size = int(len(data) * 0.70)

    val_size = int((len(data) - train_size) / 2)
    train, val, test = data[0:train_size, :], data[train_size:(len(data) - val_size), :], data[
                                                                                          (len(
                                                                                              data) - val_size):len(
                                                                                              data), :]
    # print("train: {}, val: {}, test: {}".format(train.shape, val.shape, test.shape))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_single = MinMaxScaler(feature_range=(-1, 1))

    train = np.concatenate([scaler_single.fit_transform(train[:, 0].reshape(-1, 1)),
                            scaler.fit_transform(train[:, 1:])], axis=1)
    val = np.concatenate([scaler_single.fit_transform(val[:, 0].reshape(-1, 1)),
                          scaler.fit_transform(val[:, 1:])], axis=1)
    test = np.concatenate([scaler_single.fit_transform(test[:, 0].reshape(-1, 1)),
                           scaler.fit_transform(test[:, 1:])], axis=1)
    # # print(train[0:15])

    look_back = TIME_STEPS
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "train_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/train_prob.csv")
    xt_train_prob = prob[['hyper','viterbi','pos_mean', 'neg_mean']].values
    scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_train_prob = scaler_senti.fit_transform(xt_train_prob)
    train = np.concatenate([train, xt_train_prob], axis=1)
    trainX, trainY = create_dataset(train, look_back)
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "val_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/val_prob.csv")
    xt_val_prob = prob[['hyper','viterbi','pos_mean', 'neg_mean']].values
    # scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_val_prob = scaler_senti.transform(xt_val_prob)
    val = np.concatenate([val, xt_val_prob], axis=1)
    valX, valY = create_dataset(val, look_back)
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "test_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/test_prob.csv")
    xt_test_prob = prob[['hyper','viterbi','pos_mean', 'neg_mean']].values
    xt_test_prob = scaler_senti.transform(xt_test_prob)
    test = np.concatenate([test, xt_test_prob], axis=1)
    testX, testY = create_dataset(test, look_back)
    x_t = trim_dataset(trainX, BATCH_SIZE)
    y_t = trim_dataset(trainY, BATCH_SIZE)
    x_v = trim_dataset(valX, BATCH_SIZE)
    y_v = trim_dataset(valY, BATCH_SIZE)
    # x_t1 = trim_dataset(trainX1, BATCH_SIZE)
    # y_t1 = trim_dataset(trainY1, BATCH_SIZE)
    # x_v1 = trim_dataset(valX1, BATCH_SIZE)
    # y_v1 = trim_dataset(valY1, BATCH_SIZE)
    x_test = trim_dataset(testX, BATCH_SIZE)
    y_test = trim_dataset(testY, BATCH_SIZE)
    for i in range(runs):
        # model1 = create_model_senti(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, False)
        # history1 = model1.fit(x_t, y_t, epochs=100, verbose=0, batch_size=BATCH_SIZE,
        #                     shuffle=True, validation_data=(x_v, y_v))
        # model1.save_weights("weights.h5")
        model = create_model_tech(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, True)
        # model.load_weights("weights.h5", by_name=True, skip_mismatch=True)
        history = model.fit(x_t, y_t, epochs=100, verbose=0, batch_size=BATCH_SIZE,
                            shuffle=False, validation_data=(x_v, y_v))
        # model.save('lstm_model.h5')
        # Plot training
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['loss', 'validation'], loc='upper right')
        # plt.show()
        # # #
        # # make predictions
        # model = load_model('lstm_model.h5')
        y_pred = model.predict(trim_dataset(x_test, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred = y_pred.flatten()
        y_test_t = trim_dataset(y_test, BATCH_SIZE)
        # print(y_test_t)
        # print(y_test_t.shape)
        # print(y_test_t[0:15])

        y_pred = scaler_single.inverse_transform(y_pred.reshape(-1, 1))
        y_test_t = scaler_single.inverse_transform(y_test_t.reshape(-1, 1))
        error = mean_squared_error(y_test_t, y_pred)
        # print(y_pred)
        # # print(y_test_t)
        true_positive = 0
        true_negative = 0
        for i in range(len(y_pred)):
            if y_pred[i] > 0 and y_test_t[i] > 0:
                true_positive += 1
            elif y_pred[i] < 0 and y_test_t[i] < 0:
                true_negative += 1
        accuracy = (true_negative + true_positive) / (len(y_pred)) * 100
        print(accuracy)
        acc_.append(accuracy)
        print(trainY.shape)
        # calculate root mean squared error
        # trainScore = np.math.sqrt(y_pred)
        # print('Train Score: %.6f RMSE' % (trainScore))
        testScore = np.math.sqrt(error)
        print('Test Score: %.6f RMSE' % (testScore))
        rsme.append(testScore)
        plt.plot(y_test_t)
        plt.plot(y_pred)
        plt.title('model predictions, lr= %0.4f, units[%d,%d,%d]' % (lr, units[0], units[1], units[2]))
        plt.ylabel('close change')
        plt.xlabel('dates')
        plt.legend(['real', 'predict'], loc='upper right')
        plt.grid(True)
        plt.show()
    print(max(acc_))
    print(np.mean(acc_))
    print(min(rsme))
    print(np.mean(rsme))

    return max(acc_), np.mean(acc_), min(rsme), np.mean(rsme)

def run_hyp_mean_model(runs):
    BATCH_SIZE = 10
    TIME_STEPS = 5
    epochs = 100
    lr = 0.003
    units = [80,160, 30]
    acc_ = []
    rsme = []
    if colab:
        dataset = pd.read_csv(c.ROOT_DIR_Google + "Tdata_out.csv", index_col="formatted_date")
    else:
        dataset = pd.read_csv("Technical_data/Tdata_out.csv", index_col="formatted_date")

    # load the dataset
    data = dataset.values
    # split into train and test sets
    train_size = int(len(data) * 0.70)

    val_size = int((len(data) - train_size) / 2)
    train, val, test = data[0:train_size, :], data[train_size:(len(data) - val_size), :], data[
                                                                                          (len(
                                                                                              data) - val_size):len(
                                                                                              data), :]
    # print("train: {}, val: {}, test: {}".format(train.shape, val.shape, test.shape))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_single = MinMaxScaler(feature_range=(-1, 1))

    train = np.concatenate([scaler_single.fit_transform(train[:, 0].reshape(-1, 1)),
                            scaler.fit_transform(train[:, 1:])], axis=1)
    val = np.concatenate([scaler_single.fit_transform(val[:, 0].reshape(-1, 1)),
                          scaler.fit_transform(val[:, 1:])], axis=1)
    test = np.concatenate([scaler_single.fit_transform(test[:, 0].reshape(-1, 1)),
                           scaler.fit_transform(test[:, 1:])], axis=1)
    # # print(train[0:15])

    look_back = TIME_STEPS
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "train_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/train_prob.csv")
    xt_train_prob = prob[['hyper','pos_mean', 'neg_mean']].values
    scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_train_prob = scaler_senti.fit_transform(xt_train_prob)
    train = np.concatenate([train, xt_train_prob], axis=1)
    trainX, trainY = create_dataset(train, look_back)
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "val_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/val_prob.csv")
    xt_val_prob = prob[['hyper','pos_mean', 'neg_mean']].values
    # scaler_senti = MinMaxScaler(feature_range=(-1, 1))
    xt_val_prob = scaler_senti.transform(xt_val_prob)
    val = np.concatenate([val, xt_val_prob], axis=1)
    valX, valY = create_dataset(val, look_back)
    if colab:
        prob = pd.read_csv(c.ROOT_DIR_Google + "test_prob.csv")
    else:
        prob = pd.read_csv("Sentiment_data/test_prob.csv")
    xt_test_prob = prob[['hyper','pos_mean', 'neg_mean']].values
    xt_test_prob = scaler_senti.transform(xt_test_prob)
    test = np.concatenate([test, xt_test_prob], axis=1)
    testX, testY = create_dataset(test, look_back)
    x_t = trim_dataset(trainX, BATCH_SIZE)
    y_t = trim_dataset(trainY, BATCH_SIZE)
    x_v = trim_dataset(valX, BATCH_SIZE)
    y_v = trim_dataset(valY, BATCH_SIZE)
    # x_t1 = trim_dataset(trainX1, BATCH_SIZE)
    # y_t1 = trim_dataset(trainY1, BATCH_SIZE)
    # x_v1 = trim_dataset(valX1, BATCH_SIZE)
    # y_v1 = trim_dataset(valY1, BATCH_SIZE)
    x_test = trim_dataset(testX, BATCH_SIZE)
    y_test = trim_dataset(testY, BATCH_SIZE)
    for i in range(runs):
        # model1 = create_model_senti(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, False)
        # history1 = model1.fit(x_t, y_t, epochs=100, verbose=0, batch_size=BATCH_SIZE,
        #                     shuffle=True, validation_data=(x_v, y_v))
        # model1.save_weights("weights.h5")
        model = create_model_tech(x_t.shape[2], BATCH_SIZE, TIME_STEPS, lr, units, True)
        # model.load_weights("weights.h5", by_name=True, skip_mismatch=True)
        history = model.fit(x_t, y_t, epochs=100, verbose=0, batch_size=BATCH_SIZE,
                            shuffle=False, validation_data=(x_v, y_v))
        # model.save('lstm_model.h5')
        # Plot training
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['loss', 'validation'], loc='upper right')
        # plt.show()
        # # #
        # # make predictions
        # model = load_model('lstm_model.h5')
        y_pred = model.predict(trim_dataset(x_test, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred = y_pred.flatten()
        y_test_t = trim_dataset(y_test, BATCH_SIZE)
        # print(y_test_t)
        # print(y_test_t.shape)
        # print(y_test_t[0:15])

        y_pred = scaler_single.inverse_transform(y_pred.reshape(-1, 1))
        y_test_t = scaler_single.inverse_transform(y_test_t.reshape(-1, 1))
        error = mean_squared_error(y_test_t, y_pred)
        # print(y_pred)
        # # print(y_test_t)
        true_positive = 0
        true_negative = 0
        for i in range(len(y_pred)):
            if y_pred[i] > 0 and y_test_t[i] > 0:
                true_positive += 1
            elif y_pred[i] < 0 and y_test_t[i] < 0:
                true_negative += 1
        accuracy = (true_negative + true_positive) / (len(y_pred)) * 100
        print(accuracy)
        acc_.append(accuracy)
        print(trainY.shape)
        # calculate root mean squared error
        # trainScore = np.math.sqrt(y_pred)
        # print('Train Score: %.6f RMSE' % (trainScore))
        testScore = np.math.sqrt(error)
        print('Test Score: %.6f RMSE' % (testScore))
        rsme.append(testScore)
        plt.plot(y_test_t)
        plt.plot(y_pred)
        plt.title('model predictions, lr= %0.4f, units[%d,%d,%d]' % (lr, units[0], units[1], units[2]))
        plt.ylabel('close change')
        plt.xlabel('dates')
        plt.legend(['real', 'predict'], loc='upper right')
        plt.grid(True)
        plt.show()
    print(max(acc_))
    print(np.mean(acc_))
    print(min(rsme))
    print(np.mean(rsme))

    return max(acc_), np.mean(acc_), min(rsme), np.mean(rsme)

# word_dict, prob_dict, xt_train, xt_val, xt_test, tweet_train, tweet_val, tweet_test = setup_sentiment_analyser(load=True)
# units=[140,80,30]
# get_probabilities(word_dict, prob_dict, xt_train, xt_val, xt_test, tweet_train, tweet_val, tweet_test)
# evaluate_bayes()
# evaluate_viterbi()
a,b,c,d = run_technical_model(10, BATCH_SIZE = 10, TIME_STEPS = 5, epochs = 100, lr = 0.003, units = [80,160,30])
e,f,g,h = run_hyp_model(10)
i,j,k,l = run_viterbi_model(10)
m,n,o,p =run_mean_model(10)
q,r,s,t = run_hyp_vit_model(10)
a1,a2,a3, a4 = run_hyp_mean_model(10)
u,v,w,x = run_hyp_vit_mean_model(10)
print("Tech: {}, {}, {}, {}".format(a,b,c,d))
print("Hype: {}, {}, {}, {}".format(e,f,g,h))
print("Vit: {}, {}, {}, {}".format(i,j,k,l))
print("Mean: {}, {}, {}, {}".format(m,n,o,p))
print("Hype+Vit: {}, {}, {}, {}".format(q,r,s,t))
print("Hype+Mean: {}, {}, {}, {}".format(a1,a2,a3,a4))
print("All: {}, {}, {}, {}".format(u,v,w,x))




# lr_search=[0.001,0.002,0.003,0.004,0.005]
# units_search=[[150,80,10],[160,80,10],[150,90,10],[160,90,10],[150,80,20],[160,80,20],[150,90,20],[160,90,20]]
# for i in range(len(lr_search)):
#     for j in range(len(units_search)):
#         run_technical_model(1,32,10,100,lr_search[i],units_search[j])
