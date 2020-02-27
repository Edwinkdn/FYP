import datetime
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
from yahoofinancials import YahooFinancials
import csv
import pandas as pd
import date_handler as d_hand
import config as c
from config import Running_Colab as colab
if colab:
    import sys
    sys.path.append('/content/drive/My Drive/FYP')
import numpy as np

def build_tdata_history(ticker, min_date, max_date, colab):
    yahoo_financials = YahooFinancials(ticker)
    historical_stock_prices = yahoo_financials.get_historical_price_data(min_date, max_date, 'daily')
    nya_data = historical_stock_prices[ticker]
    price_data = nya_data['prices']
    if colab:
        csv_out = open(c.ROOT_DIR_Google+'Tdata_out.csv', mode='w')  # opens csv file
    else:
        csv_out = open('Technical_data/Tdata_out.csv', mode='w')  # opens csv file
    writer = csv.writer(csv_out)  # create the csv writer object
    fields = ['formatted_date','high', 'low', 'open', 'close', 'volume', 'adjclose']  # field names
    writer.writerow(fields)  # writes field
    for line in price_data:
        # writes a row and gets the fields from the json object
        writer.writerow([line.get('formatted_date')+ " "+ "16:00:00",
                         line.get('high'),
                         line.get('low'),
                         line.get('open'),
                         line.get('close'),
                         line.get('volume'),
                         line.get('adjclose')])
    csv_out.close()

def update_tdata_history(ticker, colab):
    for line in reversed(list(open(c.ROOT_DIR_Google+"Tdata_out.csv"))):
        last = line
        break
    token = last.split(",")
    min_date = token[0]
    # print(min_date)
    time = datetime.datetime.now()
    time = datetime.datetime.strftime(time, "%H:%M:%S")
    if time < "16:00:00":
        max_date = (datetime.datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        max_date = datetime.datetime.today().strftime('%Y-%m-%d')
    # print(max_date)
    if(max_date>min_date):
        yahoo_financials = YahooFinancials(ticker)
        historical_stock_prices = yahoo_financials.get_historical_price_data(min_date, max_date, 'daily')
        nya_data = historical_stock_prices[ticker]
        price_data = nya_data['prices']
        if colab:
            csv_out = open(c.ROOT_DIR_Google+'Tdata_out.csv', mode='a')  # opens csv file
        else:
            csv_out = open('Technical_data/Tdata_out.csv', mode='a')  # opens csv file
        writer = csv.writer(csv_out)  # create the csv writer object
        for line in price_data:
            # writes a row and gets the fields from the json object
            writer.writerow([line.get('formatted_date'),
                             line.get('high'),
                             line.get('low'),
                             line.get('open'),
                             line.get('close'),
                             line.get('volume'),
                             line.get('adjclose')])
        csv_out.close()
def find_market_closures(colab):
    if colab:
        price_data = pd.read_csv(c.ROOT_DIR_Google + 'Tdata_out.csv')
    else:
        price_data = pd.read_csv('Technical_data/Tdata_out.csv')
    dates = price_data['formatted_date'].tolist()
    print(dates)

    start = str(min(dates))
    end = str(max(dates))
    curr = start

    holidays = []
    while curr != end:
        n = d_hand.get_next_day(curr)
        if d_hand.is_weekday(n):
            if n not in dates:
                holidays.append(n)
        curr = n
    if colab:
        with open(c.ROOT_DIR_Google + 'NYSE_closure.txt', 'w', newline='') as file:
            for row in holidays:
                writer = csv.writer(file)
                writer.writerows([[row]])
        file.close()
    else:
        with open('Technical_data/NYSE_closure.txt', 'w', newline='') as file:
            for row in holidays:
                writer = csv.writer(file)
                writer.writerows([[row]])
        file.close()
def generate_price_change(day_t, day_t_1, scaled):
    # date_n = d_hand.get_calender_format(day_t[0])
    # date_n_before = d_hand.get_calender_format(day_t_1[0])
    # period = (date_n-date_n_before).days
    c_change = (day_t[5] -day_t_1[5])
    v_change = (day_t[4] -day_t_1[4])
    if scaled:
        if day_t_1[5] == 0.0:
            c_change = 0.0
        elif day_t_1[4] ==0.0:
            v_change = 0.0
        else:
            c_change = c_change/(day_t[5]*0.015)
            v_change = v_change/(day_t_1[4]*0.015)
    return c_change, v_change

def generate_MA(price_vector, intra_list, c_list, day):
    print('Getting {}day Moving Average...'.format(day))
    MA_list = [[],[],[]]
    for i in range(0,day-1):
        MA_list[0].append(float('NaN'))
        MA_list[1].append(float('NaN'))
        MA_list[2].append(float('NaN'))
    for i in range(day-1,len(price_vector)):
        close_, intra_, change_ = [], [], []
        for j  in range(0,day-1):
            close_.append(float(price_vector[i-j][5]))
            intra_.append(float(intra_list[i-j]))
            change_.append((float(c_list[i-j])))
        if intra_[0]  == float('NaN') or change_[0] == float('NaN'):
            MA_list[0].append(float('NaN'))
            MA_list[1].append(float('NaN'))
            MA_list[2].append(float('NaN'))
            continue
        else:
            MA_list[0].append((sum(close_)/day))
            MA_list[1].append((sum(intra_)/day))
            MA_list[2].append((sum(change_)/day))
    return MA_list

def scaling_close_vol(price_vector):
    close_, volume_=[],[]
    high_, low_, open_ = [], [], []
    for row in price_vector:
        close_.append(float(row[5]))
        volume_.append(float(row[4]))
        high_.append(float(row[1]))
        low_.append(float(row[2]))
        open_.append(float(row[3]))

    #sort and get 75th percentile as scaling factor
    close_sf = np.percentile(sorted(close_),75)
    volume_sf =np.percentile(sorted(volume_),75)
    high_sf = np.percentile(sorted(high_),75)
    low_sf = np.percentile(sorted(low_),75)
    open_sf = np.percentile(sorted(open_),75)

    #scaling
    for i in range(0,len(close_)):
        close_[i]=float(close_[i])/float(close_sf)
        volume_[i]=float(volume_[i])/float(volume_sf)
        high_[i]=float(high_[i])/float(high_sf)
        low_[i]=float(low_[i])/float(low_sf)
        open_[i]=float(open_[i])/float(open_sf)

    #put back to vector
    i=0
    for row in price_vector:
        row[4] = volume_[i]
        row[5] = close_[i]
        row[1] = high_[i]
        row[2] = low_[i]
        row[3] = open_[i]
        i+=1
    return price_vector

def generate_features_tdata_history(colab, scaled):
    if colab:
        pp_df = pd.read_csv(c.ROOT_DIR_Google + 'Tdata_out.csv')
    else:
        pp_df = pd.read_csv('Technical_data/Tdata_out.csv')
    price_vector = pp_df[['formatted_date', 'high', 'low', 'open', 'volume', 'adjclose']].values.tolist()
    if scaled:
        price_vector = scaling_close_vol(price_vector)
        pp_df[['formatted_date', 'high', 'low', 'open', 'volume', 'adjclose']] = price_vector
    amplitude_list, diff_list, intraday_list, c_change_list, v_change_list, MA5_list, MA10_list, MA20_list =[],[],[],[],[],[[],[],[]], [[],[],[]], [[],[],[]]
    amplitude_list.append(float(float('NaN')))
    diff_list.append(float('NaN'))
    intraday_list.append(float('NaN'))
    c_change_list.append(float('NaN'))
    v_change_list.append(float('NaN'))
    print("Generating technical features...")
    for i in range(1,len(price_vector)):
        amplitude_list.append((float(price_vector[i][1])-float(price_vector[i][2]))/float(price_vector[i-1][5]))
        diff_list.append((float(price_vector[i][5])-float(price_vector[i][3]))/float(price_vector[i-1][5]))
        intraday_list.append(float(price_vector[i][3])-float(price_vector[i-1][5]))
        c_change, v_change = generate_price_change(price_vector[i], price_vector[i-1], scaled)
        c_change_list.append(c_change)
        v_change_list.append(v_change)
    MA5_list = generate_MA(price_vector, intraday_list, c_change_list, 5)
    MA10_list = generate_MA(price_vector, intraday_list, c_change_list, 10)
    MA20_list = generate_MA(price_vector, intraday_list, c_change_list, 20)

    close_list, volume_list = pp_df['adjclose'], pp_df['volume']
    pp_df.drop(['high','low', 'open', 'close', 'volume','adjclose'], axis=1, inplace=True)

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # pp_df['adjclose'] = close_list
    pp_df['c_change'] = c_change_list
    pp_df['adjclose'] = close_list
    pp_df['volume'] = volume_list
    pp_df['amplitude'] = amplitude_list
    pp_df['difference'] = diff_list
    pp_df['intraday'] = intraday_list
    pp_df['v_change'] = v_change_list
    pp_df['MA5_Close'] = MA5_list[0]
    pp_df['MA10_Close'] = MA10_list[0]
    pp_df['MA20_Close'] = MA20_list[0]
    pp_df['MA5_Intra'] = MA5_list[1]
    pp_df['MA10_Intra'] = MA10_list[1]
    pp_df['MA20_Intra'] = MA20_list[1]
    pp_df['MA5_Change'] = MA5_list[2]
    pp_df['MA10_Change'] = MA10_list[2]
    pp_df['MA20_Change'] = MA20_list[2]
    pp_df.dropna(inplace=True)
    if colab:
        pp_df.to_csv(c.ROOT_DIR_Google+'Tdata_out.csv', encoding='utf-8', index=False)
    else:
        pp_df.to_csv('Technical_data/Tdata_out.csv', encoding = 'utf-8', index=False)


# def get_price_changes(scaled, colab):
#     if colab:
#         f = open(c.ROOT_DIR_Google + 'NYSE_closure.txt', 'r')
#     else:
#         f = open('NYSE_closure.txt', 'r')
#     holidays = f.read().splitlines()
#     f.close()
#     #evaulate the price changes of open from one day to next
#     if colab:
#         price_data = pd.read_csv(c.ROOT_DIR_Google + 'Tdata_out.csv')
#     else:
#         price_data = pd.read_csv('Tdata_out.csv')
#     dates = price_data['formatted_date'].tolist()
#     prices = price_data['adjclose'].tolist()
#     change_list = []
#     a_change_list =[]
#     change_list.append("")
#     a_change_list.append("")
#     print('Getting price changes...')
#     for i in range(1,len(dates)):
#         date_n = dates[i]
#         date_n_before = d_hand.get_previous_day(date_n)
#         period = 0
#         while not d_hand.is_weekday(date_n_before) or date_n_before in holidays:
#             period +=1
#             date_n_before = d_hand.get_previous_day(date_n_before)
#         if scaled:
#             price_change = (float(prices[i]) - float(prices[i-1]))/(period+1)
#         else:
#             price_change = (float(prices[i]) - float(prices[i - 1])) / (period + 1)
#         change_list.append(price_change)
#     price_data['price_change'] = change_list
#     price_data.dropna()
#     if colab:
#         price_data.to_csv(c.ROOT_DIR_Google + 'Tdata_out.csv', index=False)
#     else:
#         price_data.to_csv('Tdata_out.csv', index=False)



build_tdata_history('^NYA', '2013-11-30', '2020-02-14', colab)
generate_features_tdata_history(colab, scaled=True)
# find_market_closures(colab)
# get_price_changes()