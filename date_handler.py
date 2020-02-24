import datetime
import config as c
from config import Running_Colab as colab
if colab:
    import sys
    sys.path.append('/content/drive/My Drive/FYP')

def get_calender_format(date_time):
    format = date_time.split("-")
    year = int(format[0])
    month = int(format[1])
    day = int(format[2].split(" ")[0])
    hour = int(format[2].split(" ")[1].split(':')[0])
    minute = int(format[2].split(" ")[1].split(':')[1])
    second = int(format[2].split(" ")[1].split(':')[2])
    f_date = datetime.datetime(year,month,day,hour,minute,second)
    return f_date

def get_next_day(date_time):
    n_day = get_calender_format(date_time)
    n_day += datetime.timedelta(days=1)
    n_day = datetime.datetime.strftime(n_day,"%Y-%m-%d %H:%M:%S")
    return n_day

def get_previous_day(date_time):
    n_day = get_calender_format(date_time)
    n_day += datetime.timedelta(days=-1)
    n_day = datetime.datetime.strftime(n_day,"%Y-%m-%d %H:%M:%S")
    return n_day

def is_weekday(date_time):
    weekends = [5,6]
    day = get_calender_format(date_time)
    if day.weekday() in weekends:
        return False
    return True

def change_to_sgt(date_time):
    date_time = get_calender_format(date_time)
    date_time += datetime.timedelta(hours=13)
    date_time = datetime.datetime.strftime(date_time, "%Y-%m-%d %H:%M:%S")
    return date_time

def previous_trading_date(date_time):
    if colab:
        f = open((c.ROOT_DIR_Google + 'NYSE_closure.txt'), 'r')
    else:
        f = open('Technical_data/NYSE_closure.txt', 'r')
    holidays = f.read().splitlines()
    f.close()
    befr = get_previous_day(date_time)
    while not is_weekday(befr) or befr in holidays:
        befr = get_previous_day(befr)
    return befr

def next_trading_date(date_time):
    if colab:
        f = open((c.ROOT_DIR_Google + 'NYSE_closure.txt'), 'r')
    else:
        f = open('Technical_data/NYSE_closure.txt', 'r')
    holidays = f.read().splitlines()
    f.close()
    aft = get_next_day(date_time)
    while not is_weekday(aft) or aft in holidays:
        aft = get_next_day(aft)
    return aft