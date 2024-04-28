import datetime
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


messages = []


def date_validation(date_text):
    valid = True
    try:
        datetime.date.fromisoformat(date_text)
        valid = True
    except ValueError:
        valid = False
    finally:
        return valid


def monthdelta(date, delta):
    m, y = (date.month+delta) % 12, date.year + ((date.month)+delta-1) // 12
    if not m: m = 12
    d = min(date.day, [31,
        29 if y%4==0 and (not y%100==0 or y%400 == 0) else 28,
        31,30,31,30,31,31,30,31,30,31][m-1])
    return date.replace(day=d,month=m, year=y)


config = {
	"ES": {
        "point_value": 50,
        "tick_size": 0.25
    },
    "NQ": {
        "point_value": 20,
        "tick_size": 0.25
    },
    "RT": {
        "point_value": 50,
        "tick_size": 0.1
    },
    "YM": {
        "point_value": 5,
        "tick_size": 1
    },
	"MES": {
        "point_value": 5,
        "tick_size": 0.25
    },
    "MNQ": {
        "point_value": 2,
        "tick_size": 0.25
    },
    "CL": {
        "point_value": 1000,
        "tick_size": 0.01
    },
    "MCL": {
        "point_value": 100,
        "tick_size": 0.01
    },
    "GC": {
        "point_value": 1000,
        "tick_size": 0.01
    },
    "MGC": {
        "point_value": 10,
        "tick_size": 0.1
    },
    "NG": {
        "point_value": 10000,
        "tick_size": 0.001
    },
    "M2K": {
        "point_value": 5,
        "tick_size": 0.1
    },
    "ZB": {
        "point_value": 31.25,
        "tick_size": 0.01
    },
    "ZC": {
        "point_value": 25,
        "tick_size": 0.25
    },
    "ZM": {
        "point_value": 100,
        "tick_size": 0.1
    },
    "ZL": {
        "point_value": 600,
        "tick_size": 0.01
    },
    "ZS": {
        "point_value": 50,
        "tick_size": 0.25
    },
    "ZW": {
        "point_value": 50,
        "tick_size": 0.25
    },
    "HG": {
        "point_value": 25000,
        "tick_size": 0.005
    },
    "MHG": {
        "point_value": 2500,
        "tick_size": 0.005
    },
    "HE": {
        "point_value": 400,
        "tick_size": 0.025
    },
    "LE": {
        "point_value": 400,
        "tick_size": 0.025
    },
    "SI": {
        "point_value": 5000,
        "tick_size": 0.005
    },
    "SIL": {
        "point_value": 1000,
        "tick_size": 0.005
    },
    "MYM": {
        "point_value": 0.5,
        "tick_size": 1
    },
    "6A": {
        "point_value": 125000,
        "tick_size": 5e-05
    },
    "6B": {
        "point_value": 62500,
        "tick_size": 0.0001
    },
    "6C": {
        "point_value": 100000,
        "tick_size": 5e-05
    },
    "6E": {
        "point_value": 125000,
        "tick_size": 5e-05
    },
    "6J": {
        "point_value": 125000,
        "tick_size": 5e-05
    }
}
