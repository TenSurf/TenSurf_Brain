import logging
import requests
import json
from openai import OpenAI
from gpt.functions_python import FunctionCalls
from gpt.functions_json import functions
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


messages = []


def date_validation(date_text):
    valid = True
    try:
        datetime.strptime(date_text, "%d/%m/%Y %H:%M:%S")
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



# Hard Code Introduction
introduction = '\
This is an AI  \
'