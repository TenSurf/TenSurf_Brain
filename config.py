import logging


openai_api_key = "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"
# azure_api_endpoint = 'https://tensurf.openai.azure.com/'
azure_api_endpoint = "https://tensurfbrain1.openai.azure.com/"
azure_api_version = '2023-10-01-preview'
# azure_api_key = '74b3de375b964f73a6b7668fe459e26f'
azure_api_key = "80ddd1ad72504f2fa226755d49491a61"
GPT_MODEL_3 = "gpt-3.5-turbo-1106"
azure_GPT_MODEL_3 = "gpt_35"


logging.basicConfig(filename="logs/NOTSET.log",
                    filemode='a',
                    encoding='utf-8',
                    format='%(asctime)s %(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.NOTSET)

logging.basicConfig(filename="logs/DEBUG.log",
                    filemode='a',
                    encoding='utf-8',
                    format='%(asctime)s %(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)

logging.basicConfig(filename="logs/INFO.log",
                    filemode='a',
                    encoding='utf-8',
                    format='%(asctime)s %(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

logging.basicConfig(filename="logs/WARNING.log",
                    filemode='a',
                    encoding='utf-8',
                    format='%(asctime)s %(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.WARNING)

logging.basicConfig(filename="logs/ERROR.log",
                    filemode='a',
                    encoding='utf-8',
                    format='%(asctime)s %(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.ERROR)

logging.basicConfig(filename="logs/CRITICAL.log",
                    filemode='a',
                    encoding='utf-8',
                    format='%(asctime)s %(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.CRITICAL)
