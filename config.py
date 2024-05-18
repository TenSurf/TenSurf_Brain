import logging


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
