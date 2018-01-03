#Do some logging
import logging, datetime
logger = logging.getLogger()

def setup_file_logger(log_file):
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

def log(message, print_time):
    #outputs to Jupyter console
    if print_time:
        print('{} {}'.format(datetime.datetime.now(), message))
    else:
        print('{}'.format(message))

    #outputs to file
    logger.info(message)
