import logging

# This is used instead of prints. By setting handler level (between 1 and 50) you can adjust number of messages
# depending on the importance
formatter = logging.Formatter('%(levelname)8s %(lineno)4s %(asctime)s: %(message)s', "%Y-%m-%d %H:%M:%S")
# To include file name: %(filename)s || %(module)s  || %(funcName)s
general_handler = logging.FileHandler('../logger.log', mode='w')
general_handler.setFormatter(formatter)
general_handler.setLevel(20)  # <- sets amount that is printed in the log file

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(25)  # <- sets amount that is printed on the console

general_logger = logging.getLogger(__name__)
general_logger.setLevel(-1)
general_logger.addHandler(general_handler)
general_logger.addHandler(stream_handler)