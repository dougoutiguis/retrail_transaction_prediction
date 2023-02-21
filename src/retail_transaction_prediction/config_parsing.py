import configparser

CONF_FILE_PATH = "C:/Users/Guest1/PycharmProjects/retrail_transaction_prediction/src/retail_transaction_prediction/resources/config"
def get_file_path():
    config_file = configparser.ConfigParser()
    config_file.read(CONF_FILE_PATH)
    config_file.sections()
    return config_file["data"]["data_path"]

