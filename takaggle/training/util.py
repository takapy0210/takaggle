import datetime
import logging
import os
import numpy as np
import pandas as pd
import yaml
import joblib

CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']


class Util:

    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    @classmethod
    def dump_df_pickle(cls, df, path):
        df.to_pickle(path)

    @classmethod
    def load_df_pickle(cls, path):
        return pd.read_pickle(path)


class Logger:

    def __init__(self, path=''):
        self.general_logger = logging.getLogger(path + 'general')
        self.result_logger = logging.getLogger(path + 'result')
        self.info_logger = logging.getLogger('main')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(path + 'general.log')
        file_result_handler = logging.FileHandler(path + 'result.log')
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)
            self.info_logger.addHandler(stream_handler)
            self.info_logger.addHandler(file_result_handler)
            self.info_logger.setLevel(logging.DEBUG)

    def info(self, message):
        # 時刻をつけてコンソールとログに出力
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, scores, score_all_data=None):
        # 計算結果をコンソールと計算結果用ログに出力
        dic = dict()
        dic['name'] = run_name
        dic['score_all_data'] = score_all_data
        dic['score_fold_mean'] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f'score{i}'] = score
        self.result(self.to_ltsv(dic))

    def info_log(self, message) -> logging.Logger:
        # self.info_logger.info(message)
        self.info_logger.info('[{}] - {}'.format(self.now_string(), message))

    def now_string(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_ltsv(self, dic):
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])


class Submission:

    @classmethod
    def create_submission(cls, run_name, path, sub_y_column):
        logger = Logger(path)
        logger.info(f'{run_name} - start create submission')

        submission = pd.read_csv(RAW_DATA_DIR_NAME + 'sample_submission.csv')
        pred = Util.load_df_pickle(path + f'{run_name}_pred.pkl')
        submission[sub_y_column] = pred
        submission.to_csv(path + f'{run_name}_submission.csv', index=False)

        logger.info(f'{run_name} - end create submission')
