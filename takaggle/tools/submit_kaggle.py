"""使い方
python submit_kaggle.py --sub_file=M5-Forecasting-Accuracy/models/lgb_0329_2011/lgb_0329_2011_submission.csv --compe_name='m5-forecasting-accuracy'
"""
import datetime
import fire
from kaggle.api.kaggle_api_extended import KaggleApi

now = datetime.datetime.now()
name_prefix = now.strftime("%Y%m%d-%H%M")


def main(sub_file, compe_name):
    # KaggleApi のインスタンスを用意
    api = KaggleApi()
    # 認証を通す
    api.authenticate()
    # submit時のメッセージ
    message = name_prefix
    # sub
    api.competition_submit(sub_file, message, compe_name)


if __name__ == "__main__":
    fire.Fire(main)
