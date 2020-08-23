import pandas as pd
import numpy as np


def aggregation(train_df, test_df, target, features_list, agg_list=['mean', 'std', 'median']) -> pd.DataFrame:
    """集計特徴量を生成する関数
    :train_df:tranデータ
    :test_df:testデータ
    :target:集計元の特徴量
    :features_list:集計対象の特徴量のリスト
    :agg_list:計算する統計量のリスト
    """
    tmp_df = pd.concat([train_df, test_df], axis=0, sort=False).reset_index(drop=True)
    tmp_df = tmp_df.groupby(target)[features_list].agg(agg_list).reset_index()
    return tmp_df


def binning(train_df, test_df, target, bin_edges) -> pd.Series:
    """連続変数などをbin分割する関数
    デフォルトでは左側（小さい方）のエッジの値は含まれない。
    :train_df:tranデータ
    :test_df:testデータ
    :target:bin分割対象の特徴量
    :bin_edges:ビン分割の範囲のリスト
    """
    train_binned = pd.cut(train_df[target], bin_edges, labels=False)
    test_binned = pd.cut(test_df[target], bin_edges, labels=False)
    return train_binned, test_binned


def division(df, targets_list) -> pd.DataFrame:
    """リストの特徴量を除算する関数
    :df:dfデータ
    :targets_list:除算対象の特徴量2次元リスト[[a, b], [b, c]]と指定した場合はa/bとb/cが計算される
    """
    df_division = pd.DataFrame()
    for i in range(len(targets_list)):
        column_name = ''
        value = 0
        feature1 = targets_list[i][0]
        feature2 = targets_list[i][1]
        value = round(df[feature1] / df[feature2], 3)
        value = value.replace([np.inf, -np.inf], np.nan)
        value = value.fillna(0)

        column_name = targets_list[i][0] + '_div_' + targets_list[i][1]
        df_division[column_name] = value

    return df_division