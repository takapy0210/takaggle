import pandas as pd
import numpy as np


def aggregation(df, target_col, agg_target_col):
    """集計特徴量の生成処理

    Args:
        df (pd.DataFrame): 対象のDF
        target_col (list of str): 集計元カラム（多くの場合カテゴリ変数のカラム名リスト）
        agg_target_col (str): 集計対象のカラム（多くの場合連続変数）

    Returns:
        pd.DataFrame: データフレーム
    """

    # カラム名を定義
    target_col_name = ''
    for col in target_col:
        target_col_name += str(col)
        target_col_name += '_'

    gr = df.groupby(target_col)[agg_target_col]
    df[f'{target_col_name}{agg_target_col}_mean'] = gr.transform('mean').astype('float16')
    df[f'{target_col_name}{agg_target_col}_max'] = gr.transform('max').astype('float16')
    df[f'{target_col_name}{agg_target_col}_min'] = gr.transform('min').astype('float16')
    df[f'{target_col_name}{agg_target_col}_std'] = gr.transform('std').astype('float16')
    df[f'{target_col_name}{agg_target_col}_median'] = gr.transform('median').astype('float16')
    df[f'{target_col_name}{agg_target_col}_count'] = gr.transform('count').astype('float16')
    df[f'{target_col_name}{agg_target_col}_sum'] = gr.transform('sum').astype('float16')

    # quantile
    # 10%, 25%, 50%, 75%, 90%
    q10 = gr.quantile(0.1).reset_index().rename({agg_target_col: f'{target_col_name}{agg_target_col}_q10'}, axis=1)
    q25 = gr.quantile(0.25).reset_index().rename({agg_target_col: f'{target_col_name}{agg_target_col}_q25'}, axis=1)
    q50 = gr.quantile(0.5).reset_index().rename({agg_target_col: f'{target_col_name}{agg_target_col}_q50'}, axis=1)
    q75 = gr.quantile(0.75).reset_index().rename({agg_target_col: f'{target_col_name}{agg_target_col}_q75'}, axis=1)
    q90 = gr.quantile(0.9).reset_index().rename({agg_target_col: f'{target_col_name}{agg_target_col}_q90'}, axis=1)
    df = pd.merge(df, q10, how='left', on=target_col)
    df = pd.merge(df, q25, how='left', on=target_col)
    df = pd.merge(df, q50, how='left', on=target_col)
    df = pd.merge(df, q75, how='left', on=target_col)
    df = pd.merge(df, q90, how='left', on=target_col)

    # 差分
    df[f'{target_col_name}{agg_target_col}_max_minus_q90']\
        = df[f'{target_col_name}{agg_target_col}_max'] - df[f'{target_col_name}{agg_target_col}_q90']
    df[f'{target_col_name}{agg_target_col}_max_minus_q75']\
        = df[f'{target_col_name}{agg_target_col}_max'] - df[f'{target_col_name}{agg_target_col}_q75']
    df[f'{target_col_name}{agg_target_col}_max_minus_q50']\
        = df[f'{target_col_name}{agg_target_col}_max'] - df[f'{target_col_name}{agg_target_col}_q50']
    df[f'{target_col_name}{agg_target_col}_mean_minus_q90']\
        = df[f'{target_col_name}{agg_target_col}_mean'] - df[f'{target_col_name}{agg_target_col}_q90']
    df[f'{target_col_name}{agg_target_col}_mean_minus_q75']\
        = df[f'{target_col_name}{agg_target_col}_mean'] - df[f'{target_col_name}{agg_target_col}_q75']
    df[f'{target_col_name}{agg_target_col}_mean_minus_q50']\
        = df[f'{target_col_name}{agg_target_col}_mean'] - df[f'{target_col_name}{agg_target_col}_q50']

    # 自身の値との差分
    df[f'{target_col_name}{agg_target_col}_mean_diff'] = df[agg_target_col] - df[f'{target_col_name}{agg_target_col}_mean']
    df[f'{target_col_name}{agg_target_col}_max_diff'] = df[agg_target_col] - df[f'{target_col_name}{agg_target_col}_max']
    df[f'{target_col_name}{agg_target_col}_min_diff'] = df[agg_target_col] - df[f'{target_col_name}{agg_target_col}_min']

    return df


def division(df, target_list) -> pd.DataFrame:
    """リストの特徴量を除算する関数

    Args:
        df (pd.DataFrame): 対象のDF
        target_list (list of str): 除算対象の特徴量2次元リスト[[a, b], [b, c]]と指定した場合はa/bとb/cが計算される

    Returns:
        pd.DataFrame: データフレーム
    """
    df_division = pd.DataFrame()
    for i in range(len(target_list)):
        column_name = ''
        value = 0
        feature1 = target_list[i][0]
        feature2 = target_list[i][1]
        value = round(df[feature1] / df[feature2], 3)
        value = value.replace([np.inf, -np.inf], np.nan)
        value = value.fillna(0)

        column_name = target_list[i][0] + '_div_' + target_list[i][1]
        df_division[column_name] = value

    return df_division


def create_day_feature(df, col, prefix,
                       attrs=['year', 'quarter', 'month', 'week', 'day', 'dayofweek', 'hour', 'minute']):
    """日時特徴量の生成処理

    Args:
        df (pd.DataFrame): 日時特徴量を含むDF
        col (str)): 日時特徴量のカラム名
        prefix (str): 新しく生成するカラム名に付与するprefix
        attrs (list of str): 生成する日付特徴量. Defaults to ['year', 'quarter', 'month', 'week', 'day', 'dayofweek', 'hour', 'minute']
                             cf. https://qiita.com/Takemura-T/items/79b16313e45576bb6492

    Returns:
        pd.DataFrame: 日時特徴量を付与したDF

    """

    df.loc[:, col] = pd.to_datetime(df[col])

    for attr in attrs:
        dtype = np.int16 if attr == 'year' else np.int8
        df[prefix + '_' + attr] = getattr(df[col].dt, attr).astype(dtype)

    # 土日フラグ
    df[prefix + '_is_weekend'] = df[prefix + '_dayofweek'].isin([5, 6]).astype(np.int8)

    # 日付の周期性を算出
    def sin_cos_encode(df, col):
        df[col + '_cos'] = np.cos(2 * np.pi * df[col] / df[col].max())
        df[col + '_sin'] = np.sin(2 * np.pi * df[col] / df[col].max())
        return df

    for col in [prefix + '_' + 'quarter', prefix + '_' + 'month', prefix + '_' + 'day', prefix + '_' + 'dayofweek',
                prefix + '_' + 'hour', prefix + '_' + 'minute']:
        if col in df.columns.tolist():
            df = sin_cos_encode(df, col)

    return df
