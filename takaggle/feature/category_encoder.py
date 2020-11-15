import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def sklearn_label_encoder(df, cols, del_col=False):
    """カテゴリ変換
    sklearnのLabelEncoderでEncodingを行う

    Args:
        df: カテゴリ変換する対象のデータフレーム
        cols (list of str): カテゴリ変換する対象のカラムリスト

    Returns:
        pd.Dataframe: dfにカテゴリ変換したカラムを追加したデータフレーム
    """
    if del_col:
        for col in cols:
            le = LabelEncoder()
            le_df = pd.DataFrame(le.fit_transform(df[[col]])).add_prefix(col + '_lbl_enc_')
            # 元のDFに結合
            df = pd.concat([df, le_df], axis=1)
            # leしたカラムを除外
            df = df.drop(col, axis=1)
    else:
        # 0~割り振られる
        for col in cols:
            le = LabelEncoder()
            not_null = df[col][df[col].notnull()]  # nullのデータは変換対象外
            df[col + '_sklearn_lbl_enc'] = pd.Series(le.fit_transform(not_null), index=not_null.index)
    return df


def sklearn_oh_encoder(df, cols, del_col=False):
    for col in cols:
        ohe = OneHotEncoder(sparse=False)
        ohe_df = pd.DataFrame(ohe.fit_transform(df[[col]])).add_prefix(col + '_oh_enc_')
        # 元のDFに結合
        df = pd.concat([df, ohe_df], axis=1)
        # oheしたカラムを除外
        df = df.drop(col, axis=1)
    return df


def ordinal_encoder(df, cols):
    # 1~割り振られる
    ce_oe = ce.OrdinalEncoder(cols=cols, handle_unknown='impute')
    temp_df = ce_oe.fit_transform(df[cols]).add_suffix('_ordinal_enc')
    df = pd.concat([df, temp_df], axis=1)
    return df


def one_hot_encoder(df, cols):
    ce_ohe = ce.OneHotEncoder(cols=cols, handle_unknown='impute')
    temp_df = ce_ohe.fit_transform(df[cols]).add_suffix('_onehot_enc')
    df = pd.concat([df, temp_df], axis=1)
    return df


def binary_encoder(df, cols):
    ce_binary = ce.BinaryEncoder(cols=cols, handle_unknown='impute')
    temp_df = ce_binary.fit_transform(df[cols]).add_suffix('_binary_enc')
    df = pd.concat([df, temp_df], axis=1)
    return df


# ****** ターゲットエンコーディング *******
def target_encoder_mean(df, train_df, cols, target):
    """目的変数の平均でエンコードする
    """
    target_col_name = ''
    for col in cols:
        target_col_name += str(col)
        target_col_name += '_'
    target_mean = train_df.groupby(cols)[target].mean().reset_index()\
        .rename({target: f'{target_col_name}targetenc_mean'}, axis=1)
    return pd.merge(df, target_mean, on=cols, how='left')


def target_encoder_std(df, train_df, cols, target):
    """目的変数の標準偏差でエンコードする
    """
    target_col_name = ''
    for col in cols:
        target_col_name += str(col)
        target_col_name += '_'
    target_mean = train_df.groupby(cols)[target].std().reset_index()\
        .rename({target: f'{target_col_name}targetenc_std'}, axis=1)
    return pd.merge(df, target_mean, on=cols, how='left')


def target_encoder(df, train_df, cols, target):
    """ceを用いたターゲットエンコーディング
    """
    ce_tgt = ce.TargetEncoder(cols=cols)
    ce_tgt.fit(X=train_df[cols], y=train_df[target])
    _df = ce_tgt.transform(df[cols])
    # カラム名の変更
    for col in cols:
        _df = _df.rename({col: f'{col}_targetenc_ce'}, axis=1)
    return pd.concat([df, _df], axis=1)


def target_encoder_loo(df, train_df, cols, target):
    # こちらは正真正銘looエンコードする
    ce_loo = ce.LeaveOneOutEncoder(cols=cols)
    ce_loo.fit(X=train_df[cols], y=train_df[target])
    _df = ce_loo.transform(df[cols])
    # カラム名の変更
    for col in cols:
        _df = _df.rename({col: f'{col}_targetenc_ce_loo'}, axis=1)
    return pd.concat([df, _df], axis=1)


def target_encoder_catboost(df, train_df, cols, target):
    ce_cbe = ce.CatBoostEncoder(cols=cols, random_state=42)
    ce_cbe.fit(X=train_df[cols], y=train_df[target])
    _df = ce_cbe.transform(df[cols])
    # カラム名の変更
    for col in cols:
        _df = _df.rename({col: f'{col}_targetenc_ce_cbe'}, axis=1)
    return pd.concat([df, _df], axis=1)


def target_encoder_jamesstein(df, train_df, cols, target):
    ce_jse = ce.JamesSteinEncoder(cols=cols, drop_invariant=True)
    ce_jse.fit(X=train_df[cols], y=train_df[target])
    _df = ce_jse.transform(df[cols])
    # カラム名の変更
    for col in cols:
        _df = _df.rename({col: f'{col}_targetenc_ce_jse'}, axis=1)
    return pd.concat([df, _df], axis=1)


def target_encoder_loo_include_self(train_df, test_df, cols, target):
    # こちらは自分も含めてtargetの平均でエンコードする
    ce_loo = ce.LeaveOneOutEncoder(cols=cols).fit(X=train_df[cols], y=train_df[target])
    tmp_train = ce_loo.transform(train_df[cols])
    if test_df is not None:
        tmp_test = ce_loo.transform(test_df[cols])
        return tmp_train, tmp_test
    return tmp_train, None
