import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder


def sklearn_label_encoder(df, cols):
    """カテゴリ変換
    sklearnのLabelEncoderでEncodingを行う

    Args:
        df: カテゴリ変換する対象のデータフレーム
        cols (list of str): カテゴリ変換する対象のカラムリスト

    Returns:
        pd.Dataframe: dfにカテゴリ変換したカラムを追加したデータフレーム
    """
    # 0~割り振られる
    for col in cols:
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]  # nullのデータは変換対象外
        df[col + '_sklearn_lbl_enc'] = pd.Series(le.fit_transform(not_null), index=not_null.index)
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
def target_encoder_mean(train_df, test_df, cols, target):
    target_mean = train_df.groupby(cols)[target].mean()
    return pd.DataFrame(train_df[cols].map(target_mean)), pd.DataFrame(test_df[cols].map(target_mean))


def target_encoder_std(train_df, test_df, cols, target):
    target_mean = train_df.groupby(cols)[target].std()
    return pd.DataFrame(train_df[cols].map(target_mean)), pd.DataFrame(test_df[cols].map(target_mean))


def target_encoder(train_df, test_df, cols, target):
    ce_tgt = ce.TargetEncoder(cols=cols)
    tmp_train = ce_tgt.fit_transform(X=train_df[cols], y=train_df[target])
    if test_df is not None:
        tmp_test = ce_tgt.transform(test_df[cols])
        return tmp_train, tmp_test
    return tmp_train, None


def target_encoder_loo(train_df, test_df, cols, target):
    # こちらは正真正銘looエンコードする
    ce_loo = ce.LeaveOneOutEncoder(cols=cols)
    tmp_train = ce_loo.fit_transform(X=train_df[cols], y=train_df[target])
    if test_df is not None:
        tmp_test = ce_loo.transform(test_df[cols])
        return tmp_train, tmp_test
    return tmp_train, None


def target_encoder_loo_include_self(train_df, test_df, cols, target):
    # こちらは自分も含めてtargetの平均でエンコードする
    ce_loo = ce.LeaveOneOutEncoder(cols=cols).fit(X=train_df[cols], y=train_df[target])
    tmp_train = ce_loo.transform(train_df[cols])
    if test_df is not None:
        tmp_test = ce_loo.transform(test_df[cols])
        return tmp_train, tmp_test
    return tmp_train, None


def target_encoder_catboost(train_df, test_df, cols, target):
    ce_cbe = ce.CatBoostEncoder(cols=cols, random_state=42)
    tmp_train = ce_cbe.fit_transform(X=train_df[cols], y=train_df[target])
    if test_df is not None:
        tmp_test = ce_cbe.transform(test_df[cols])
        return tmp_train, tmp_test
    return tmp_train, None


def target_encoder_jamesstein(train_df, test_df, cols, target):
    ce_jse = ce.JamesSteinEncoder(cols=cols, drop_invariant=True)
    tmp_train = ce_jse.fit_transform(X=train_df[cols], y=train_df[target])
    if test_df is not None:
        tmp_test = ce_jse.transform(test_df[cols])
        return tmp_train, tmp_test
    return tmp_train, None
