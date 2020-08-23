import numpy as np
import pandas as pd


# 欠損値の確認
def missing_values(data):
    print('■□■□ missing_values □■□■')
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# 頻出値の確認
def most_frequent_values(data):
    print('■□■□ most_frequent_values □■□■')
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        itm = data[col].value_counts().index[0]
        val = data[col].value_counts().values[0]
        items.append(itm)
        vals.append(val)
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return(np.transpose(tt))


# ユニーク値の確認
def unique_values(data):
    print('■□■□ unique_values □■□■')
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    tt['Percent'] = np.round(uniques / total * 100, 3)
    return(np.transpose(tt))


def show_all(df):
    display(df.shape)
    display(df.head())
    display(missing_values(df))
    display(most_frequent_values(df))
    display(unique_values(df))
    return None
