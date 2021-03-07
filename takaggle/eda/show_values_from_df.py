import numpy as np
import pandas as pd
from IPython.display import display_html


# 欠損値の確認
def missing_values(data):
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


# これを呼ぶ
def show_all(df, n=5):
    print(f'Data shape :{df.shape}')
    df_describe = df.describe().style.set_table_attributes("style='display:inline'").set_caption('Describe Data Num')
    df_describe_o = df.describe(include=[object]).style.set_table_attributes("style='display:inline'").set_caption('Describe Data Object')
    df_head = df.head(n).style.set_table_attributes("style='display:inline'").set_caption('Head Data')
    df_tail = df.tail(n).style.set_table_attributes("style='display:inline'").set_caption('Tail Data')
    df_missing = missing_values(df).style.set_table_attributes("style='display:inline'").set_caption('Missing Value')
    df_frequent = most_frequent_values(df).style.set_table_attributes("style='display:inline'").set_caption('Frequent Value')
    df_unique = unique_values(df).style.set_table_attributes("style='display:inline'").set_caption('Unique Value')

    display_html(df_describe._repr_html_(), raw=True)
    display_html(df_describe_o._repr_html_(), raw=True)
    display_html(df_head._repr_html_(), raw=True)
    display_html(df_tail._repr_html_(), raw=True)
    display_html(df_missing._repr_html_(), raw=True)
    display_html(df_frequent._repr_html_(), raw=True)
    display_html(df_unique._repr_html_(), raw=True)
    return None

