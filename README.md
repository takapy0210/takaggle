# takaggle
A set of scripts used in the data analysis competition

## Description
TBD

## Requirement
- [python package](https://github.com/takapy0210/takaggle/blob/master/requirements.txt)

## Install
```sh
pip install git+https://github.com/takapy0210/takaggle
```

# Usage

### import

```python
from takaggle.eda import show_all
from takaggle.feature import category_encoder
from takaggle.utils import get_logger
```

## eda

```python
from takaggle.eda import missing_values, most_frequent_values, unique_values, show_all

df = pd.read_csv('hoge.csv')

# show missing value
missing_values(df)

# show most frequent value
most_frequent_values(df)

# show unique value
unique_values(df)

# View all of the above at once
show_all(df)
```

## utils


```python
from takaggle.utils import get_logger, elapsed_time

# Log output
LOGGER = get_logger()
LOGGER.info('hogehoge.')

# Measure the processing time of a function
@elapsed_time(LOGGER)
def load_data() -> dict:
    """データの読み込み"""

    # 読み込むファイルを定義
    inputs = {
        'train': '../data/train.csv',
        'test': '../data/test.csv',
    }
    dfs = {}
    for k, v in inputs.items():
        dfs[k] = pd.read_csv(v)
        LOGGER.info(f'"{k}" dataframe shape: {dfs[k].shape}')

    return dfs

dfs = load_data()
# [2021-11-23 23:07:26] [ INFO] ***** Beg: load_data *****
# [2021-11-23 23:07:27] [ INFO] "train" dataframe shape: (91333, 18)
# [2021-11-23 23:07:28] [ INFO] "test" dataframe shape: (91822, 17)
# [2021-11-23 23:07:28] [ INFO] ***** End: load_data 2.02sec [1.4GB(+0.1GB)] *****

```

Measure the processing time of a function

```python
from takaggle.utils import get_logger, elapsed_time



```


## Document
TBD

## Test
TBD

## Other
TBD

## Deploy
increment `deploy.sh` tag_name

```sh
tag_name="v1.0.7"
```

```sh
sh deploy.sh
```
