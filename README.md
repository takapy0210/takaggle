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
from takaggle.utils import get_logger

LOGGER = get_logger()
LOGGER.info('hogehoge.')

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
