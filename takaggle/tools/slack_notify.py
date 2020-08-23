import requests
import json
import os

TOKEN = os.environ['SLACK_TOKEN']
CHANNEL = os.environ['SLACK_CHANNEL']


def slack_notify(notify_type='1', title=None, value=None, run_name=None, ts=None):

    url = 'https://slack.com/api/chat.postMessage'

    if notify_type == '1':
        color = None
        text = ':kerneler: < Notice `{}`'.format(run_name)
    elif notify_type == '2':
        message = '<@U7KG6AE8Y>'
        slack_title = ':kerneler: < `{}` learning is done.'.format(run_name)
        color = '#00FF00'
        text = message + '\n' +  slack_title,
        title = 'submitは下記コマンドで！'
        value = '{}python submit_kaggle.py --sub_file={}/{}_submission.csv{}'.format('`', run_name, run_name, '`')
    elif notify_type == '3':
        message = '<@U7KG6AE8Y>'
        slack_title = ':kerneler: < Notice `{}`'.format(run_name)
        color = '#00FF00'
        text = message + '\n' +  slack_title,
        title = title
        value = value
    else:
        slack_title = 'NaN'
        color = '#DF0101'
        text = 'error'
        title = 'miss_title'
        value = 'miss_value'

    data = {
        'token': TOKEN,
        'channel': CHANNEL,
        'text': text,
        'attachments': json.dumps([
            {
                'color': color,
                'fields': [
                    {
                        'title': title,
                        'value': value
                    }
                ]
            }
        ]),
        'thread_ts': ts
    }

    res = requests.post(url, data=data)
    json_data = res.json()
    return json_data


def slack_file_upload(file, ts=None):

    url = 'https://slack.com/api/files.upload'
    filename = file.split('.')[0]
    files = {'file': open(file, 'rb')}
    data = {
        'token': TOKEN,
        'channels': CHANNEL,
        'title': filename,
        'thread_ts': ts
    }
    requests.post(url, data=data, files=files)

    return 'slack notify Done'
