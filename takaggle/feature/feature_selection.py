import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def stract_hists(feature, train, test, adjust=False, plot=False):
    n_bins = 10
    train_data = train[feature]
    test_data = test[feature]
    if adjust:
        test_data *= train_data.mean() / (test_data.mean() + 1e-7)
    perc_90 = np.percentile(train_data, 95)
    train_data = np.clip(train_data, 0, perc_90)
    test_data = np.clip(test_data, 0, perc_90)
    train_hist = np.histogram(train_data, bins=n_bins)[0] / len(train_data)
    test_hist = np.histogram(test_data, bins=n_bins)[0] / len(test_data)
    msre = mean_squared_error(train_hist, test_hist)
    if plot:
        print(msre)
        plt.bar(range(n_bins), train_hist, color='blue', alpha=0.5)
        plt.bar(range(n_bins), test_hist, color='red', alpha=0.5)
        plt.show()
    return msre


def select_ajusted(reduce_train, reduce_test):
    to_exclude = []
    ajusted_test = reduce_test.copy()
    for feature in ajusted_test.columns:
        if feature not in ['accuracy_group', 'installation_id', 'accuracy_group', 'session_title']:
            data = reduce_train[feature]
            train_mean = data.mean()
            data = ajusted_test[feature]
            test_mean = data.mean()
            try:
                error = stract_hists(feature, train=reduce_train, test=reduce_test, adjust=True)
                ajust_factor = train_mean / (test_mean + 1e-7)
                # if ajust_factor > 10 or ajust_factor < 0.1:
                if error > 0.01:
                    to_exclude.append(feature)
                    # print('除外リストに追加 特徴量:{}, train_mean:{}, test_mean:{}, error:{}'.format(feature, train_mean, test_mean, error))
                else:
                    ajusted_test[feature] *= ajust_factor
            except:
                to_exclude.append(feature)
                # print('except 除外リストに追加 特徴量:{}, train_mean:{}, test_mean:{}'.format(feature, train_mean, test_mean))

    return to_exclude, ajusted_test
