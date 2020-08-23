import numpy as np
from sklearn.metrics import cohen_kappa_score


def qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
    :param max_rat:
    :return:
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


def eval_qwk_lgb(y_pred, data):
    """
    Fast cappa eval function for lgb.
    """
    y_true = data.get_label()
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return 'cappa', qwk(y_true, y_pred), True


def trans_target(y_pred):
    y_pred[y_pred <= 1.12232214] = 0
    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1
    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2
    y_pred[y_pred > 2.22506454] = 3
    return y_pred


def eval_qwk_lgb_regr(y_pred, data):
    """
    Fast cappa eval function for lgb.
    """
    y_true = data.get_label()
    y_pred = trans_target(y_pred)

    return 'cappa', qwk(y_true, y_pred), True


def metrics_qwk(y_true, y_pred):
    y_pred = trans_target(y_pred)
    return qwk(y_true, y_pred)


def metrics_cohen_kappa_score(y_true, y_pred):
    y_pred = trans_target(y_pred)
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
