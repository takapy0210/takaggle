class CustomTimeSeriesSplitter:
    def __init__(self, n_splits=5, train_days=80, test_days=20, day_col="d", pred_days=28):
        self.n_splits = n_splits
        self.train_days = train_days
        self.test_days = test_days
        self.day_col = day_col
        self.pred_days = pred_days

    def split(self, X, y=None, groups=None):
        SEC_IN_DAY = 3600 * 24  # 1日の総時間（秒）
        # 全サンプルの日付から、先頭日付(一番過去の日付)との差分を計算（秒）
        # 全サンプル数の長さseriesがsetになる
        sec = (X[self.day_col] - X[self.day_col].iloc[0]) * SEC_IN_DAY
        duration = sec.max()  # 一番直近のデータ秒を取得

        # 学習データとテストデータの秒数を計算
        train_sec = self.train_days * SEC_IN_DAY
        test_sec = self.test_days * SEC_IN_DAY
        total_sec = test_sec + train_sec

        if self.n_splits == 1:
            train_start = duration - total_sec
            train_end = train_start + train_sec

            train_mask = (sec >= train_start) & (sec < train_end)
            test_mask = sec >= train_end

            yield sec[train_mask].index.values, sec[test_mask].index.values

        else:
            # step = (duration - total_sec) / (self.n_splits - 1)
            # testデータに含める秒数（ここでは28日分の秒数）
            step = self.pred_days * SEC_IN_DAY  # 予測したい日数（秒）

            for idx in range(self.n_splits):
                # train_start = idx * step
                shift = (self.n_splits - (idx + 1)) * step  # 当splitでの予測データ範囲
                train_start = duration - total_sec - shift
                train_end = train_start + train_sec
                test_end = train_end + test_sec

                # trainデータとするindex
                train_mask = (sec > train_start) & (sec <= train_end)

                # testデータとするindex
                if idx == self.n_splits - 1:
                    test_mask = sec > train_end
                else:
                    test_mask = (sec > train_end) & (sec <= test_end)

                yield sec[train_mask].index.values, sec[test_mask].index.values

    def get_n_splits(self):
        return self.n_splits
