import numpy as np
import pandas as pd
import mlflow
import sys
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split
from typing import Optional, Tuple, Union

from takaggle.training.model import Model
from takaggle.training.util import Logger, Util
from takaggle.training.TimeSeriesSplitter import CustomTimeSeriesSplitter

# 定数
shap_sampling = 10000
corr_sampling = 10000


def rmse(val_y, val_pred):
    return np.sqrt(mean_squared_error(val_y, val_pred))


def stratified_group_k_fold(X, y, groups, k, seed=None) -> (list, list):
    """StratifiedKFoldで分割する関数

    Args:
        X (pd.DataFrame): trainデータ
        y (pd.DataFrame): 目的変数のDF
        groups (pd.DataFrame): groupに指定するカラムのDF
        k (int): k数
        seed (int): seet. Defaults to None.

    Yields:
        list: trainデータのindexリスト
        list: validarionのindexリスト

    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)

    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]
        yield train_indices, test_indices


class Runner:

    def __init__(self, run_name, model_cls, features, setting, params, cv):
        """コンストラクタ
        :run_name: runの名前
        :model_cls: モデルのクラス
        :features: 特徴量のリスト
        :setting: 設定リスト
        :params: ハイパーパラメータ
        :cv: CVの設定
        """

        # setting情報
        self.target = setting.get('target')  # 目的変数
        self.calc_shap = setting.get('calc_shap')  # shapを計算するか否か
        self.save_train_pred = setting.get('save_train_pred')  # 学習データでの予測値を保存するか否か
        self.feature_dir_name = setting.get('feature_directory')  # 学習データの読み込み先ディレクトリ
        self.model_dir_name = setting.get('model_directory')  # 学習データの読み込み先ディレクトリ
        self.train_file_name = setting.get('train_file_name')
        self.test_file_name = setting.get('test_file_name')
        self.run_name = run_name  # run名
        self.model_cls = model_cls  # モデルクラス
        self.features = features  # 使用する特徴量のリスト
        self.params = params  # モデルのハイパーパラメータ
        self.out_dir_name = self.model_dir_name + run_name + '/'
        self.logger = Logger(self.out_dir_name)

        # 評価指標
        self.metrics_name = setting.get('metrics')
        self.logger.info(f'{self.run_name} - metrics is {self.metrics_name}')
        if self.metrics_name == 'MSE':
            self.metrics = mean_squared_error
        elif self.metrics_name == 'RMSE':
            self.metrics = rmse
        elif self.metrics_name == 'RMSLE':
            self.metrics = mean_squared_log_error
        elif self.metrics_name == 'MAE':
            self.metrics = mean_absolute_error
        elif self.metrics_name == 'ACC':
            self.metrics = accuracy_score
        elif self.metrics_name == 'CUSTOM':
            self.metrics = None
        else:
            self.metrics = None

        # データの読み込み
        self.train_x = self.load_x_train()  # 学習データの読み込み
        self.train_y = self.load_y_train()  # 学習データの読み込み

        # cv情報
        self.cv_method = cv.get('method')  # CVのメソッド名
        self.n_splits = cv.get('n_splits')  # k数
        self.random_state = cv.get('random_state')  # seed
        self.shuffle = cv.get('shuffle')  # shffleの有無
        self.cv_target_gr_column = cv.get('cv_target_gr')  # GroupKFold or StratifiedGroupKFoldを使用する時に指定する
        self.cv_target_sf_column = cv.get('cv_target_sf')  # StratifiedKFold or StratifiedGroupKFoldを使用する時に指定する
        self.test_size = cv.get('test_size')  # train_test_split用

        # ファイル出力用変数
        # 各fold,groupのスコアをファイルに出力するための2次元リスト
        self.score_list = []
        self.score_list.append(['run_name', self.run_name])
        self.fold_score_list = []

        # その他の情報
        self.remove_train_index = None  # trainデータからデータを絞り込む際に使用する。除外するindexを保持。
        if self.calc_shap:
            self.shap_values = np.zeros(self.train_x.shape)
        self.categoricals = []  # カテゴリ変数を指定する場合に使用する

        # ログにデータ件数を出力
        self.logger.info(f'{self.run_name} - train_x shape: {self.train_x.shape}')
        self.logger.info(f'{self.run_name} - train_y shape: {self.train_y.shape}')

        # TimeSeriecSplits用
        self.train_days = cv.get('train_days')
        self.test_days = cv.get('test_days')
        self.day_col = cv.get('day_col')
        self.pred_days = cv.get('pred_days')

    def visualize_corr(self):
        """相関係数を算出する
        """
        fig, ax = plt.subplots(figsize=(30, 20))
        plt.rcParams["font.size"] = 12  # 図のfontサイズ
        plt.tick_params(labelsize=14)  # 図のラベルのfontサイズ
        plt.tight_layout()

        # use a ranked correlation to catch nonlinearities
        df = self.train_x.copy()
        df[self.target] = self.train_y.copy()
        corr = df.sample(corr_sampling).corr(method='spearman')
        sns.heatmap(corr.round(3), annot=True,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values)

        # 保存
        plt.savefig(self.out_dir_name + self.run_name + '_corr.png', dpi=300, bbox_inches="tight")
        plt.close()

        del df, corr

    def shap_feature_importance(self) -> None:
        """計算したshap値を可視化して保存する
        """
        all_columns = self.train_x.columns.values.tolist() + [self.target]
        ma_shap = pd.DataFrame(sorted(zip(abs(self.shap_values).mean(axis=0), all_columns), reverse=True),
                               columns=['Mean Abs Shapley', 'Feature']).set_index('Feature')
        ma_shap = ma_shap.sort_values('Mean Abs Shapley', ascending=True)

        fig = plt.figure(figsize=(8, 25))
        plt.tick_params(labelsize=12)  # 図のラベルのfontサイズ
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('shap value')
        ax.barh(ma_shap.index, ma_shap['Mean Abs Shapley'], label='Mean Abs Shapley',  align="center", alpha=0.8)
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=0, fontsize=10)
        ax.legend(loc='upper left')
        plt.savefig(self.out_dir_name + self.run_name + '_shap.png', dpi=300, bbox_inches="tight")
        plt.close()

    def get_feature_name(self):
        """ 学習に使用する特徴量を返却
        """
        return self.train_x.columns.values.tolist()

    def train_fold(self, i_fold: Union[int, str]) -> Tuple[Model, Optional[np.array],
                                                           Optional[np.array], Optional[float]]:
        """クロスバリデーションでのfoldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        validation = i_fold != 'all'
        train_x = self.train_x.copy()
        train_y = self.train_y.copy()

        if validation:
            # 学習データ・バリデーションデータのindexを取得
            if self.cv_method == 'KFold':
                tr_idx, va_idx = self.load_index_k_fold(i_fold)
            elif self.cv_method == 'StratifiedKFold':
                tr_idx, va_idx = self.load_index_sk_fold(i_fold)
            elif self.cv_method == 'GroupKFold':
                tr_idx, va_idx = self.load_index_gk_fold_shuffle(i_fold)
            elif self.cv_method == 'StratifiedGroupKFold':
                tr_idx, va_idx = self.load_index_sgk_fold(i_fold)
            elif self.cv_method == 'TrainTestSplit':
                tr_idx, va_idx = self.load_index_train_test_split()
            elif self.cv_method == 'CustomTimeSeriesSplitter':
                tr_idx, va_idx = self.load_index_custom_ts_fold(i_fold)
            else:
                print('CVメソッドが正しくないため終了します')
                sys.exit(0)

            tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
            va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

            # TODO: ここもconfigで良い感じに管理できると良いね
            # pseudo labelingを行う場合
            """
            # pseudo labelingデータを追加する
            pseudo_df = pd.read_pickle(self.feature_dir_name + 'pseudo_labeling_lgb_hogehoge.pkl')
            pseudo_df_x = pseudo_df.drop('target', axis=1)[self.features]
            pseudo_df_y = pseudo_df['target']
            # 結合
            tr_x = pd.concat([tr_x, pseudo_df_x], axis=0)
            tr_y = pd.concat([tr_y, pseudo_df_y], axis=0)
            """

            # 学習を行う
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)

            # TODO: shap値もちゃんと計算できるようにしたい
            # バリデーションデータへの予測・評価を行う
            if self.calc_shap:
                va_pred, self.shap_values[va_idx[:shap_sampling]] = model.predict_and_shap(va_x, shap_sampling)
            else:
                va_pred = model.predict(va_x)

            # 評価指標がRMSLEの場合、log1pで学習させているため、元に戻して計算する
            if self.metrics_name == 'RMSLE':
                va_pred = np.expm1(va_pred)
                va_pred = np.where(va_pred < 0, 0, va_pred)
                score = np.sqrt(self.metrics(np.expm1(va_y), va_pred))
            elif self.metrics_name == 'ACC':
                va_pred = np.round(va_pred)
                score = self.metrics(va_y, va_pred)
            else:
                score = self.metrics(va_y, va_pred)

            # foldごとのスコアをリストに追加
            self.fold_score_list.append([f'fold{i_fold}', round(score, 4)])

            # TODO: ここもconfigで良い感じに管理できると良いね
            # 特定のカラム（グループ）ごとにスコアを算出したい場合
            # カテゴリ変数で、予測が容易なものもあれば難しいものもある場合に、ここを追加することで
            # そのカテゴリごとのスコアを確認できる
            # 事前にクラス変数にlistを宣言する必要がある
            """
            # 特別仕様: groupごとのスコアを算出
            _temp_df = pd.read_pickle(self.feature_dir_name + 'X_train.pkl')[['chip_id', 'chip_exc_wl']]
            _temp_df = _temp_df.iloc[va_idx].reset_index(drop=True)
            _temp_df = pd.concat([_temp_df, va_y.reset_index(drop=True), pd.Series(va_pred, name='pred')], axis=1)

            # chip_idの辞書
            with open(self.feature_dir_name + 'chip_dic.pkl', 'rb') as f:
                chip_dict = pickle.load(f)

            for i in sorted(_temp_df['chip_id'].unique().tolist()):
                chip_df = _temp_df.query('chip_id == @i')
                chip_y = chip_df['target']
                chip_pred = chip_df['pred']
                chip_score = self.metrics(chip_y, chip_pred)
                # chip_idごとのスコアをリストに追加
                self.chip_score_list.append([chip_dict[i], round(chip_score, 4)])

            for i in sorted(_temp_df['chip_exc_wl'].unique().tolist()):
                chip_exc_wl_df = _temp_df.query('chip_exc_wl == @i')
                chip_exc_wl_y = chip_exc_wl_df['target']
                chip_exc_wl_pred = chip_exc_wl_df['pred']
                chip_exc_wl_score = self.metrics(chip_exc_wl_y, chip_exc_wl_pred)
                # chip_exc_wlごとのスコアをリストに追加
                self.chip_exc_wl_score_list.append([i, round(chip_exc_wl_score, 4)])
            """

            # モデル、インデックス、予測値、評価を返す
            return model, va_idx, va_pred, score
        else:
            # 学習データ全てで学習を行う
            model = self.build_model(i_fold)
            model.train(train_x, train_y)

            # モデルを返す
            return model, None, None, None

    def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        self.logger.info(f'{self.run_name} - start training cv')
        if self.cv_method in ['KFold', 'TrainTestSplit', 'CustomTimeSeriesSplitter']:
            self.logger.info(f'{self.run_name} - cv method: {self.cv_method}')
        else:
            self.logger.info(f'{self.run_name} - cv method: {self.cv_method} - group: {self.cv_target_gr_column} - stratify: {self.cv_target_sf_column}')

        scores = []  # 各foldのscoreを保存
        va_idxes = []  # 各foldのvalidationデータのindexを保存
        preds = []  # 各foldの推論結果を保存

        # 各foldで学習を行う
        for i_fold in range(self.n_splits):
            # 学習を行う
            self.logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            self.logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            model.save_model(self.out_dir_name)

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        # 全体のスコアを算出
        if self.cv_method not in ['TrainTestSplit', 'CustomTimeSeriesSplitter']:
            if self.metrics_name == 'RMSLE':
                score_all_data = np.sqrt(self.metrics(np.expm1(self.train_y), preds))
            else:
                score_all_data = self.metrics(self.train_y, preds)
        else:
            score_all_data = None

        # oofデータに対するfoldごとのscoreをcsvに書き込む（foldごとに分析する用）
        self.score_list.append(['score_all_data', score_all_data])
        self.score_list.append(['score_fold_mean', np.mean(scores)])
        for i in self.fold_score_list:
            self.score_list.append(i)
        with open(self.out_dir_name + f'{self.run_name}_score.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(self.score_list)

        # foldごとのスコアもmlflowでトラッキングする
        def score_mean(df):
            df = df.groupby('run_name').mean().round(4).reset_index().sort_values('run_name')
            return df
        _score_df = pd.read_csv(self.out_dir_name + f'{self.run_name}_score.csv')
        _score_df = score_mean(_score_df)
        _score_df = _score_df.T
        _score_df.columns = _score_df.iloc[0]
        _score_df = _score_df.drop(_score_df.index[0])
        for col in _score_df.columns.tolist():
            mlflow.log_metric(col, _score_df[col].values[0])

        # 学習データでの予測結果の保存
        if self.save_train_pred:
            Util.dump_df_pickle(pd.DataFrame(preds), self.out_dir_name + f'.{self.run_name}_train.pkl')

        # 評価結果の保存
        self.logger.result_scores(self.run_name, scores, score_all_data)

        # shap feature importanceデータの保存
        if self.calc_shap:
            self.shap_feature_importance()

    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う
        あらかじめrun_train_cvを実行しておく必要がある
        """
        self.logger.info(f'{self.run_name} - start prediction cv')
        test_x = self.load_x_test()
        preds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_splits):
            self.logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model(self.out_dir_name)
            if self.metrics_name == 'RMSLE':
                pred = np.expm1(model.predict(test_x))
            else:
                pred = model.predict(test_x)
            preds.append(pred)
            self.logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 推論結果の保存（submit対象データ）
        Util.dump_df_pickle(pd.DataFrame(pred_avg), self.out_dir_name + f'{self.run_name}_pred.pkl')

        self.logger.info(f'{self.run_name} - end prediction cv')

    def run_train_all(self) -> None:
        """学習データすべてで学習し、そのモデルを保存する"""
        self.logger.info(f'{self.run_name} - start training all')

        # 学習データ全てで学習を行う
        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model(self.out_dir_name)

        self.logger.info(f'{self.run_name} - end training all')

    def run_predict_all(self) -> None:
        """学習データすべてで学習したモデルにより、テストデータの予測を行う
        あらかじめrun_train_allを実行しておく必要がある
        """
        self.logger.info(f'{self.run_name} - start prediction all')

        test_x = self.load_x_test()

        # 学習データ全てで学習したモデルで予測を行う
        i_fold = 'all'
        model = self.build_model(i_fold)
        model.load_model(self.out_dir_name)
        pred = model.predict(test_x)

        # 予測結果の保存
        Util.dump(pred, f'../model/pred/{self.run_name}-test.pkl')

        self.logger.info(f'{self.run_name} - end prediction all')

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う
        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f'{self.run_name}-fold{i_fold}'
        return self.model_cls(run_fold_name, self.params, self.categoricals)

    def load_x_train(self) -> pd.DataFrame:
        """学習データの特徴量を読み込む
        列名で抽出する以上のことを行う場合、このメソッドの修正が必要
        :return: 学習データの特徴量
        """
        # 複数のpklファイルにデータが散らばっている場合 -----------
        # dfs = [pd.read_pickle(self.feature_dir_name + f'{f}_train.pkl') for f in self.features]
        # df = pd.concat(dfs, axis=1)
        # -------------------------------------------------

        # csv or pklにまとまっている場合 -----------
        # df = pd.read_csv('../input/train.csv')[self.features]
        # df = pd.read_pickle(self.feature_dir_name + 'X_train.pkl')
        df = pd.read_pickle(self.feature_dir_name + f'{self.train_file_name}')
        df = df[self.features]
        # -------------------------------------------------

        # 特定のサンプルを除外して学習させる場合 -----------
        # self.remove_train_index = df[(df['age']==64) | (df['age']==66) | (df['age']==67)].index
        # df = df.drop(index = self.remove_train_index)
        # df = df[self.features]
        # -------------------------------------------------

        return df

    def load_y_train(self) -> pd.Series:
        """学習データの目的変数を読み込む
        対数変換や使用するデータを削除する場合には、このメソッドの修正が必要
        :return: 学習データの目的変数
        """
        # train_y = pd.read_pickle(self.feature_dir_name + self.target + '_train.pkl')
        df = pd.read_pickle(self.feature_dir_name + f'{self.train_file_name}')

        # 特定のサンプルを除外して学習させる場合 -------------
        # df = df.drop(index=self.remove_train_index)
        # -----------------------------------------

        if self.metrics_name == 'RMSLE':
            return pd.Series(np.log1p(df[self.target]))

        return pd.Series(df[self.target])

    def load_x_test(self) -> pd.DataFrame:
        """テストデータの特徴量を読み込む
        :return: テストデータの特徴量
        """
        # 複数のpklファイルにデータが散らばっている場合 -----------
        # dfs = [pd.read_pickle(self.feature_dir_name + f'{f}_test.pkl') for f in self.features]
        # df = pd.concat(dfs, axis=1)
        # -------------------------------------------------

        # csv or pklにまとまっている場合 -----------
        df = pd.read_pickle(self.feature_dir_name + f'{self.test_file_name}')
        df = df[self.features]
        # -------------------------------------------------

        return df

    def load_group_target(self) -> pd.Series:
        """StratifiedGroupKFoldで使用するgroup対象のデータを返却する
        """
        df = self.load_x_train() if self.train_x is None else self.train_x
        return df[self.cv_target_gr_column]

    def load_stratify_target(self) -> pd.Series:
        """StratifiedGroupKFoldで使用するstratify対象のデータを返却する

        基本的には目的変数yでOKなはず
        """
        if self.target == self.cv_target_sf_column:
            return self.train_y
        else:
            df = self.load_x_train() if self.train_x is None else self.train_x
            return df[self.cv_target_sf_column]

    def load_index_k_fold(self, i_fold: int) -> np.array:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        train_y = self.train_y
        dummy_x = np.zeros(len(train_y))
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        return list(kf.split(dummy_x))[i_fold]

    def load_index_sk_fold(self, i_fold: int) -> np.array:
        """StratifiedKFold クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        stratify_data = self.load_stratify_target()  # 分布の比率を維持したいデータの対象
        dummy_x = np.zeros(len(stratify_data))
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        return list(kf.split(dummy_x, stratify_data))[i_fold]

    def load_index_gk_fold(self, i_fold: int) -> np.array:
        """GroupKFold クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        group_data = self.load_group_target()
        train_y = self.train_y
        dummy_x = np.zeros(len(group_data))
        kf = GroupKFold(n_splits=self.n_splits)
        return list(kf.split(dummy_x, train_y, groups=group_data))[i_fold]

    def load_index_gk_fold_shuffle(self, i_fold: int) -> np.array:
        """GroupKFold（shuffleバージョン） クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        # scikit-learnのGroupKFoldはshuffleできないので、自作した
        group_data = self.load_group_target()
        unique_group_data = group_data.unique()
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        group_k_list = []

        for tr_group_idx, va_group_idx in kf.split(unique_group_data):
            tr_group = unique_group_data[tr_group_idx]
            va_group = unique_group_data[va_group_idx]

            is_tr = pd.DataFrame(group_data.isin(tr_group))
            is_va = pd.DataFrame(group_data.isin(va_group))

            tr_idx = list(is_tr[is_tr[self.cv_target_gr_column] == True].index)
            va_idx = list(is_va[is_va[self.cv_target_gr_column] == True].index)

            temp_list = (tr_idx, va_idx)
            group_k_list.append(temp_list)

        return group_k_list[i_fold]

    def load_index_sgk_fold(self, i_fold: int) -> np.array:
        """StratifiedGroupKFold クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        stratify_data = self.load_stratify_target()  # 分布の比率を維持したいデータ（基本的には正解ラベル）
        group_data = self.load_group_target()  # グループにしたいデータ

        stratified_group_k_list = []
        for fold, (trn_idx, val_idx) in enumerate(stratified_group_k_fold(self.train_x, stratify_data, group_data,
                                                                          k=self.n_splits, seed=self.random_state)):
            stratified_group_k_list.append((trn_idx, val_idx))

        return stratified_group_k_list[i_fold]

    def load_index_train_test_split(self) -> np.array:
        """fold-out train_testスプリットでインデックスを返す
        :return: レコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        train_y = self.train_y
        dummy_x = np.zeros(len(train_y))
        indices = np.arange(len(train_y))
        _, _, _, _, train_idx, test_idx = train_test_split(dummy_x, dummy_x, indices, test_size=self.test_size,
                                                           shuffle=self.shuffle, random_state=self.random_state)
        return train_idx, test_idx

    def load_index_custom_ts_fold(self, i_fold: int) -> np.array:
        """CustomTimeSeriesSplitter クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        cv_params = {
            "n_splits": self.n_splits,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "day_col": self.day_col,
            "pred_days": self.pred_days
        }

        tskf = CustomTimeSeriesSplitter(**cv_params)
        return list(tskf.split(self.train_x))[i_fold]
