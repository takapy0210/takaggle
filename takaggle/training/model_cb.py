import os
import sys
from catboost import CatBoostRegressor, CatBoostClassifier
from model import Model
from util import Util


class ModelCB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット
        validation = va_x is not None

        # ハイパーパラメータの設定
        params = dict(self.params)
        verbose_eval = params.pop('verbose_eval')
        self.pred_type = params.pop('pred_type')

        if self.pred_type == 'Regressor':
            self.model = CatBoostRegressor(**params)
        elif self.pred_type == 'Classifier':
            self.model = CatBoostClassifier(**params)
        else:
            print('pred_typeが正しくないため終了します')
            sys.exit(0)

        # 学習
        if validation:
            self.model.fit(
                tr_x,
                tr_y,
                eval_set=[(va_x, va_y)],
                verbose=verbose_eval,
                use_best_model=True,
                cat_features=self.categoricals
            )
        else:
            # TODO: 全件で学習できるようにする
            pass

    def predict(self, te_x):
        if self.pred_type == 'Regressor':
            return self.model.predict(te_x)
        elif self.pred_type == 'Classifier':
            return self.model.predict_proba(te_x)[:, 1]
        return self.model.predict(te_x)

    def save_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)

    def load_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)
