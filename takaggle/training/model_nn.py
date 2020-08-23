import os
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras import optimizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from model import Model
from util import Util


class ModelNN(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット・スケーリング
        validation = va_x is not None

        self.one_hot_encoder = Util.load('one-hot-enc.pkl')
        tr_x = self.one_hot_encoder.transform(tr_x[self.categoricals])

        scaler = StandardScaler()
        # scaler = MinMaxScaler()
        scaler.fit(tr_x)
        tr_x = scaler.transform(tr_x)

        if validation:
            va_x = self.one_hot_encoder.transform(va_x[self.categoricals])
            va_x = scaler.transform(va_x)

        # パラメータ
        classes = self.params['classes']
        layers = self.params['layers']
        dropout = self.params['dropout']
        units = self.params['units']
        nb_epoch = self.params['nb_epoch']
        patience = self.params['patience']

        # モデルの構築
        model = Sequential()
        model.add(Dense(units, input_shape=(tr_x.shape[1],)))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        for l in range(layers - 1):
            units = int(units/2)
            model.add(Dense(units))
            model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        model.add(Dense(classes))
        adam = optimizers.Adam(lr=1e-4)
        model.compile(optimizer=adam, loss="mean_absolute_error")

        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                            verbose=1, restore_best_weights=True)
            save_best = ModelCheckpoint('nn_model.w8', save_weights_only=True, save_best_only=True, verbose=1)
            model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=128, verbose=2,
                        validation_data=(va_x, va_y), callbacks=[save_best, early_stopping])
        else:
            model.fit(tr_x, tr_y, nb_epoch=nb_epoch, batch_size=128, verbose=2)

        # モデル・スケーラーの保持
        model.load_weights('nn_model.w8')
        self.model = model
        self.scaler = scaler

    def predict(self, te_x):
        te_x = self.one_hot_encoder.transform(te_x[self.categoricals])
        te_x = self.scaler.transform(te_x)
        pred = self.model.predict(te_x)
        return np.ravel(pred)  # 1次元に変換する

    def save_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(path, f'{self.run_fold_name}-scaler.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        Util.dump(self.scaler, scaler_path)

    def load_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(path, f'{self.run_fold_name}-scaler.pkl')
        self.model = load_model(model_path)
        self.scaler = Util.load(scaler_path)
        self.one_hot_encoder = Util.load('one-hot-enc.pkl')
