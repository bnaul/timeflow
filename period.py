import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import sample_data

batch_size = 500
nb_epoch = 200
gpu_frac = 0.31

np.random.seed(0)
N_train = 50000; N_test = 1000
N = N_train + N_test
train = np.arange(N_train); test = np.arange(N_test) + N_train
n_min = 250; n_max = 250

X, Y = sample_data.periodic(N, n_min, n_max, t_max=2*np.pi, even=True,
                            A_shape=5., noise_sigma=2e-9, w_min=0.1,
                            w_max=1.)
X = X[:, :, 1:2]

from keras.callbacks import Callback
from IPython.core.display import clear_output

class ClearOutput(Callback):
    def on_epoch_end(self, epoch, logs={}):
        clear_output(wait=True)

import os
import shutil
import tensorflow as tf
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM,
                          Dropout, merge, Reshape, Flatten, RepeatVector)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ProgbarLogger, TensorBoard
from keras.utils.visualize_util import model_to_dot
from IPython.display import clear_output, SVG, display

def even_lstm_period_estimator(lstm_size, num_layers, drop_frac, lr):
    print("gpu frac: ", gpu_frac)
    if gpu_frac is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sess = tf.Session()
    else:
        gpu_opts = tf.ConfigProto(gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_frac))
        sess = tf.Session(config=gpu_opts)

    K.set_session(sess)
    run = "lstm_{:03d}_x{}_{:1.0e}_drop{}".format(lstm_size, num_layers, lr, drop_frac).replace('e-', 'm').replace('.', '')
    print(run)
    model = Sequential()
    model.add(LSTM(lstm_size, input_shape=(n_max, X.shape[-1]),
                   return_sequences=(num_layers > 1)))
    for i in range(1, num_layers):
        model.add(Dropout(drop_frac))
        model.add(LSTM(lstm_size, return_sequences=(i != num_layers - 1)))
    model.add(Dense(Y.shape[-1], activation='linear'))
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='mse', metrics=[])

    log_dir = os.path.expanduser('~/Dropbox/Documents/timeflow/keras_logs/period/{}'.format(run))
    shutil.rmtree(log_dir, ignore_errors=True)
    history = model.fit(X[train], Y[train], 
                         nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.2,
                         callbacks=[ProgbarLogger(), TensorBoard(log_dir=log_dir, write_graph=False)])
    model.save_weights(os.path.join(log_dir, 'weights.h5'), overwrite=True)
    return (run, history, model)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("lstm_size", type=int)
    parser.add_argument("num_layers", type=int)
    parser.add_argument("drop_frac", type=float)
    args = parser.parse_args()
                
    run, history, model = even_lstm_period_estimator(args.lstm_size,
                                                     args.num_layers,
                                                     args.drop_frac, 2e-3)
