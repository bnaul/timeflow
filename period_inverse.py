import numpy as np
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector)
from keras.models import Model, Sequential

import sample_data
import keras_util as ku


def even_lstm_sinusoid(X, Y, size, num_layers, drop_frac, lr, loss, batch_size, nb_epoch):
    model = Sequential()
    model.add(RepeatVector(n_max,  input_shape=(X.shape[-1], )))
    model.add(LSTM(size, return_sequences=True))
    for i in range(1, num_layers):
        model.add(Dropout(drop_frac))
        model.add(LSTM(size, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))

    run = "lstm_{:03d}_x{}_{:1.0e}_drop{}".format(size, num_layers, lr,
                                                  drop_frac).replace('e-', 'm').replace('.', '')
    history, model = ku.train_and_log(X, Y, run, model, nb_epoch, batch_size, lr, loss,
                                      SIM_TYPE)

    return (run, history, model)


def even_gru_sinusoid(X, Y, size, num_layers, drop_frac, lr, loss, batch_size, nb_epoch):
    model = Sequential()
    model.add(RepeatVector(n_max,  input_shape=(X.shape[-1], )))
    model.add(GRU(size, return_sequences=True))
    for i in range(1, num_layers):
        model.add(Dropout(drop_frac))
        model.add(GRU(size, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))

    run = "gru_{:03d}_x{}_{:1.0e}_drop{}".format(size, num_layers, lr,
                                                  drop_frac).replace('e-', 'm').replace('.', '')
    history, model = ku.train_and_log(X, Y, run, model, nb_epoch, batch_size, lr, loss,
                                      SIM_TYPE)

    return (run, history, model)


if __name__ == '__main__':
    np.random.seed(0)
    SIM_TYPE = 'period_inverse'
    N_train = 50000; N_test = 1000
    N = N_train + N_test
    train = np.arange(N_train); test = np.arange(N_test) + N_train
    n_min = 100; n_max = 100

    model_dict = {'lstm': even_lstm_sinusoid, 'gru': even_gru_sinusoid}

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int)
    parser.add_argument("num_layers", type=int)
    parser.add_argument("drop_frac", type=float)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--nb_epoch", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--loss", type=str, default='mse')
    parser.add_argument("--model_type", type=str, default='lstm')
    parser.add_argument("--gpu_frac", type=float, default=0.31)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    X, Y = sample_data.periodic(N, n_min, n_max, t_max=2*np.pi, even=True,
                                A_shape=5., noise_sigma=2e-9, w_min=0.1,
                                w_max=1.)
    X = X[:, :, 1:2]

    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))
    model_fn = model_dict[args.model_type]
    run, history, model = model_fn(Y[train], X[train], args.size,
                                   args.num_layers,
                                   args.drop_frac, args.lr,
                                   args.loss, args.batch_size,
                                                 args.nb_epoch)
