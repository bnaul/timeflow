import numpy as np
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector)
from keras.models import Model, Sequential

import sample_data
import keras_util as ku


def even_lstm_period_estimator(output_len, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(LSTM(size, input_shape=(n_max, 1),
                   return_sequences=(num_layers > 1)))
    for i in range(1, num_layers):
        model.add(Dropout(drop_frac))
        model.add(LSTM(size, return_sequences=(i != num_layers - 1)))
    model.add(Dense(output_len, activation='linear'))
    return model


def even_gru_period_estimator(output_len, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(GRU(size, input_shape=(n_max, 1),
                   return_sequences=(num_layers > 1)))
    for i in range(1, num_layers):
        model.add(Dropout(drop_frac))
        model.add(GRU(size, return_sequences=(i != num_layers - 1)))
    model.add(Dense(output_len, activation='linear'))
    return model


if __name__ == '__main__':
    import argparse
    import os
    import numpy as np
    from keras import backend as K

    import sample_data
    import keras_util as ku

    np.random.seed(0)
    SIM_TYPE = os.path.splitext(os.path.basename(__file__))[0]
    N_train = 50000; N_test = 1000
    N = N_train + N_test
    train = np.arange(N_train); test = np.arange(N_test) + N_train
    n_min = 250; n_max = 250
    X, Y = sample_data.periodic(N, n_min, n_max, t_max=2*np.pi, even=True,
                                A_shape=5., noise_sigma=2e-9, w_min=0.1,
                                w_max=1.)
    X = X[:, :, 1:2]

    model_dict = {'lstm': even_lstm_period_estimator, 'gru': even_gru_period_estimator}
#                  'relu': even_relu_period_estimator, 'sin': even_sin_period_estimator}

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

    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))
    model = model_dict[args.model_type](output_len=Y.shape[-1], n_step=n_max,
                                        **vars(args))

    run = "{}_{:03d}_x{}_{:1.0e}_drop{}".format(args.model_type, args.size,
                                                args.num_layers, args.lr,
                                                int(100 * args.drop_frac)).replace('e-', 'm')
    history = ku.train_and_log(X[train], Y[train], run, model,
                               sim_type=SIM_TYPE, **vars(args))
