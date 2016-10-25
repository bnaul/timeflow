import numpy as np
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, AtrousConv1D, MaxPooling1D)
from keras.models import Model, Sequential

import sample_data
import keras_util as ku


def lstm_periodogram_estimator(output_len, input_size, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(LSTM(size, input_shape=(n_step, input_size),
                   return_sequences=(num_layers > 1)))
    for i in range(1, num_layers):
        model.add(Dropout(drop_frac))
        model.add(LSTM(size, return_sequences=(i != num_layers - 1)))
    model.add(Dense(output_len, activation='linear'))
    return model


def gru_periodogram_estimator(output_len, input_size, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(GRU(size, input_shape=(n_step, input_size),
                   return_sequences=(num_layers > 1)))
    for i in range(1, num_layers):
        model.add(Dropout(drop_frac))
        model.add(GRU(size, return_sequences=(i != num_layers - 1)))
    model.add(Dense(output_len, activation='linear'))
    return model


def conv_periodogram_estimator(output_len, input_size, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(Conv1D(size, kwargs['filter'], activation='relu',
                     input_shape=(n_step, input_size)))
#    model.add(MaxPooling1D(5))
    model.add(Dropout(drop_frac))
    for i in range(1, num_layers):
        model.add(Conv1D(size, kwargs['filter'], activation='relu',
                         input_shape=(n_step, input_size)))
#        model.add(MaxPooling1D(5))
        model.add(Dropout(drop_frac))
    model.add(Flatten())
    model.add(Dense(size, activation='relu'))
    model.add(Dense(output_len, activation='linear'))
    return model


def atrous_periodogram_estimator(output_len, input_size, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    # TODO try tanh * sigmoid activation, other intializations?
    model.add(AtrousConv1D(size, kwargs['filter'], activation='relu',
                           input_shape=(n_step, input_size),
                           border_mode='valid',
#                           causal=True,
                           atrous_rate=1))
    model.add(Dropout(drop_frac))
    for i in range(1, num_layers):
        model.add(AtrousConv1D(size, kwargs['filter'], activation='relu',
                               input_shape=(n_step, input_size),
                               border_mode='valid',
#                               causal=True,
                               atrous_rate=2 ** i))
        model.add(Dropout(drop_frac))
    model.add(Flatten())
    model.add(Dense(size, activation='relu'))
    model.add(Dense(output_len, activation='linear'))
    return model


if __name__ == '__main__':
    import argparse
    import os
    import numpy as np
    from keras import backend as K
    from scipy.fftpack import fft, dct
    from gatspy.periodic import LombScargleFast

    import sample_data
    import keras_util as ku

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int)
    parser.add_argument("num_layers", type=int)
    parser.add_argument("drop_frac", type=float)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--nb_epoch", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--loss", type=str, default='mse')
    parser.add_argument("--loss_weights", type=float, nargs='*')
    parser.add_argument("--model_type", type=str, default='lstm')
    parser.add_argument("--gpu_frac", type=float, default=0.31)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=2e-9)
    parser.add_argument("--sim_type", type=str, default='periodogram')
    parser.add_argument("--filter", type=int, default=5)
    parser.add_argument("--N_train", type=int, default=50000)
    parser.add_argument("--N_test", type=int, default=1000)
    parser.add_argument("--n_min", type=int, default=50)
    parser.add_argument("--n_max", type=int, default=50)
    parser.add_argument('--even', dest='even', action='store_true')
    parser.add_argument('--uneven', dest='even', action='store_false')
    parser.add_argument('--no_train', dest='no_train', action='store_true')
    parser.set_defaults(even=True)
    args = parser.parse_args()

    np.random.seed(0)
    N = args.N_train + args.N_test
    train = np.arange(args.N_train); test = np.arange(args.N_test) + args.N_train
    X, Y = sample_data.periodic(N, args.n_min, args.n_max, t_max=2*np.pi, even=args.even,
                                A_shape=5., noise_sigma=args.sigma, w_min=0.1,
                                w_max=1.)
    Y[:, 0] **= -1  # period instead of frequency
    
    if args.loss_weights:
        Y *= args.loss_weights

    if args.even:
        X = X[:, :, 1:2]
        F = dct(X)[:, :, 0]
    else:
        raise NotImplementedError("Use Lomb-Scargle here")

    model_dict = {'lstm': lstm_periodogram_estimator, 'gru': gru_periodogram_estimator,
                  'conv': conv_periodogram_estimator, 'atrous': atrous_periodogram_estimator}

    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))
    model = model_dict[args.model_type](output_len=F.shape[-1],
                                        input_size=X.shape[-1],
                                        n_step=args.n_max, **vars(args))

    run = "{}_{:03d}_x{}_{:1.0e}_drop{}".format(args.model_type, args.size,
                                                args.num_layers, args.lr,
                                                int(100 * args.drop_frac)).replace('e-', 'm')
    if 'conv' in run:
        run += '_f{}'.format(args.filter)
    history = ku.train_and_log(X[train], F[train], run, model, **vars(args))
