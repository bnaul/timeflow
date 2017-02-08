import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU, SimpleRNN,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, AtrousConv1D)
from keras.models import Model, Sequential
from keras.initializations import normal, identity

import sample_data
import keras_util as ku
from autoencoder import decoder


def main(args=None):
    if args is None:
        args = ku.parse_model_args()

    np.random.seed(0)
    N = args.N_train + args.N_test
    train = np.arange(args.N_train); test = np.arange(args.N_test) + args.N_train
    X, Y, X_raw, labels = sample_data.periodic(N, args.n_min, args.n_max,
                                               even=args.even,
                                               noise_sigma=args.sigma,
                                               kind=args.data_type)

    if args.even:
        X = X[:, :, 1:2]
    else:
        X[:, :, 0] = ku.times_to_lags(X_raw[:, :, 0])
        X[np.isnan(X)] = -1.
        X_raw[np.isnan(X_raw)] = -1.

    Y = sample_data.phase_to_sin_cos(Y)

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D, 'atrous': AtrousConv1D}
    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))

    encode = Input(shape=(Y.shape[1],), name='main_input')
    if args.even:
        model_input = encode
    else:
        model_input = [encode, Input(shape=(X.shape[1], X.shape[-1] - 1),
                                     name='aux_input')]

    decode = decoder(encode, layer=model_type_dict[args.model_type],
                     n_step=X.shape[1], **vars(args))
    model = Model(model_input, decode)

    run = ku.get_run_id(**vars(args))

    if args.even:
        history = ku.train_and_log(Y[train], X[train, :, :], run, model, **vars(args))
    else:
        sample_weight = (X[train, :, -1] != -1)
        history = ku.train_and_log({'main_input': Y[train], 'aux_input': X[train, :, 0:1]},
                                   X_raw[train, :, 1:2], run, model,
                                   sample_weight=sample_weight, **vars(args))
    return X, Y, model, args


if __name__ == '__main__':
    X, Y, model, args = main()
