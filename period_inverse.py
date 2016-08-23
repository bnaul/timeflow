from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU, SimpleRNN,
                          Dropout, merge, Reshape, Flatten, RepeatVector)
from keras.models import Model, Sequential
from keras.initializations import normal, identity


def even_lstm_sinusoid(input_len, output_len, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(RepeatVector(n_step, input_shape=(input_len,)))
    model.add(LSTM(size, return_sequences=True))
    for i in range(1, num_layers):
        model.add(Dropout(drop_frac))
        model.add(LSTM(size, return_sequences=True))
    model.add(TimeDistributed(Dense(output_len)))
    return model


def even_gru_sinusoid(input_len, output_len, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(RepeatVector(n_step, input_shape=(input_len,)))
    model.add(GRU(size, return_sequences=True))
    for i in range(1, num_layers):
        model.add(Dropout(drop_frac))
        model.add(GRU(size, return_sequences=True))
    model.add(TimeDistributed(Dense(output_len)))
    return model


def even_relu_sinusoid(input_len, output_len, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(RepeatVector(n_step, input_shape=(input_len,)))
    model.add(SimpleRNN(size, return_sequences=True, init=lambda shape, name:
                        normal(shape, scale=0.001, name=name), inner_init=lambda shape,
                        name: identity(shape, scale=1.0, name=name), activation='relu'))
    for i in range(1, num_layers):
        model.add(Dropout(drop_frac))
        model.add(SimpleRNN(size, return_sequences=True, init=lambda shape, name:
                            normal(shape, scale=0.001, name=name), inner_init=lambda
                            shape, name: identity(shape, scale=1.0,
                                                  name=name), activation='relu'))
    model.add(TimeDistributed(Dense(output_len)))
    return model


def even_sin_sinusoid(input_len, output_len, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(RepeatVector(n_step, input_shape=(input_len,)))
    model.add(TimeDistributed(Dense(output_len)))
    model.add(SimpleRNN(size, return_sequences=True, activation=K.sin))
    for i in range(1, num_layers):
        model.add(Dropout(drop_frac))
        model.add(TimeDistributed(Dense(output_len)))
        model.add(SimpleRNN(size, return_sequences=True, activation=K.sin))
    model.add(TimeDistributed(Dense(output_len)))
    return model


if __name__ == '__main__':
    import argparse
    import os
    import numpy as np
    from keras import backend as K

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
    parser.add_argument("--model_type", type=str, default='lstm')
    parser.add_argument("--gpu_frac", type=float, default=0.31)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=2e-9)
    parser.add_argument("--sim_type", type=str, default='period')
    parser.add_argument("--filter", type=int, default=5)
    parser.add_argument("--N_train", type=int, default=50000)
    parser.add_argument("--N_test", type=int, default=1000)
    parser.add_argument("--n_min", type=int, default=250)
    parser.add_argument("--n_max", type=int, default=250)
    parser.add_argument("--even", type=bool, default=True)
    args = parser.parse_args()

    np.random.seed(0)
    N = args.N_train + args.N_test
    train = np.arange(args.N_train); test = np.arange(args.N_test) + args.N_train
    X, Y = sample_data.periodic(N, args.n_min, args.n_max, t_max=2*np.pi,
                                even=args.even, A_shape=5.,
                                noise_sigma=args.sigma, w_min=0.1, w_max=1.)
    if args.even:
        X = X[:, :, 1:2]

    model_dict = {'lstm': even_lstm_sinusoid, 'gru': even_gru_sinusoid,
#                  'relu': even_relu_sinusoid, 'sin': even_sin_sinusoid
#                  'conv': even_conv_sinusoid,
                 }

    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))
    model = model_dict[args.model_type](output_len=X.shape[-1],
                                        input_len=Y.shape[-1],
                                        n_step=args.n_max, **vars(args))

    run = "{}_{:03d}_x{}_{:1.0e}_drop{}".format(args.model_type, args.size,
                                                args.num_layers, args.lr,
                                                int(100 * args.drop_frac)).replace('e-', 'm')
    if 'conv' in run:
        run += '_f{}'.format(args.filter)
    history = ku.train_and_log(Y[train], X[train], run, model, **vars(args))
