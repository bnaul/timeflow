# TODO update
import numpy as np
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, MaxPooling1D)
from keras.models import Model, Sequential

import sample_data
import keras_util as ku


def even_dense_classifier(output_len, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(Dense(size, input_shape=(n_step,), activation='relu'))
    model.add(Dropout(drop_frac))
    for i in range(1, num_layers):
        model.add(Dense(size, activation='relu'))
        model.add(Dropout(drop_frac))
    model.add(Dense(output_len, activation='softmax'))
    return model


def even_lstm_classifier(output_len, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(LSTM(size, input_shape=(n_step, 1),
                   return_sequences=(num_layers > 1)))
    model.add(Dropout(drop_frac))
    for i in range(1, num_layers):
        model.add(LSTM(size, return_sequences=(i != num_layers - 1)))
        model.add(Dropout(drop_frac))
    model.add(Dense(output_len, activation='softmax'))
    return model


def even_gru_classifier(output_len, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(GRU(size, input_shape=(n_step, 1),
                   return_sequences=(num_layers > 1)))
    model.add(Dropout(drop_frac))
    for i in range(1, num_layers):
        model.add(GRU(size, return_sequences=(i != num_layers - 1)))
        model.add(Dropout(drop_frac))
    model.add(Dense(output_len, activation='softmax'))
    return model


def even_conv_classifier(output_len, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(Conv1D(size, kwargs['filter'], activation='relu', input_shape=(n_step, 1)))
#    model.add(MaxPooling1D(5))
    model.add(Dropout(drop_frac))
    for i in range(1, num_layers):
        model.add(Conv1D(size, kwargs['filter'], activation='relu', input_shape=(n_step, 1)))
#        model.add(MaxPooling1D(5))
        model.add(Dropout(drop_frac))
    model.add(Flatten())
    model.add(Dense(size, activation='relu'))
    model.add(Dense(output_len, activation='softmax'))
    return model


if __name__ == '__main__':
    import argparse
    import os
    import numpy as np
    from keras.utils.np_utils import to_categorical

    import sample_data
    import keras_util as ku

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int)
    parser.add_argument("num_layers", type=int)
    parser.add_argument("drop_frac", type=float)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--loss", type=str, default='categorical_crossentropy')
    parser.add_argument("--model_type", type=str, default='lstm')
    parser.add_argument("--gpu_frac", type=float, default=0.31)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=2e-9)
    parser.add_argument("--sim_type", type=str, default='classification')
    parser.add_argument("--metrics", nargs='+', default=['accuracy'])
    parser.add_argument("--filter", type=int, default=5)
    args = parser.parse_args()

    np.random.seed(0)
    N_train = 50000; N_test = 0
    N = N_train + N_test
    train = np.arange(N_train); test = np.arange(N_test) + N_train
    n_min = 250; n_max = 250
    X_low, _ = sample_data.periodic(int(N / 2), n_min, n_max, t_max=2*np.pi,
                                    even=True, A_shape=5.,
                                    noise_sigma=args.sigma, w_min=0.1,
                                    w_max=0.5)
    X_high, _ = sample_data.periodic(int(N / 2), n_min, n_max, t_max=2*np.pi,
                                     even=True, A_shape=5.,
                                     noise_sigma=args.sigma, w_min=0.5,
                                     w_max=1.0)
    X = np.row_stack((X_low, X_high))[:, :, 1:2]
    Y = to_categorical(np.row_stack((np.zeros((int(N / 2), 1), dtype=int),
                                     np.ones((int(N / 2), 1), dtype=int))), 2)

    model_dict = {'lstm': even_lstm_classifier, 'gru': even_gru_classifier,
                  'conv': even_conv_classifier}

    model = model_dict[args.model_type](output_len=Y.shape[-1], n_step=n_max,
                                        **vars(args))

    run = "{}_{:03d}_x{}_{:1.0e}_drop{}".format(args.model_type, args.size,
                                                args.num_layers, args.lr,
                                                int(100 * args.drop_frac)).replace('e-', 'm')
    if 'conv' in run:
        run += '_f{}'.format(args.filter)
    history = ku.train_and_log(X[train], Y[train], run, model, **vars(args))
