import numpy as np
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, AtrousConv1D, MaxPooling1D, SimpleRNN)
from keras.models import Model, Sequential

from autoencoder import rnn_encoder
import sample_data
import keras_util as ku


def conv_period_estimator(output_len, input_size, n_step, size, num_layers, drop_frac, **kwargs):
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


def atrous_period_estimator(output_len, input_size, n_step, size, num_layers, drop_frac, **kwargs):
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
    import sample_data
    np.random.seed(0)
    args = ku.parse_model_args()
    N = args.N_train + args.N_test
    train = np.arange(args.N_train); test = np.arange(args.N_test) + args.N_train
    X, Y = sample_data.periodic(N, args.n_min, args.n_max, t_max=2*np.pi, even=args.even,
                                A_shape=5., noise_sigma=args.sigma, w_min=0.1,
                                w_max=1.)

    # freq amp phase -> freq cos_amp sin_amp
    if not args.phase_format:
        A = Y[:, 1] * np.sin(Y[:, 2])
        B = Y[:, 1] * np.cos(Y[:, 2])
        Y[:, 1] = A
        Y[:, 2] = B

    Y[:, 0] **= -1  # period instead of frequency
    if args.loss_weights:
        Y *= args.loss_weights


    if args.even:
        X = X[:, :, 1:2]
    else:
        X[:, :, 0] = np.c_[np.diff(X[:, :, 0]), np.zeros(X.shape[0])]

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN}
    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))

    model_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    encode = rnn_encoder(model_input, layer=model_type_dict[args.model_type],
                         size=args.size, num_layers=args.num_layers,
                         drop_frac=args.drop_frac)
    out = Dense(Y.shape[-1])(encode)
    model = Model(model_input, out)

    run = ku.get_run_id(**vars(args))
 
    history = ku.train_and_log(X[train], Y[train], run, model, **vars(args))
