import numpy as np
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, AtrousConv1D, MaxPooling1D, SimpleRNN)
from keras.models import Model, Sequential

from autoencoder import encoder
import sample_data
import keras_util as ku


def main(args=None):
    if not args:
        args = ku.parse_model_args()

    np.random.seed(0)
    N = args.N_train + args.N_test
    train = np.arange(args.N_train); test = np.arange(args.N_test) + args.N_train
    X, Y = sample_data.periodic(N, args.n_min, args.n_max, t_max=2*np.pi, even=args.even,
                                A_shape=5., noise_sigma=args.sigma, w_min=0.1,
                                w_max=1.)

    # freq amp phase -> freq cos_amp sin_amp
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

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D, 'atrous': AtrousConv1D}
    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))

    model_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    encode = encoder(model_input, layer=model_type_dict[args.model_type], **vars(args))
    out = Dense(Y.shape[-1])(encode)
    model = Model(model_input, out)

    run = ku.get_run_id(**vars(args))
 
    history = ku.train_and_log(X[train], Y[train], run, model, **vars(args))
    return X, Y, model


if __name__ == '__main__':
    X, Y, model = main()
