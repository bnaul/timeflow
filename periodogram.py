import numpy as np
from scipy.fftpack import dct
from scipy.signal import lombscargle
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, AtrousConv1D, MaxPooling1D, SimpleRNN)
from keras.models import Model, Sequential

from autoencoder import encoder
import sample_data
import keras_util as ku


def main(args=None):
    args = ku.parse_model_args(args)

    np.random.seed(0)
    N = args.N_train + args.N_test
    train = np.arange(args.N_train); test = np.arange(args.N_test) + args.N_train
    X, Y, X_raw, labels = sample_data.periodic(N, args.n_min, args.n_max,
                                               even=args.even,
                                               noise_sigma=args.sigma,
                                               kind=args.data_type)

    if args.even:
        X = X[:, :, 1:2]
        F = dct(X)[:, :, 0]
    else:
#        freqs = np.linspace(0., 2 * np.pi, args.n_max + 1)[1:]
        freqs = np.array([0.5])
        F = np.zeros((X.shape[0], freqs.shape[0]))
        for i in range(X.shape[0]):
            F[i] = lombscargle(X[i, :, 0], X[i, :, 1], freqs)

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D, 'atrous': AtrousConv1D}

    model_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    encode = encoder(model_input, layer=model_type_dict[args.model_type], **vars(args))
    out = Dense(F.shape[-1])(encode)
    model = Model(model_input, out)

    run = ku.get_run_id(**vars(args))
 
    history = ku.train_and_log(X[train], F[train], run, model, **vars(args))
    return X, Y, F, model


if __name__ == '__main__':
    X, Y, F, model = main()
