import numpy as np
np.random.seed(0)
from sklearn.preprocessing import StandardScaler
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, AtrousConv1D, MaxPooling1D, SimpleRNN)
from custom_layers import PhasedLSTM
from keras.models import Model, Sequential

from autoencoder import encoder
import sample_data
import keras_util as ku


def main(args=None):
    if not args:
        args = ku.parse_model_args()

    N = args.N_train + args.N_test
    train = np.arange(args.N_train); test = np.arange(args.N_test) + args.N_train
    X, Y, X_raw, labels = sample_data.periodic(N, args.n_min, args.n_max,
                                                   even=args.even,
                                                   noise_sigma=args.sigma,
                                                   kind=args.data_type,
                                                   # t_scale=0.05
                                                  )

    if args.even:
        X = X[:, :, 1:2]
        if args.model_type == 'phased':
            raise NotImplementedError("Phased LSTM not implemented for --even")
    else:
        X[:, :, 0] = ku.times_to_lags(X_raw[:, :, 0])
        X[np.isnan(X)] = -1.
        X_raw[np.isnan(X_raw)] = -1.

    Y = sample_data.phase_to_sin_cos(Y)
    scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
    scaler.fit_transform(Y)
    if args.loss_weights:  # in practice, only used to zero out some columns
        Y *= args.loss_weights

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D, 'atrous': AtrousConv1D, 'phased': PhasedLSTM}

    model_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    encode = encoder(model_input, layer=model_type_dict[args.model_type],
                     output_size=Y.shape[-1], **vars(args))
    model = Model(model_input, encode)

    run = ku.get_run_id(**vars(args))

    history = ku.train_and_log(X[train], Y[train], run, model, **vars(args))
    return X, Y, X_raw, scaler, model, args


if __name__ == '__main__':
    X, Y, X_raw, scaler, model, args = main()
