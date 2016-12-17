import glob
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, AtrousConv1D, MaxPooling1D, SimpleRNN)
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences

import keras_util as ku
from autoencoder import encoder, decoder


def preprocess(X_raw, m_max):
    X = X_raw.copy()

    wrong_units = np.nanmax(X[:, :, 1], axis=1) > m_max
    X = X[~wrong_units, :, :]

    # Replace times w/ lags
    X[:, :, 0] = ku.times_to_lags(X[:, :, 0])

    # Subtract mean magnitude
    global_mean = np.nanmax(X[:, :, 1])
    X[:, :, 1] -= global_mean

    return X, {}


def main(args=None):
    if not args:
        args = ku.parse_model_args()

    np.random.seed(0)

    filenames = glob.glob('./data/survey_lcs/*')
    lengths = {f: sum(1 for line in open(f)) for f in filenames}
    filenames = [f for f in filenames if lengths[f] >= args.n_min
                                      and lengths[f] <= args.n_max]
    filenames = sorted(filenames, key=lambda f: lengths[f])

    X_list = [pd.read_csv(f, header=None).sort_values(by=0).values
              for f in filenames[:args.first_N]]

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D, 'atrous': AtrousConv1D}
    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))
    X_raw = pad_sequences(X_list, value=np.nan, dtype='float', padding='post')
    X, scale_params = preprocess(X_raw, args.m_max)
    main_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    aux_input = Input(shape=(X.shape[1], X.shape[-1] - 1), name='aux_input')
    model_input = [main_input, aux_input]
    encode = encoder(main_input, layer=model_type_dict[args.model_type], **vars(args))
    decode = decoder(encode, layer=model_type_dict[args.model_type], n_step=X.shape[1],
                     aux_input=aux_input, **vars(args))
    model = Model(model_input, decode)

    run = ku.get_run_id(**vars(args))

    sample_weight = np.isnan(X[:, :, -1]).astype('float')
    X[np.isnan(X)] = -1.
    history = ku.train_and_log({'main_input': X, 'aux_input': X[:, :, [0, 2]]}, X[:, :, 1:2],
                               run, model, sample_weight=sample_weight, **vars(args))

    return X, X_raw, model, args


if __name__ == '__main__':
    X, X_raw, model, args = main()
