import glob
import os
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, AtrousConv1D, MaxPooling1D, SimpleRNN)
try:
    from keras.layers import PhasedLSTM
except:
    PhasedLSTM = None
    print("Skipping PhasedLSTM...")
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences

import keras_util as ku
from autoencoder import encoder, decoder
from db_models import db, LightCurve


# TODO interpolate at different time points
def preprocess(X_raw, m_max=None, center=False, scale=False, drop_errors=True):
    X = X_raw.copy()

    if m_max:
        wrong_units = np.nanmax(X[:, :, 1], axis=1) > m_max
        X = X[~wrong_units, :, :]

    # Replace times w/ lags
    X[:, :, 0] = ku.times_to_lags(X[:, :, 0])

    if center:
        means = np.atleast_2d(np.nanmean(X[:, :, 1], axis=1)).T
        X[:, :, 1] -= means
    else:
        means = None
    if scale:
        scales = np.atleast_2d(np.nanmax(np.abs(X[:, :, 1]), axis=1)).T
        X[:, :, 1] /= scales
    else:
        scales = None

    if drop_errors:
        X = X[:, :, :2]

    return X, {'means': means, 'scales': scales, 'wrong_units': wrong_units}


def main(args=None):
    if not args:
        args = ku.parse_model_args()

    np.random.seed(0)

#    filenames = glob.glob('./data/asas/*/*')
#    # use old sort for pandas backwards compatibility
#    X_list = [pd.read_csv(f, header=None, comment='#', delim_whitespace=True,
#                           names=['t', 'm', 'm1', 'm2', 'm3', 'm4', 'e',
#                                  'e1', 'e2', 'e3', 'e4', 'grade',
#                                  'frame']).sort(columns='t')
#               for f in filenames]
#    X_list = [x[x.grade < 'C'] for x in X_list]
##    for x in X_list:
##        x['m'] = x[['m' + str(i) for i in range(5)]].mean(axis=1)
##        x['e'] = x[['e' + str(i) for i in range(5)]].mean(axis=1)
#    X_list = [x[['t', 'm', 'e']].values for x in X_list]
#    # split into length n_min chunks
#    X_list = [x[np.abs(x[:, 1] - np.median(x[:, 1])) <= 8, :] for x in X_list]
#    X_list = [el for x in X_list
#              for el in np.array_split(x, np.arange(args.n_max, len(x), step=args.n_max))]
#    X_list = [x for x in X_list if len(x) >= args.n_min]
    full = LightCurve.select()
    if args.lomb_score:
        full = full.where(LightCurve.best_score >= args.lomb_score)
    split = [el for lc in full for el in lc.split(args.n_min, args.n_max)]
    X_list = [np.c_[lc.times, lc.measurements, lc.errors] for lc in split]
    X_raw = pad_sequences(X_list, value=np.nan, dtype='float', padding='post')

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D, 'atrous': AtrousConv1D, 'phased': PhasedLSTM}
    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))
    X, scale_params = preprocess(X_raw, args.m_max, True, True, True)
    main_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    aux_input = Input(shape=(X.shape[1], X.shape[-1] - 1), name='aux_input')
    model_input = [main_input, aux_input]
    encode = encoder(main_input, layer=model_type_dict[args.model_type], 
                     output_size=args.embedding, **vars(args))
    decode = decoder(encode, layer=model_type_dict[args.decode_type if args.decode_type
                                                   else args.model_type],
                     n_step=X.shape[1], aux_input=aux_input, **vars(args))
    model = Model(model_input, decode)

    run = ku.get_run_id(**vars(args))

    sample_weight = (~np.isnan(X[:, :, -1])).astype('float')
    X[np.isnan(X)] = -1.
    history = ku.train_and_log({'main_input': X, 'aux_input': np.delete(X, 1, axis=2)},
                               X[:, :, 1:], run, model, sample_weight=sample_weight,
                               **vars(args))

    return X, X_raw, model, args


if __name__ == '__main__':
    X, X_raw, model, args = main()
