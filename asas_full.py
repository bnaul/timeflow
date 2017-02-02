import glob
import os
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, AtrousConv1D, MaxPooling1D, SimpleRNN)
from custom_layers import PhasedLSTM
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences

import keras_util as ku
from autoencoder import encoder, decoder
try:
    from db_models import LightCurve
except:
    LightCurve = None


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

    if LightCurve is not None:
        full = LightCurve.select()
        if args.lomb_score:
            full = full.where(LightCurve.best_score >= args.lomb_score)
        split = [el for lc in full for el in lc.split(args.n_min, args.n_max)]
        X_list = [np.c_[lc.times, lc.measurements, lc.errors] for lc in split]
    else:
        print("DB could not be loaded; reading from disk...")
        from gatspy.periodic import LombScargleFast
        try:
            from StringIO import StringIO
        except:
            from io import StringIO
        X_list = []
        for fname in glob.glob('./data/asas/*/*'):
            with open(fname) as f:
                dfs = [pd.read_csv(StringIO(chunk), comment='#', delim_whitespace=True) for chunk in f.read().split('#     ')[1:]]
                if len(dfs) > 0:
                    df = pd.concat(dfs)[['HJD', 'MAG_0', 'MER_0', 'GRADE']].sort(columns='HJD')
                    df = df[df.GRADE <= 'B']
                    df.drop_duplicates(subset=['HJD'], inplace=True)  #keep='first'
                    X_i = df[['HJD', 'MAG_0', 'MER_0']].values

                    if args.lomb_score:
                        ls_params = {'period_range': (0.005 * (max(X_i[:, 0]) - min(X_i[:, 0])),
                                                      0.95 * (max(X_i[:, 0]) - min(X_i[:, 0]))),
                                     'quiet': True}
                        model_gat = LombScargleFast(fit_period=True,
                                                    optimizer_kwds=ls_params,
                                                    silence_warnings=True)
                        model_gat.fit(X_i[:, 0], X_i[:, 1], X_i[:, 2])
                        if model_gat.score(model_gat.best_period) < args.lomb_score:
                            continue

                    X_list.append(X_i)

        # split into length n_min chunks
        X_list = [el for x in X_list
                  for el in np.array_split(x, np.arange(args.n_max, len(x), step=args.n_max))]
        X_list = [x for x in X_list if len(x) >= args.n_min]

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
