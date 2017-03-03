import glob
import os
import numpy as np
import pandas as pd
import joblib
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, AtrousConv1D, MaxPooling1D, SimpleRNN)
from custom_layers import PhasedLSTM
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences

import keras_util as ku
from autoencoder import encoder, decoder
from light_curve import LightCurve


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

    full = joblib.load('data/asas/light_curves.pkl')
    if args.lomb_score:
        full = [lc for lc in full if lc.best_score >= args.lomb_score]
    split = [el for lc in full for el in lc.split(args.n_min, args.n_max)]
    X_list = [np.c_[lc.times, lc.measurements, lc.errors] for lc in split]

    X_raw = pad_sequences(X_list, value=np.nan, dtype='float', padding='post')

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D, 'atrous': AtrousConv1D, 'phased': PhasedLSTM}
    X, scale_params = preprocess(X_raw, args.m_max, True, True, True)
    main_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    aux_input = Input(shape=(X.shape[1], X.shape[-1] - 1), name='aux_input')
    model_input = [main_input, aux_input]
    encode = encoder(main_input, layer=model_type_dict[args.model_type], 
                     output_size=args.embedding, **vars(args))
    decode = decoder(encode, num_layers=args.decode_layers if args.decode_layers
                                                           else args.num_layers,
                     layer=model_type_dict[args.decode_type if args.decode_type
                                           else args.model_type],
                     n_step=X.shape[1], aux_input=aux_input,
                     **{k: v for k, v in vars(args).items() if k != 'num_layers'})
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
