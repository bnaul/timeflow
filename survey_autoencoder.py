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
def preprocess(X_raw, m_max=None):
    X = X_raw.copy()

    if m_max:
        wrong_units = np.nanmax(X[:, :, 1], axis=1) > m_max
        X = X[~wrong_units, :, :]

    # Replace times w/ lags
    X[:, :, 0] = ku.times_to_lags(X[:, :, 0])

    means = np.atleast_2d(np.nanmean(X[:, :, 1], axis=1)).T
    X[:, :, 1] -= means

#    scales = np.atleast_2d(np.nanmax(np.abs(X[:, :, 1]), axis=1)).T
    scales = np.atleast_2d(np.std(X[:, :, 1], axis=1)).T
    X[:, :, 1] /= scales

    # drop_errors
    X = X[:, :, :2]

    return X, means, scales, wrong_units


def main(args=None):
    if not args:
        args = ku.parse_model_args()

    np.random.seed(0)

    if not args.survey_files:
        raise ValueError("No survey files given")
    lc_lists = [joblib.load(f) for f in args.survey_files]
    n_reps = [max(len(y) for y in lc_lists) // len(x) for x in lc_lists]
    combined = sum([x * i for x, i in zip(lc_lists, n_reps)], [])
    if args.lomb_score:
        combined = [lc for lc in combined if lc.best_score >= args.lomb_score]
    split = [el for lc in combined for el in lc.split(args.n_min, args.n_max)]
    X_list = [np.c_[lc.times, lc.measurements, lc.errors] for lc in split]

    X_raw = pad_sequences(X_list, value=np.nan, dtype='float', padding='post')

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D, 'atrous': AtrousConv1D, 'phased': PhasedLSTM}
    X, means, scales, wrong_units = preprocess(X_raw, args.m_max)
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

#    sample_weight = (~np.isnan(X[:, :, -1])).astype('float')
    sample_weight = 1. / X_raw[:, :, 2]
    sample_weight = (sample_weight.T / np.nanmean(sample_weight, axis=1)).T
    sample_weight[np.isnan(sample_weight)] = 0.0
    X[np.isnan(X)] = -1.

    errors = X_raw[:, :, 2] / scales

    history = ku.train_and_log({'main_input': X, 'aux_input': np.delete(X, 1, axis=2)},
                               X[:, :, [1]], run, model, sample_weight=sample_weight,
                               errors=errors, **vars(args))

    return X, X_raw, model, args


if __name__ == '__main__':
    X, X_raw, model, args = main()
