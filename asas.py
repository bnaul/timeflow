import numpy as np
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
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from autoencoder import encoder
from asas_full import preprocess
import keras_util as ku
#import cesium.datasets
from db_models import db, LightCurve


def main(args=None):
    if not args:
        args = ku.parse_model_args()

    args.loss = 'categorical_crossentropy'

    np.random.seed(0)

    top_classes = ['Mira', 'RR_Lyrae_FM', 'W_Ursae_Maj']
    full = LightCurve.select().where(LightCurve.label << top_classes)
    split = [el for lc in full for el in lc.split(args.n_min, args.n_max)]
    X_list = [np.c_[lc.times, lc.measurements, lc.errors] for lc in split]

    classnames, indices = np.unique([lc.label for lc in split], return_inverse=True)
    Y = to_categorical(indices, len(classnames))

    X_raw = pad_sequences(X_list, value=0., dtype='float', padding='post')
    X, scale_params = preprocess(X_raw, args.m_max, True, True, True)
    Y = Y[~scale_params['wrong_units']]

    # Remove errors
    X = X[:, :, :2]

    train = np.sort(np.random.choice(np.arange(len(X)), int(len(X) * 0.8), replace=False))
    valid = np.arange(len(X))[~np.in1d(np.arange(len(X)), train)]

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D, 'atrous': AtrousConv1D, 'phased': PhasedLSTM}

    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))

    model_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    encode = encoder(model_input, layer=model_type_dict[args.model_type], **vars(args))
    
    scale_param_input = Input(shape=(2,), name='scale_params')
    merged = merge([encode, scale_param_input], mode='concat')

    out = Dense(args.size + 2, activation='relu')(merged)
    out = Dense(Y.shape[-1], activation='softmax')(out)
    model = Model([model_input, scale_param_input], out)

    run = ku.get_run_id(**vars(args))

    history = ku.train_and_log([X[train], np.c_[scale_params['means'],
                                                scale_params['scales']][train]],
                               Y[train], run, model, metrics=['accuracy'],
                               validation_data=([X[valid], np.c_[scale_params['means'],
                                                       scale_params['scales']][valid]],
                                                Y[valid]),
                               **vars(args))
    return X, X_raw, Y, model, args


if __name__ == '__main__':
    X, X_raw, Y, model, args = main()
