import os
import numpy as np
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, MaxPooling1D, SimpleRNN)
from custom_layers import PhasedLSTM
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from autoencoder import encoder
from asas_full import preprocess
import keras_util as ku
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

    if args.even:
        X = X[:, :, 1:]

    shuffled_inds = np.random.permutation(np.arange(len(X)))
    train = np.sort(shuffled_inds[:args.N_train])
    valid = np.sort(shuffled_inds[args.N_train:])

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D, 'phased': PhasedLSTM}

    model_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    encode = encoder(model_input, layer=model_type_dict[args.model_type],
                     output_size=args.embedding, **vars(args))
    
    scale_param_array = np.c_[scale_params['means'], scale_params['scales']]
    scale_param_input = Input(shape=(2,), name='scale_params')
    merged = merge([encode, scale_param_input], mode='concat')

    out = Dense(args.size + 2, activation='relu')(merged)
    out = Dense(Y.shape[-1], activation='softmax')(out)
    model = Model([model_input, scale_param_input], out)

    run = ku.get_run_id(**vars(args))

    if args.pretrain:
        model.load_weights(os.path.join('keras2_logs', args.pretrain, run, 'weights.h5'),
                           by_name=True)
#        encoding_index = np.where([l.name == 'encoding' for l in model.layers])[0].item()
#        for layer in model.layers[:encoding_index + 1]:
#            layer.trainable = False  # TODO what should we be allowed to update?

    history = ku.train_and_log([X[train], scale_param_array[train]], Y[train],
                               run, model, metrics=['accuracy'],
                               validation_data=([X[valid], scale_param_array[valid]], Y[valid]),
                               **vars(args))
    return X, X_raw, Y, model, args


if __name__ == '__main__':
    X, X_raw, Y, model, args = main()
