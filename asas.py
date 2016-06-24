import numpy as np
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector)
from keras.models import Model, Sequential

import sample_data
import keras_util as ku


def uneven_lstm_classifier(output_len, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(LSTM(size, input_shape=(n_max, 3), # with errors
                   return_sequences=(num_layers > 1)))
    for i in range(1, num_layers):
        model.add(Dropout(drop_frac))
        model.add(LSTM(size, return_sequences=(i != num_layers - 1)))
    model.add(Dense(output_len, activation='softmax'))
    return model


def uneven_gru_classifier(output_len, n_step, size, num_layers, drop_frac, **kwargs):
    model = Sequential()
    model.add(GRU(size, input_shape=(n_max, 3), # with errors
                   return_sequences=(num_layers > 1)))
    for i in range(1, num_layers):
        model.add(Dropout(drop_frac))
        model.add(GRU(size, return_sequences=(i != num_layers - 1)))
    model.add(Dense(output_len, activation='softmax'))
    return model


if __name__ == '__main__':
    import argparse
    import os
    import numpy as np
    from keras import backend as K

    import sample_data
    import keras_util as ku

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int)
    parser.add_argument("num_layers", type=int)
    parser.add_argument("drop_frac", type=float)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--nb_epoch", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--loss", type=str, default='categorical_crossentropy')
    parser.add_argument("--model_type", type=str, default='gru')
    parser.add_argument("--gpu_frac", type=float, default=0.31)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=2e-9)
    parser.add_argument("--sim_type", type=str, default='asas')
    parser.add_argument("--metrics", nargs='+', default=['accuracy'])
    args = parser.parse_args()

    from cesium.datasets.asas_training import fetch_asas_training
    from keras.utils.np_utils import to_categorical
    from keras.preprocessing.sequence import pad_sequences
    np.random.seed(0)
    data = fetch_asas_training()
    X_list = [np.c_[data['times'][i], data['measurements'][i],
                    data['errors'][i]] for i in range(len(data['times']))]
    n_max = max(len(t) for t in data['times'])
    X = pad_sequences(X_list, maxlen=n_max, value=-1., dtype='float')
    classnames, indices = np.unique(data['classes'], return_inverse=True)
    Y = to_categorical(indices, len(classnames))
    

    model_dict = {'gru': uneven_gru_classifier, 'lstm': uneven_lstm_classifier}

    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))
    model = model_dict[args.model_type](output_len=Y.shape[-1], n_step=n_max,
                                        **vars(args))

    run = "{}_{:03d}_x{}_{:1.0e}_drop{}".format(args.model_type, args.size,
                                                args.num_layers, args.lr,
                                                int(100 * args.drop_frac)).replace('e-', 'm')
    history = ku.train_and_log(X, Y, run, model, **vars(args))
