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
import sample_data
import keras_util as ku
import cesium.datasets


def main(args=None):
    if not args:
        args = ku.parse_model_args()

    args.loss = 'categorical_crossentropy'

    np.random.seed(0)

    data = cesium.datasets.fetch_asas_training()
    top_classes = ['Classical_Cepheid', 'Semireg_PV', 'Mira', 'RR_Lyrae_FM',
                   'W_Ursae_Maj', 'LSP', 'RSG']
#    top_classes = np.unique(data['classes'])
    top_class_inds = np.in1d(data['classes'], top_classes)
    X_list = [np.c_[t, m, e] for t, m, e in zip(data['times'],
                                                data['measurements'],
                                                data['errors'])]
    X_list = [X_list[i] for i in np.where(top_class_inds)[0]]
    sub_array_list = [np.array_split(x, np.arange(args.n_max, len(x),
                                                  step=args.n_max)) for x in
                      X_list]
    sub_array_list = [[el for el in x if len(el) >= args.n_min] for x in
                      sub_array_list]
    X_list = [el for x in sub_array_list for el in x]

    classnames, indices = np.unique(data['classes'][top_class_inds],
                                    return_inverse=True)
    y = np.repeat(indices, [len(x) for x in sub_array_list])
    Y = to_categorical(y, len(top_classes))

    X_raw = pad_sequences(X_list, value=0., dtype='float', padding='post')
    #X, scale_params = preprocess(X_raw, args.m_max)
    X = X_raw.copy()

    # Remove errors
    X = X[:, :, :2]

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D, 'atrous': AtrousConv1D, 'phased': PhasedLSTM}

    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))

    model_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    encode = encoder(model_input, layer=model_type_dict[args.model_type], **vars(args))
    out = Dense(Y.shape[-1], activation='softmax')(encode)
    model = Model(model_input, out)

    run = ku.get_run_id(**vars(args))
 
    history = ku.train_and_log(X, Y, run, model, metrics=['accuracy'], **vars(args))
    return X, X_raw, Y, model, args


if __name__ == '__main__':
    X, X_raw, Y, model, args = main()
