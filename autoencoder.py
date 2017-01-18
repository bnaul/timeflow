import numpy as np
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector, Masking,
                          Recurrent, AtrousConv1D, Conv1D,
                          MaxPooling1D, SimpleRNN, BatchNormalization)
try:
    from keras.layers import PhasedLSTM
except:
    PhasedLSTM = None
    print("Skipping PhasedLSTM...")
from keras.models import Model, Sequential

import sample_data
import keras_util as ku


# input: (t, m, e), (t, m), or (m)
def encoder(model_input, layer, size, num_layers, drop_frac=0.0, batch_norm=False,
            output_size=None, filter_length=None, **parsed_args):
    if output_size is None:
        output_size = size

    encode = model_input
    for i in range(num_layers):
        kwargs = {}
        if issubclass(layer, Recurrent):
            kwargs['return_sequences'] = (i < num_layers - 1)
        if issubclass(layer, Conv1D):
            kwargs['activation'] = 'relu'  # TODO pass in
            kwargs['filter_length'] = filter_length
            kwargs['border_mode'] = 'same'
        if issubclass(layer, AtrousConv1D):
            kwargs['atrous_rate'] = 2 ** (i % 9)
            
        encode = layer(size, name='encode_{}'.format(i), **kwargs)(encode)
        if drop_frac > 0.0:
            encode = Dropout(drop_frac, name='drop_encode_{}'.format(i))(encode)
        if batch_norm:
            encode = BatchNormalization(mode=2, name='bn_encode_{}'.format(i))(encode)

    if len(encode.get_shape()) > 2:
        encode = Flatten(name='flatten')(encode)
    encode = Dense(output_size, activation='linear', name='encoding')(encode)
    return encode


# aux input: (t) or (t, e) or None
# output: just m (output_len==1)
def decoder(encode, layer, n_step, size, num_layers, drop_frac=0.0, aux_input=None,
            batch_norm=False, filter_length=None, **parsed_args):
    if drop_frac > 0.0:
        encode = Dropout(drop_frac, name='drop_decode')(encode)
    if issubclass(layer, Recurrent):
        decode = RepeatVector(n_step, name='repeat')(encode)
    else:
        decode = Dense(size * n_step, activation='linear', name='dense_linear')(encode)
        decode = Reshape((n_step, size), name='reshape_linear')(decode)
    if aux_input is not None:
        decode = merge([aux_input, decode], mode='concat')

    for i in range(num_layers):
        kwargs = {}
        if issubclass(layer, Recurrent):
            kwargs['return_sequences'] = True
        if issubclass(layer, Conv1D):
            kwargs['activation'] = 'relu'  # TODO pass in
            kwargs['filter_length'] = filter_length
            kwargs['border_mode'] = 'same'
        if issubclass(layer, AtrousConv1D):
            kwargs['atrous_rate'] = 2 ** (i % 9)

        decode = layer(size, name='decode_{}'.format(i), **kwargs)(decode)
        if batch_norm:
            decode = BatchNormalization(mode=2, name='bn_decode_{}'.format(i))(decode)

    if issubclass(layer, Recurrent):
        decode = TimeDistributed(Dense(1, activation='linear'), name='time_dist')(decode)
    else:
        decode = layer(1, activation='linear', filter_length=filter_length,
                       border_mode='same', name='conv_linear')(decode)
        decode = Reshape((n_step, 1), name='reshape_out')(decode)
    return decode


def main(args=None):
    if not args:
        args = ku.parse_model_args()

    np.random.seed(0)
    train = np.arange(args.N_train); test = np.arange(args.N_test) + args.N_train
    X, Y, X_raw, labels, = sample_data.periodic(args.N_train + args.N_test, args.n_min,
                                                args.n_max, t_max=2*np.pi, even=args.even,
                                                A_shape=5., noise_sigma=args.sigma,
                                                w_min=0.1, w_max=1., kind=args.data_type)

    if args.even:
        X = X[:, :, 1:2]
        X_raw = X_raw[:, :, 1:2]
    else:
        X[:, :, 0] = ku.times_to_lags(X_raw[:, :, 0])
        X[np.isnan(X)] = -1.
        X_raw[np.isnan(X_raw)] = -1.

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D, 'atrous': AtrousConv1D, 'phased': PhasedLSTM}
    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))

    main_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    if args.even:
        model_input = main_input
        aux_input = None
        if args.model_type == 'phased':
            raise NotImplementedError("Phased LSTM not implemented for --even")
    else:
        aux_input = Input(shape=(X.shape[1], X.shape[-1] - 1), name='aux_input')
        model_input = [main_input, aux_input]

    encode = encoder(main_input, layer=model_type_dict[args.model_type],
                     output_size=args.embedding, **vars(args))
    decode = decoder(encode, layer=model_type_dict[args.decode_type if args.decode_type
                                                   else args.model_type],
                     n_step=X.shape[1], aux_input=aux_input,
                     **vars(args))
    model = Model(model_input, decode)

    run = ku.get_run_id(**vars(args))
 
    if args.even:
        history = ku.train_and_log(X[train], X_raw[train], run, model, **vars(args))
    else:
        sample_weight = (X[train, :, -1] != -1)
        history = ku.train_and_log({'main_input': X[train], 'aux_input': X[train, :, 0:1]},
                                   X_raw[train, :, 1:2], run, model,
                                   sample_weight=sample_weight, **vars(args))
    return X, Y, X_raw, model, args


if __name__ == '__main__':
    X, Y, X_raw, model, args = main()
