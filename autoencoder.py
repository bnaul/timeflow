import numpy as np
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector, Masking,
                          Recurrent, AtrousConv1D, Conv1D, Lambda, Bidirectional,
                          MaxPooling1D, UpSampling1D, SimpleRNN, BatchNormalization)
from custom_layers import PhasedLSTM
from keras.models import Model, Sequential

import sample_data
import keras_util as ku


# input: (t, m, e), (t, m), or (m)
def encoder(model_input, layer, size, num_layers, drop_frac=0.0, batch_norm=False,
            output_size=None, filter_length=None, pool=None, bidirectional=False,
            **parsed_args):
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

        # TODO apply this more elegantly? something like a decorator?
        if not bidirectional or not issubclass(layer, Recurrent):
            encode = layer(size, name='encode_{}'.format(i), **kwargs)(encode)
        else:
            encode = Bidirectional(layer(size, name='encode_{}'.format(i), **kwargs))(encode)

        if drop_frac > 0.0:
            encode = Dropout(drop_frac, name='drop_encode_{}'.format(i))(encode)
        if batch_norm:
            encode = BatchNormalization(name='bn_encode_{}'.format(i))(encode)
        if pool:
            encode = MaxPooling1D(pool, border_mode='same', name='pool_{}'.format(i))(encode)
        if i < num_layers - 1 and issubclass(layer, PhasedLSTM): # TODO experimental
                aux_input = Lambda(lambda a: a[:, :, 0:1],
                                   output_shape=lambda s: (s[0], s[1], 1))(model_input)
                encode = merge([aux_input, encode], mode='concat')

    if len(encode.get_shape()) > 2:
        encode = Flatten(name='flatten')(encode)
    encode = Dense(output_size, activation='linear', name='encoding')(encode)
    return encode


# aux input: (t) or (t, e) or None
# output: just m (output_len==1)
def decoder(encode, layer, n_step, size, num_layers, drop_frac=0.0, aux_input=None,
            batch_norm=False, filter_length=None, pool=None, bidirectional=False,
            **parsed_args):
    if issubclass(layer, Recurrent):
        decode = RepeatVector(n_step, name='repeat')(encode)
    else:
#        if pool:
#            n_step_init = n_step // (pool ** (num_layers - 1)) 
#        else:
        n_step_init = n_step
        decode = Dense(1 * n_step_init, activation='linear', name='dense_linear')(encode)
        decode = Reshape((n_step_init, 1), name='reshape_linear')(decode)

    if aux_input is not None:
        decode = merge([aux_input, decode], mode='concat')

    for i in range(num_layers):
        if i > 0:  # skip these for first layer (for symmetry)
            if batch_norm:
                decode = BatchNormalization(name='bn_decode_{}'.format(i))(decode)
            if drop_frac > 0.0:
                decode = Dropout(drop_frac, name='drop_decode_{}'.format(i))(decode)
            if pool:
#                decode = UpSampling1D(pool, name='upsample_{}'.format(i))(decode)
                pass

        kwargs = {}
        if issubclass(layer, Recurrent):
            kwargs['return_sequences'] = True
        if issubclass(layer, Conv1D):
            kwargs['activation'] = 'relu'  # TODO pass in
            kwargs['filter_length'] = filter_length
            kwargs['border_mode'] = 'same'
        if issubclass(layer, AtrousConv1D):
            kwargs['atrous_rate'] = 2 ** (i % 9)

        if not bidirectional or not issubclass(layer, Recurrent):
            decode = layer(size, name='decode_{}'.format(i), **kwargs)(decode)
        else:
            decode = Bidirectional(layer(size, name='decode_{}'.format(i), **kwargs))(decode)

#        if i < num_layers - 1:  # skip for last layer
#            if aux_input is not None and issubclass(layer, Recurrent):  # TODO experimental
#                decode = merge([aux_input, decode], mode='concat')

    if issubclass(layer, Recurrent):
        decode = TimeDistributed(Dense(1, activation='linear'), name='time_dist')(decode)
    else:
        decode = layer(1, activation='linear', filter_length=filter_length,
                       border_mode='same', name='conv_linear')(decode)
        decode = Reshape((n_step, 1), name='reshape_out')(decode)
    return decode


def main(args=None):
    np.random.seed(0)
    args = ku.parse_model_args(args)

    train = np.arange(args.N_train); test = np.arange(args.N_test) + args.N_train
    X, Y, X_raw, labels, = sample_data.periodic(args.N_train + args.N_test, args.n_min,
                                                args.n_max, even=args.even,
                                                noise_sigma=args.sigma,
                                                kind=args.data_type,
#                                                t_scale=0.05
                                               )

    if args.even:
        X = X[:, :, 1:2]
        X_raw = X_raw[:, :, 1:2]
    else:
        X[:, :, 0] = ku.times_to_lags(X_raw[:, :, 0])
        X[np.isnan(X)] = -1.
        X_raw[np.isnan(X_raw)] = -1.

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D, 'atrous': AtrousConv1D, 'phased': PhasedLSTM}

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
    decode = decoder(encode, num_layers=args.decode_layers if args.decode_layers
                                                           else args.num_layers,
                     layer=model_type_dict[args.decode_type if args.decode_type
                                           else args.model_type],
                     n_step=X.shape[1], aux_input=aux_input,
                     **{k: v for k, v in vars(args).items() if k != 'num_layers'})
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
