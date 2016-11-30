import numpy as np
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, MaxPooling1D, SimpleRNN)
from keras.models import Model, Sequential

import sample_data
import keras_util as ku


# input: (t, m, e), (t, m), or (m)
def rnn_encoder(model_input, layer, size, num_layers, drop_frac=0.0,
                embedding_size=None):
    if embedding_size is None:
        embedding_size = size

    # TODO can we combine? think there might be an issue but can't remember why
    encode = layer(size if num_layers > 1 else embedding_size,
                        return_sequences=(num_layers > 1))(model_input)
    if drop_frac > 0.0:
        encode = Dropout(drop_frac)(encode)

    for i in range(1, num_layers):
        encode = layer(size if i != num_layers else embedding_size,
                       return_sequences=(i < num_layers - 1))(encode)
        if drop_frac > 0.0:
            encode = Dropout(drop_frac)(encode)

    return encode


# aux input: (t) or (t, e) or None
# output: just m (output_len==1)
def rnn_decoder(encode, layer, n_step, size, num_layers, drop_frac=0.0, aux_input=None):
    decode = RepeatVector(n_step)(encode)
    if aux_input is not None:
        decode = merge([aux_input, decode], mode='concat')
    for i in range(1, num_layers + 1):
        decode = layer(size, return_sequences=True)(decode)
        if drop_frac > 0.0:
            decode = Dropout(drop_frac)(decode)

#    # TODO try removing reLU?
#    out_relu = TimeDistributed(Dense(1, activation='relu'))(decode)
    out = TimeDistributed(Dense(1, activation='linear'))(decode)
    return out


if __name__ == '__main__':
    import sample_data
    args = ku.parse_model_args()

    np.random.seed(0)
    train = np.arange(args.N_train); test = np.arange(args.N_test) + args.N_train
    X, Y = sample_data.periodic(args.N_train + args.N_test, args.n_min,
                                args.n_max, t_max=2*np.pi, even=args.even,
                                A_shape=5., noise_sigma=args.sigma, w_min=0.1,
                                w_max=1.)

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN}
    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))

    main_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    if args.even:
        model_input = main_input
        aux_input = None
    else:
        aux_input = Input(shape=(X.shape[1], X.shape[-1] - 1), name='aux_input')
        model_input = [main_input, aux_input]

    encode = rnn_encoder(main_input, layer=model_type_dict[args.model_type],
                         size=args.size, num_layers=args.num_layers,
                         drop_frac=args.drop_frac,
                         embedding_size=args.embedding)
    decode = rnn_decoder(encode, layer=model_type_dict[args.model_type],
                         n_step=X.shape[1], size=args.size,
                         num_layers=args.num_layers, drop_frac=args.drop_frac,
                         aux_input=aux_input)
    model = Model(model_input, decode)

    run = ku.get_run_id(**vars(args))
 
    if args.even:
        # TODO restore 1d input?
#        history = ku.train_and_log(X[train], X[train], run, model, **vars(args))
        history = ku.train_and_log({'main_input': X[train], 'aux_input': X[train, :, 0:1]},
                                   X[train, :, 1:2], run, model,#sample_weight=sample_weight,
                                   **vars(args))
    else:
        # Replace times w/ lags
        X[:, :, 0] = np.c_[np.diff(X[:, :, 0]), np.zeros(X.shape[0])]

        sample_weight = (X[train, :, -1] != -1)
        history = ku.train_and_log({'main_input': X[train], 'aux_input': X[train, :, 0:1]},
                                   X[train, :, 1:2], run, model, sample_weight=sample_weight,
                                   **vars(args))
