import argparse
from functools import wraps
import json
import os
import shutil
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ProgbarLogger, TensorBoard, EarlyStopping


def parse_model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("size", nargs="?", type=int)
    parser.add_argument("num_layers", nargs="?", type=int)
    parser.add_argument("drop_frac", nargs="?", type=float)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--nb_epoch", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--loss", type=str, default='mse')
    parser.add_argument("--loss_weights", type=float, nargs='*')
    parser.add_argument("--model_type", type=str, default='lstm')
    parser.add_argument("--gpu_frac", type=float, default=0.31)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=2e-9)
    parser.add_argument("--sim_type", type=str, default='period')
    parser.add_argument("--filter_length", type=int, default=5)
    parser.add_argument("--N_train", type=int, default=50000)
    parser.add_argument("--N_test", type=int, default=1000)
    parser.add_argument("--n_min", type=int, default=100)
    parser.add_argument("--n_max", type=int, default=100)
    parser.add_argument('--even', dest='even', action='store_true')
    parser.add_argument('--uneven', dest='even', action='store_false')
    parser.add_argument('--no_train', dest='no_train', action='store_true')
    parser.add_argument('--embedding', type=int, default=None)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument('--batch_norm', dest='batch_norm', action='store_true')
    parser.set_defaults(even=True, batch_norm=False)
    args = parser.parse_args()
    return args


def get_run_id(model_type, size, num_layers, lr, drop_frac=0.0, filter_length=None,
               embedding=None, batch_norm=False, **kwargs):
    run = "{}_{:03d}_x{}_{:1.0e}_drop{}".format(model_type, size,
                                                num_layers, lr,
                                                int(100 * drop_frac)).replace('e-', 'm')
    if batch_norm:
        run += "_bn"
    if filter_length is not None:
        run += '_f{}'.format(filter_length)
    if embedding:
        run += '_emb{}'.format(embedding)

    return run


def limited_memory_session(gpu_frac, gpu_id):
    if gpu_frac <= 0.0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return tf.Session()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) if gpu_id is not None else ''
        gpu_opts = tf.ConfigProto(gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_frac))
        return tf.Session(config=gpu_opts)


def train_and_log(X, Y, run, model, nb_epoch, batch_size, lr, loss, sim_type,
                  metrics=[], sample_weight=None, no_train=False, patience=20,
                  **kwargs):
    optimizer = Adam(lr=lr)
    print(metrics)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                  sample_weight_mode='temporal' if sample_weight is not None else None)

    log_dir = os.path.join(os.getcwd(), 'keras_logs', sim_type, run)
    print(log_dir)
    if os.path.exists(os.path.join(log_dir, 'weights.h5')):
        print("Loading {}...".format(os.path.join(log_dir, 'weights.h5')))
        history = []
        model.load_weights(os.path.join(log_dir, 'weights.h5'))
    else:
        if no_train:
            raise FileNotFoundError("No weights found.")
        shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir)
        param_log = {key: value for key, value in locals().items()
                     if key not in ['X', 'Y', 'model', 'optimizer',
                                    'sample_weight', 'kwargs']}
        param_log.update(kwargs)
        json.dump(param_log, open(os.path.join(log_dir, 'param_log.json'), 'w'),
                  sort_keys=True, indent=2)
        history = model.fit(X, Y, nb_epoch=nb_epoch, batch_size=batch_size,
                            validation_split=0.2, callbacks=[ProgbarLogger(),
                                                             TensorBoard(log_dir=log_dir,
                                                                         write_graph=False),
                                                             EarlyStopping(patience=patience)],
                            sample_weight=sample_weight)
        model.save_weights(os.path.join(log_dir, 'weights.h5'), overwrite=True)
    return history
