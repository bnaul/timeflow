import json
import os
import shutil
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ProgbarLogger, TensorBoard, EarlyStopping


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
                  metrics=[], sample_weight=None, no_train=False, **kwargs):
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
                     if key not in ['X', 'Y', 'model', 'optimizer', 'kwargs']}
        param_log.update(kwargs)
        json.dump(param_log, open(os.path.join(log_dir, 'param_log.json'), 'w'),
                  sort_keys=True, indent=2)
        history = model.fit(X, Y, nb_epoch=nb_epoch, batch_size=batch_size,
                            validation_split=0.2, callbacks=[ProgbarLogger(),
                                                             TensorBoard(log_dir=log_dir,
                                                                         write_graph=False),
                                                             EarlyStopping(patience=20)],
                            sample_weight=sample_weight)
        model.save_weights(os.path.join(log_dir, 'weights.h5'), overwrite=True)
    return history
