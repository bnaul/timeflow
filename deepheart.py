import numpy as np
from keras.preprocessing.sequence import pad_sequences
from DeepHeart.deepheart.parser import PCG

np.random.seed(0)
pcg_raw = PCG('data/deepheart/', doFFT=False)
pcg_raw.initialize_wav_data()
X_full = pcg_raw.X
Y = pcg_raw.y
y = (Y[:, 0] == 1).astype('int')
n_min = min(len(x) for x in X_full)
n_max = max(len(x) for x in X_full)

from keras.utils.np_utils import to_categorical

import keras_util as ku
from classification import even_gru_classifier as gru
from classification import even_conv_classifier as conv
#from classification import even_dense_classifier as dense

print("n_min: ".format(n_min))

gpu_frac = 0.48
gpu_id = 1
size = 64
num_layers = 3
drop_frac = 0.25
lr = 5e-4
sim_type = 'deepheart_gru{}'.format(n_min)

X_trunc = pad_sequences(X_full, maxlen=n_min, dtype='float32', value=-1., padding='post', truncating='post')
X = np.expand_dims(X_trunc, axis=2)
model = gru(output_len=Y.shape[1], n_step=n_min, size=size, num_layers=num_layers, drop_frac=drop_frac)
#sim_type = 'deepheart_dense'
#model = dense(output_len=Y.shape[1], n_step=n_min, size=size, num_layers=num_layers, drop_frac=drop_frac)
#sim_type = 'deepheart_conv'
#filter = 5
#model = conv(output_len=Y.shape[1], n_step=n_min, size=size, num_layers=num_layers, drop_frac=drop_frac, filter=filter)
run = "{}_{:03d}_x{}_{:1.0e}_drop{}".format(sim_type, size, num_layers, lr, int(100 * drop_frac)).replace('e-', 'm')
print(run)
history = ku.train_and_log(X, Y, run, model, nb_epoch=100, batch_size=50, lr=lr,
                           loss='categorical_crossentropy', sim_type=sim_type, metrics=['accuracy'])

"""
X = X_full

import dask
from cesium.featurize import featurize_time_series

features_to_use = ['amplitude',
                   'percent_beyond_1_std',
                   'maximum',
                   'max_slope',
                   'median',
                   'median_absolute_deviation',
                   'percent_close_to_median',
                   'minimum',
                   'skew',
                   'std',
                   'weighted_average'
                  ]
fset_cesium = featurize_time_series(times=None, values=X, features_to_use=features_to_use,
                                    targets=y, scheduler=dask.async.get_sync)

import numpy as np
import scipy.stats

def mean_signal(t, m, e):
    return np.mean(m)

def std_signal(t, m, e):
    return np.std(m)

def mean_square_signal(t, m, e):
    return np.mean(m ** 2)

def abs_diffs_signal(t, m, e):
    return np.sum(np.abs(np.diff(m)))

def skew_signal(t, m, e):
    return scipy.stats.skew(m)

guo_features = {
    'mean': mean_signal,
    'std': std_signal,
    'mean2': mean_square_signal,
    'abs_diffs': abs_diffs_signal,
    'skew': skew_signal
}
fset_guo = featurize_time_series(times=None, values=X,
                                 features_to_use=list(guo_features.keys()),
                                 targets=y, custom_functions=guo_features,
                                 scheduler=dask.async.get_sync)

import pywt

n_channels = 5
dwts = [pywt.wavedec(m, pywt.Wavelet('db1'), level=n_channels-1) for m in X]
fset_dwt = featurize_time_series(times=None, values=dwts,
                                 features_to_use=list(guo_features.keys()),
                                 targets=y, custom_functions=guo_features,
                                 scheduler=dask.async.get_sync)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

from cesium.build_model import build_model_from_featureset

train, test = train_test_split(np.arange(len(X)), random_state=0)

rfc_param_grid = {'n_estimators': [8, 16, 32, 64, 128, 256, 512, 1024]}
model_cesium = build_model_from_featureset(fset_cesium.isel(name=train),
                                           RandomForestClassifier(max_features='auto',
                                                                  random_state=0),
                                           params_to_optimize=rfc_param_grid)

knn_param_grid = {'n_neighbors': [1, 2, 3, 4]}
model_guo = build_model_from_featureset(fset_guo.isel(name=train),
                                        KNeighborsClassifier(),
                                        params_to_optimize=knn_param_grid)

model_dwt = build_model_from_featureset(fset_dwt.isel(name=train),
                                        KNeighborsClassifier(),
                                        params_to_optimize=knn_param_grid)


from sklearn.metrics import accuracy_score
from cesium.predict import model_predictions

preds_cesium = model_predictions(fset_cesium, model_cesium, return_probs=False)
preds_guo = model_predictions(fset_guo, model_guo, return_probs=False)
preds_dwt = model_predictions(fset_dwt, model_dwt, return_probs=False)

print("Built-in cesium features: training accuracy={:.2%}, test accuracy={:.2%}".format(
          accuracy_score(preds_cesium.prediction[train], y[train]),
          accuracy_score(preds_cesium.prediction[test], y[test])))

print("Guo et al. features: training accuracy={:.2%}, test accuracy={:.2%}".format(
          accuracy_score(preds_guo.prediction[train], y[train]),
          accuracy_score(preds_guo.prediction[test], y[test])))

print("Wavelet transform features: training accuracy={:.2%}, test accuracy={:.2%}".format(
          accuracy_score(preds_dwt.prediction[train], y[train]),
          accuracy_score(preds_dwt.prediction[test], y[test])))
"""
