import numpy as np
from keras.preprocessing.sequence import pad_sequences


def random_uneven_times(N, n_min, n_max, a=2, scale=0.05):
    lags = [scale * np.random.pareto(a, size=np.random.randint(n_min, n_max + 1))
            for i in range(N)]
    return [np.concatenate(([0], np.cumsum(lags_i))) for lags_i in lags]


def periodic(N, n_min, n_max, t_max=None, even=True, A_shape=1., noise_sigma=1., w_min=0.01,
             w_max=1., t_shape=2, t_scale=0.05):
    """Returns sinuosid data (values, (frequency, amplitude, phase, offset))"""
    if even:
        if t_max is None:
            t_max = float(n_max)
        t = [np.linspace(0., t_max, n_max) for i in range(N)]
    else:
        t = random_uneven_times(N, n_min, n_max, t_shape, t_scale)
    w = np.random.uniform(w_min, w_max, size=N)
    A = np.random.gamma(shape=A_shape, scale=1. / A_shape, size=N)
    phi = 2 * np.pi * np.random.random(size=N)
    b = np.random.normal(scale=1, size=N)
    X_list = [np.c_[t[i], A[i] * np.sin(2 * np.pi * w[i] * t[i] + phi[i]) 
                    + np.random.normal(scale=noise_sigma, size=len(t[i])) + b[i]]
              for i in range(N)]
    X = pad_sequences(X_list, maxlen=n_max, value=0., dtype='float')
    Y = np.c_[w, A, phi, b]
    
    return X, Y 


def synthetic_control(N, n_min, n_max, t_max=None, even=True, sigma=2.):
    if t_max is None:
        t_max = float(n_max)

    base = lambda t: 30. + sigma * np.random.uniform(-3, 3, size=len(t))
    patterns = [base,
                lambda t: base(t) + (np.random.uniform(10, 15) *
                                     np.sin(2 * np.pi * t / np.random.uniform(10, 15))),
                lambda t: base(t) + np.random.uniform(0.2, 0.5) * t,
                lambda t: base(t) - np.random.uniform(0.2, 0.5) * t,
                lambda t: base(t) + ((t >= np.random.uniform(t_max / 3., 2. * t_max / 3.)) *
                                     np.random.uniform(7.5, 20)),
                lambda t: base(t) - ((t >= np.random.uniform(t_max / 3., 2. * t_max / 3.)) *
                                     np.random.uniform(7.5, 20))]
    y = np.random.randint(6, size=N)
     
    if even and n_min == n_max:
        t = np.linspace(0., t_max, n_max)
        X = np.asarray([patterns[y[i]](t) for i in range(N)]).reshape((-1, n_max, 1))
    else:
        t = random_uneven_times(N, n_min, n_max, t_max)
        X = pad_sequences([np.c_[t[i], patterns[y[i]](t[i])] for i in range(N)], maxlen=n_max,
                          value=0., dtype='float')
    return X, y
