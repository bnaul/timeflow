import numpy as np
from keras.preprocessing.sequence import pad_sequences


def random_uneven_times(N, n_min, n_max, t_max):
    return [t_max * np.sort(np.random.random(size=np.random.randint(n_min, n_max + 1)))
            for i in range(N)]


def noisy_sin(_times, A, sigma, w):
    phi = 2 * np.pi * np.random.random()
    b = np.random.normal(scale=1)
    return (A * np.sin(2 * np.pi * w * _times + phi) 
            + np.random.normal(scale=sigma, size=len(_times)) + b)


def periodic(N, n_min, n_max, t_max=None, even=True, A=1., sigma=1., w_min=0.01, w_max=1.):
    if t_max is None:
        t_max = float(n_max)

    if even:
        t = [np.linspace(0., t_max, n_max) for i in range(N)]
    else:
        t = random_uneven_times(N, n_min, n_max, t_max)
    w = np.random.uniform(w_min, w_max, N)
    X = pad_sequences([np.c_[t[i], noisy_sin(t[i], A, sigma, w[i])] for i in range(N)],
                      maxlen=n_max, value=0., dtype='float')
    
    return X, w


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
