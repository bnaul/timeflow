import glob
import os
from io import StringIO

import numpy as np
import pandas as pd
import joblib
from gatspy.periodic import LombScargleFast


class LightCurve():
    def __init__(self, times, measurements, errors, survey=None, name=None,
                 best_period=None, best_score=None, label=None):
        self.times = times
        self.measurements = measurements
        self.errors = errors
        self.survey = survey
        self.name = name
        self.best_period = best_period
        self.best_score = best_score
        self.label = label

    def __repr__(self):
        return "LightCurve(" + ', '.join("{}={}".format(k, v)
                                         for k, v in self.__dict__.items()) + ")"

    def __len__(self):
        return len(self.times)

    def split(self, n_min=0, n_max=np.inf):
        inds = np.arange(len(self.times))
        splits = [x for x in np.array_split(inds, np.arange(n_max, len(inds), step=n_max))
                  if len(x) >= n_min]
        return [LightCurve(survey=self.survey, name=self.name,
                           times=[self.times[i] for i in s],
                           measurements=[self.measurements[i] for i in s],
                           errors=[self.errors[i] for i in s], best_period=self.best_period,
                           best_score=self.best_score, label=self.label)
                for s in splits]

    def load_all():
        light_curves = []
        bigmacc = pd.read_csv('data/asas/bigmacc.txt', delimiter='\t', index_col='ASAS_ID')
        for fname in glob.glob('./data/asas/*/*'):
            with open(fname) as f:
                dfs = [pd.read_csv(StringIO(chunk), comment='#', delim_whitespace=True) for chunk in f.read().split('#     ')[1:]]
                if len(dfs) > 0:
                    df = pd.concat(dfs)[['HJD', 'MAG_0', 'MER_0', 'GRADE']].sort_values(by='HJD')
                    df = df[df.GRADE <= 'B']
                    df.drop_duplicates(subset=['HJD'], keep='first', inplace=True)
                    lc = LightCurve(name=os.path.basename(fname), survey='ASAS',
                                    times=df.HJD.values, measurements=df.MAG_0.values,
                                    errors=df.MER_0.values)
                    lc.label = bigmacc.loc[lc.name].CLASS if lc.name in bigmacc.index else None
                    period_range = (0.005 * (max(lc.times) - min(lc.times)),
                                    0.95 * (max(lc.times) - min(lc.times)))
                    model_gat = LombScargleFast(fit_period=True, silence_warnings=True,
                        optimizer_kwds={'period_range': period_range, 'quiet': True})
                    model_gat.fit(lc.times, lc.measurements, lc.errors)
                    lc.best_period = model_gat.best_period
                    lc.best_score = model_gat.score(model_gat.best_period).item()
                    light_curves.append(lc)
        return light_curves


if __name__ == "__main__":
    print("Adding light curve data")
    light_curves = LightCurve.load_all()
    joblib.dump(light_curves, 'light_curves.pkl', compress=3)
