import inspect
import sys
import time

import numpy as np
import peewee as pw
from playhouse.postgres_ext import ArrayField
from playhouse.shortcuts import model_to_dict
from playhouse import signals


db = pw.PostgresqlDatabase(autocommit=True, autorollback=True,
                           database='timeflow', host=None, port=5432,
                           user='timeflow', password=None)
#db = pw.SqliteDatabase('lc.db')


class BaseModel(signals.Model):
    def __str__(self):
        return to_json(self.__dict__())

    def __dict__(self):
        return model_to_dict(self, recurse=False, backrefs=False)

    class Meta:
        database = db


class LightCurve(BaseModel):
    """ORM model of the LightCurve table"""
    survey = pw.CharField(null=True)
    name = pw.CharField(unique=True)
    times = ArrayField(pw.DoubleField, index=False)
    measurements = ArrayField(pw.DoubleField, index=False)
    errors = ArrayField(pw.DoubleField, index=False)
    best_period = pw.DoubleField(null=True)
    best_score = pw.DoubleField(null=True)
    label = pw.CharField(null=True)

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



models = [
    obj for (name, obj) in inspect.getmembers(sys.modules[__name__])
    if inspect.isclass(obj) and issubclass(obj, pw.Model)
    and not obj == BaseModel
]


def create_tables(retry=5):
    for i in range(1, retry + 1):
        try:
            db.create_tables(models, safe=True)
            return
        except Exception as e:
            if (i == retry):
                raise e
            else:
                print('Could not connect to database...sleeping 5')
                time.sleep(5)

def drop_tables():
    db.drop_tables(models, safe=True)


def load_light_curves():
    import glob
    import os
    from io import StringIO
    import pandas as pd

    data_source = []
    for fname in glob.glob('./data/asas/*/*'):
        with open(fname) as f:
            dfs = [pd.read_csv(StringIO(chunk), comment='#', delim_whitespace=True) for chunk in f.read().split('#     ')[1:]]
            if len(dfs) > 0:
                df = pd.concat(dfs)[['HJD', 'MAG_0', 'MER_0', 'GRADE']].sort_values(by='HJD')
                df = df[df.GRADE <= 'B']
                df.drop_duplicates(subset=['HJD'], keep='first', inplace=True)

                data_source.append({'name': os.path.basename(fname), 'survey': 'ASAS', 'times': df.HJD.values, 
                                    'measurements': df.MAG_0.values, 'errors': df.MER_0.values,
    #                                'best_period': model_gat.best_period, 'best_score': model_gat.score(model_gat.best_period).item()
                                   })
                
    with db.atomic():
        for idx in range(0, len(data_source), 100):
            LightCurve.insert_many(data_source[idx:idx + 100]).execute()


def calculate_lomb_scargle():
    from gatspy.periodic import LombScargleFast

    for lc in LightCurve.select().where(LightCurve.best_score >> None):
        model_gat = LombScargleFast(fit_period=True, optimizer_kwds={'period_range': (0.005 * (max(lc.times) - min(lc.times)), 0.95 * (max(lc.times) - min(lc.times))), 'quiet': True}, silence_warnings=True)
        model_gat.fit(lc.times, lc.measurements, lc.errors)
        lc.best_period = model_gat.best_period
        lc.best_score = model_gat.score(model_gat.best_period).item()
        lc.save()


def store_class_labels():
# Update class information
    from db_models import LightCurve

    bigmacc = pd.read_csv('data/asas/bigmacc.txt', delimiter='\t')
    with db.atomic():
        for i, row in bigmacc.iterrows():
            try:
                lc = LightCurve.get(name=row.ASAS_ID)
                lc.label = row.CLASS
                lc.save()
            except LightCurve.DoesNotExist:
                pass

if __name__ == "__main__":
    print("Dropping tables...")
    drop_tables()
    print("Creating tables: {}".format([m.__name__ for m in models]))
    create_tables()
    print("Adding light curve data")
    load_light_curves()
    print("Calculating Lomb Scargle scores")
    calculate_lomb_scargle()
    print("Loading class information")
    store_class_labels()
    db.close()



"""
id -u postgres

if [[ $? == 0 ]]; then
    echo "Configuring Linux postgres"
    sudo -u postgres psql -c 'CREATE DATABASE timeflow;'
    sudo -u postgres psql -c 'CREATE USER timeflow;'
    sudo -u postgres psql -c 'GRANT ALL PRIVILEGES ON DATABASE timeflow to timeflow;'
else
    echo "Configuring OSX postgres"
    createdb -w timeflow
    createuser timeflow
    psql -U timeflow -c 'GRANT ALL PRIVILEGES ON DATABASE timeflow to timeflow;'
fi
"""
