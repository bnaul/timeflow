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
                           times=self.times[i], measurements=self.measurements[i],
                           errors=self.errors[i], best_period=self.best_period,
                           best_score=self.best_score, label=self.label)
                for i in inds]



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
    db.drop_tables(models, safe=True, cascade=True)


if __name__ == "__main__":
    print("Dropping tables...")
    drop_tables()
    print("Creating tables: {}".format([m.__name__ for m in models]))
    create_tables()


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
