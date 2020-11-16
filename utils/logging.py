from utils import *
from sklearn import metrics

def _get_folder(_path):
    if os.name == 'nt':
        split = _path.split('\\')
    else:
        split = _path.split('/')
    filename = split[-1]
    folder = _path[:-len(filename)]
    return folder, filename

def _makefile(folder, path, header=None):
    os.makedirs(folder, exist_ok=True)
    with open(path, 'w') as f:
        if header is not None:
            f.write(header+'\n')

class LossLog:
    def __init__(self, path: str):
        self._path = path
        self._folder, self._filename = _get_folder(self._path)
        _makefile(self._folder, self._path, header =  'epoch\tloss\ttime\n')

    def append(self, epoch, loss, time):
        with open(self._path, 'a') as f:
            s = '{:d}\t{:3.6f}\t{:3.2f}\n'.format(epoch, loss, time)
            f.write(s)


class MultiParameterLog:
    def __init__(self, path: str, parameter_names: list = None):
        self._path = path
        self._folder, self._filename = _get_folder(self._path)
        self.n_params = len(parameter_names)
        self.params = parameter_names
        _makefile(self._folder, self._path, header = '\t'.join(parameter_names))

    def append(self, epoch, *args):
        with open(self._path, 'a') as f:
            s = '{:d}\t'.format(epoch)
            for a in args:
                s += '{:3.6f}\t'.format(a)
            s += '\n'
            f.write(s)

class MultiParameterLogJSON:
    def __init__(self, path: str = None):
        self._path = path
        if path is not None:
            self._folder, self._filename = _get_folder(self._path)
        self.log = []

    def append(self, save=True, **kwargs):
        self.log.append(kwargs)
        if save and self._path is not None:
            with open(self._path, 'w') as f:
                json.dump(self.log, f, indent=True)

class PredictionLog:
    def __init__(self, path: str, columns = None):
        self._path = path
        self._folder, self._filename = _get_folder(self._path)
        self.df = self._make_df(columns)

    def _make_df(self, columns_list=None):
        return pd.DataFrame(columns=columns_list)

    def append_old_v(self, epoch, ids, y_true, y_pred, y_pred_softmax):
        entry = self._make_df()
        entry['epoch'] = pd.Series([epoch] * len(y_true))
        entry['id'] = pd.Series(ids)
        entry['y_true'] = pd.Series(y_true)
        entry['y_pred'] = pd.Series(y_pred)
        self.df = self.df.append(entry, ignore_index=True)

    def append(self, **kwargs):
        entry = self._make_df(list(kwargs.keys()))
        for k, v in kwargs.items():
            entry[k] = pd.Series(v)
        self.df = self.df.append(entry, ignore_index = True, sort=False)

    def append_multi_task(self, epoch, img_id, task, y_true, y_pred):
        self.append(
            epoch = [epoch] * img_id.shape.as_list()[0],
            id = img_id.numpy().astype(str),
            task = [task] * img_id.shape.as_list()[0],
            y_true = y_true.numpy(),
            y_pred = y_pred.numpy()
        )
    def save(self):
        self.df.to_csv(self._path, index = False)

    def get_columns(self):
        return list(self.df.columns)

class ReportParams:
    def __init__(self, path, **kwargs):
        self._path = path
        self._kwargs = kwargs
        self._folder, self._filename = _get_folder(self._path)
        _makefile(self._folder, self._path)
        if kwargs != {}:
            self._make_report()

    def _make_report(self):
        max_pad = max([len(str(x)) for x in self._kwargs.keys()])
        s = ''
        for k, v in self._kwargs.items():
            s += '{}|{}\n'.format(str(k).ljust(max_pad), v)
        with open(self._path, 'w') as f:
            f.write(s)

    def append(self, **kwargs):
        # merge dicts
        self._kwargs = {**self._kwargs, **kwargs}
        # append new values
        self._make_report()


class ConfusionMatrix():
    def __init__(self, y_true, y_pred, labels=None, sample_weight=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
        self.sample_weight = sample_weight
        self.cm = metrics.confusion_matrix(y_true, y_pred, labels, sample_weight)

    def get_metrics(self, metrics_to_get: tuple =('recall', 'precision', 'accuracy', 'f1'), average='macro'):
        scores = {}
        if 'recall' in metrics_to_get:
            scores['recall'] = metrics.recall_score(
                self.y_true, self.y_pred, labels=self.labels, sample_weight=self.sample_weight, average=average
            )
        if 'precision' in metrics_to_get:
            scores['precision'] = metrics.precision_score(
                self.y_true, self.y_pred, labels=self.labels, sample_weight=self.sample_weight, average=average
            )
        if 'accuracy' in metrics_to_get:
            scores['accuracy'] = metrics.accuracy_score(
                self.y_true, self.y_pred, sample_weight=self.sample_weight
            )
        if 'f1' in metrics_to_get:
            scores['f1'] = metrics.f1_score(
                self.y_true, self.y_pred, labels=self.labels, sample_weight=self.sample_weight, average=average
            )
        return scores

    def __add__(self, other):
        return ConfusionMatrix(
            y_true= np.concatenate([self.y_true, other.y_true], axis=0),
            y_pred= np.concatenate([self.y_pred, other.y_pred], axis=0),
            labels=self.labels, sample_weight=self.sample_weight
        )

    def __iadd__(self, other):
        return self.__add__(other)

class AllMetricsLog():
    def __init__(
            self, path=None, labels=None, sample_weight=None,
            metrics_to_get: tuple =('recall', 'precision', 'accuracy', 'f1'), average='macro'
    ):
        self._path = path
        self._labels = labels
        self._sample_weight = sample_weight
        self._metrics = metrics_to_get
        self._average = average
        self.cm_log = {}
        self.metrics_log = {}

    def append(self, y_true, y_pred, epoch=None, add_to_existing = True):
        this_cm = ConfusionMatrix(y_true, y_pred, self._labels, self._sample_weight)
        log_index = epoch if epoch is not None else len(self.cm_log)
        if log_index in self.cm_log.keys() and add_to_existing:
            self.cm_log[log_index] += this_cm
        else:
            self.cm_log[log_index] = this_cm
        this_metrics = self.cm_log[log_index].get_metrics(self._metrics, self._average)
        self.metrics_log[log_index] = this_metrics
        return self.cm_log[log_index], self.metrics_log[log_index]

    def save_metrics(self):
        if self._path is not None:
            with open(self._path, 'w') as f:
                json.dump(self.metrics_log, f, indent=True)
        else:
            Warning('Could not save, path not specified.')

    def save_cms(self):
        # TODO: implement this, needs json encoding of numpy array
        pass