import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import os
import pandas as pd
import pickle

def to_categorical(y):
    y_cat = tf.argmax(y, axis=-1)
    return y_cat


class MyCSVLogger():
    def __init__(self, path_to_csv):
        self.path_to_csv = path_to_csv
        self.df = None

    def on_epoch_end(self, logs):
        # add new data
        try:
            tmp = pd.DataFrame.from_records(logs)
        except:
            tmp = pd.DataFrame(pd.Series(logs)).transpose()
        if self.df is None:
            self.df = tmp
        else:
            self.df = pd.concat([self.df, tmp], axis=0, sort=False)
        self.df.reset_index(drop=True, inplace=True)
        # save
        self.df.to_csv(self.path_to_csv, index=False)


class AllMetrics():
    def __init__(self, normalize_cmat=True, name=''):
        self.name = name
        self.normalize_cmat = 'true' if normalize_cmat else None
        self.reset_states()

    def __call__(self, *args, **kwargs):
        self.update_state(*args, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None, **kwargs):
        # save y_true
        if self.y_true is None:
            self.y_true = y_true
        else:
            self.y_true = np.concatenate([self.y_true, y_true], axis=0)
        # save y_pred
        if self.y_pred is None:
            self.y_pred = y_pred
        else:
            self.y_pred = np.concatenate([self.y_pred, y_pred], axis=0)
        # number of observations
        self.batches_seen += 1
        self.seen += len(y_true)

    def update_scores(self):
        self.recall = recall_score(self.y_true, self.y_pred, average='macro')
        self.recall_per_class = recall_score(self.y_true, self.y_pred, average=None)
        self.precision = precision_score(self.y_true, self.y_pred, average='macro')
        self.precision_per_class = precision_score(self.y_true, self.y_pred, average=None)
        self.f1 = f1_score(self.y_true, self.y_pred, average='macro')
        self.f1_per_class = f1_score(self.y_true, self.y_pred, average=None)
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        # TODO : evaluate also specificity?
        self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred, normalize=self.normalize_cmat)


    def result(self, to_show=(0, 1, 2, 3)):
        self.update_scores()
        metrics_to_show = []
        if 0 in to_show: metrics_to_show.append(self.accuracy)
        if 1 in to_show: metrics_to_show.append(self.recall)
        if 2 in to_show: metrics_to_show.append(self.precision)
        if 3 in to_show: metrics_to_show.append(self.f1)
        return metrics_to_show

    def reset_states(self):
        self.recall = 0.
        self.precision = 0.
        self.f1 = 0.
        self.accuracy = 0.
        self.specificity = 0.
        self.confusion_matrix = None
        self.batches_seen = 0
        self.seen = 0
        self.y_true = None
        self.y_pred = None

    def __repr__(self):
        if self.y_true is not None and self.y_pred is not None:
            return '{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(*self.result())
        else:
            return 'Uninitialized {}'.format(self.name)

    def as_dict(self):
        sfx = '' if self.name == '' else '_'+self.name
        to_return = {
            'acc'+sfx : self.accuracy,
            'rec'+sfx : self.recall,
            'prec'+sfx : self.precision,
            'f1'+sfx : self.f1
        }
        return to_return

    def save(self, fname):
        with open(fname, 'wb') as file_h:
            pickle.dump(self, file_h)

    def load(self, fname):
        with open(fname, 'rb') as file_h:
            tmp = pickle.load(file_h)
        for key, value in tmp.__dict__.items():
            setattr(self, key, value)


class ConfusionMatrix():
    def __init__(self, y_true, y_pred, label_tag, epoch_loss=None, process_onehot=False, ):
        self.classes = label_tag
        # self.classes = self._parse_classes()
        self.max_class_char_len = max([len(i) for i in list(self._parse_classes(self.classes))])
        # if process_onehot:
        #     self.matrix = tf.confusion_matrix(
        #         self._process_onehot(labels), self._process_onehot(prediction),
        #         dtype=tf.float32
        #     )
        # else:
        self.matrix = confusion_matrix(y_true.numpy(), y_pred.numpy(), labels=self.classes)
        self.labels = y_true
        self.prediction = y_pred
        self.epoch_loss = epoch_loss

    def _process_onehot(self, y):
        return tf.argmax(y, axis=1)

    def _parse_classes(self, classes):
        _classes = []
        for c in classes:
            _classes.append(str(c))
        return _classes

    def accuracy(self):
        diag = np.diag(self.matrix)
        diag = np.sum(diag)
        n = np.sum(self.matrix)
        return diag / n

    def mean_class_recall(self, per_class=False):
        recall_per_class = []
        for i, cl in enumerate(self.classes):
            # count positive
            positive = np.sum(self.matrix[i])
            # if positive > 0 it means there are elements of that class in the batch
            if positive > 0:
                tp = self.matrix[i, i]
                recall = tp / positive
                recall_per_class.append(recall)
        if per_class:
            return (np.mean(recall_per_class), recall_per_class)
        else:
            return np.mean(recall_per_class)

    def mean_class_precision(self, per_class=False):
        precision_per_class = []
        for i, cl in enumerate(self.classes):
            # count positive
            positive = np.sum(self.matrix[i])
            # if positive > 0 it means there are elements of that class in the batch
            if positive > 0:
                tp = self.matrix[i, i]
                predicted = np.sum(self.matrix[:, i])
                if predicted != 0.:
                    precision = np.divide(tp, predicted)
                else:
                    precision = 0.
                precision_per_class.append(precision)

        if per_class:
            return (np.mean(precision_per_class), precision_per_class)
        else:
            return np.mean(precision_per_class)

    def mean_class_f1_score(self, per_class=False):
        mc_recall, pc_recall = self.mean_class_recall(per_class=True)
        mc_precision, pc_precision = self.mean_class_precision(per_class=True)
        pc_f1score = []
        for rec, prec in zip(pc_recall, pc_precision):
            if rec != 0. and prec !=0.:
                f1 = 2. * (rec * prec) / (rec + prec)
            else:
                f1 = 0.
            pc_f1score.append(f1)

        if per_class:
            return (np.mean(pc_f1score), pc_f1score)
        else:
            return np.mean(pc_f1score)

    def all_metrics(self):
        d =  {
            'accuracy': self.accuracy(),
            'recall': self.mean_class_recall(),
            'precision': self.mean_class_precision(),
            'f1': self.mean_class_f1_score(),
        }

        if self.epoch_loss:
            d.update({'loss': self.epoch_loss})

        return d

    def print_all_metrics(self, title=None, end='\n'):
        if title is not None:
            print(title, end='\t')
        for k, v in self.all_metrics().items():
            print(k, ':', '{:6.3f}'.format(v.numpy()), end='|\t')
        print(end=end, flush=True)

    def save_to_txt(self, path, report=False, epoch=None):
        if epoch is not None:
            header = '{:d}'.format(epoch).center(30, '_')
        else:
            header = '_' * 30

        with open(path, mode='a' if epoch > 0 else 'w') as f:
            # write header
            f.write(header)
            f.write('\n')
            # write confusion matrix
            for i, c in enumerate(list(self.classes)):
                s = ''
                s += c.ljust(self.max_class_char_len)
                s += '|\t'
                for j in range(len(list(self.classes))):
                    value = '{:5d}'.format(self.matrix[i, j].numpy())
                    s += value
                    s += '\t'
                s += '\n'
                f.write(s)
            f.write('\n')
            # write metrics report
            if report:
                for k, v in self.all_metrics().items():
                    f.write(k + ':' + '{:6.3f}'.format(v.numpy()) + '\n')
            f.write('\n')

    def save_metrics(self, path, epoch=0):
        with open(path, mode='a' if epoch > 0 else 'w') as f:
            keys_list = list(self.all_metrics().keys())
            if epoch == 0:
                # write header
                s = 'epoch\t'
                s += 'loss\t' if self.epoch_loss is not None else ''
                for k in keys_list:
                    end = '\t' if k != keys_list[-1] else '\n'
                    s += k + end
                f.write(s)
                # write values
            s = '{:d}\t'.format(epoch)
            s += '{:6.3f}\t'.format(self.epoch_loss) if self.epoch_loss is not None else ''
            for k, v in self.all_metrics().items():
                end = '\t' if k != keys_list[-1] else '\n'
                s += '{:6.3f}'.format(v.numpy()) + end
            f.write(s)


class CMHistory():
    def __init__(self, label_names):
        self.label_names = label_names
        self.cm = []

    def append(self, labels, prediction, epoch=None):
        new_cm = ConfusionMatrix(labels, prediction, label_tag=self.label_names)
        self.cm.append(
            {
                'epoch': epoch if epoch is not None else len(self.cm),
                'cm': new_cm
            }
        )


def IoU(X: tf.Tensor, Y: tf.Tensor):
    X1 = tf.reshape(X, [-1])
    Y1 = tf.reshape(Y, [-1])
    intersection = tf.reduce_sum(X1 * Y1)
    X1_area = tf.reduce_sum(X1)
    Y1_area = tf.reduce_sum(Y1)
    union = X1_area + Y1_area - intersection
    return intersection / union

class BatchPredictionLog(tf.keras.callbacks.Callback):
    def __init__(self, path, n_tasks):
        super(BatchPredictionLog, self).__init__()
        self.path = path
        self.n_tasks = n_tasks
        os.makedirs(path, exist_ok=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.batches = [None for _ in range(self.n_tasks)]
        self.seen = 0
        # self.predictions = [None for _ in range(self.n_tasks)]

    def on_batch_end(self, batch, y_true=None, y_pred=None, prefix=''):
        for i, (gt, pred) in enumerate(zip(y_true, y_pred)):
            new = np.concatenate((gt.numpy(), pred.numpy()), -1)
            if self.batches[i] is not None:
                self.batches[i] = np.concatenate((self.batches[i], new), 0)
            else:
                self.batches[i] = new
        self.seen += 1
        self.prefix = prefix
        return 0

    def on_epoch_end(self, epoch, logs=None):
        #TODO: save
        columns = []
        for t, batch in enumerate(self.batches):
            # for cc in ['gt', 'pred']:
            #     for c in range(batch.shape[-1] // 2):
            #         columns.append(
            #             'output_{}_{:02d}_{}'.format(t, c, cc)
            #         )
            new_path = os.path.join(self.path, '{:03d}'.format(epoch))
            os.makedirs(new_path, exist_ok=True)
            np.save(os.path.join(new_path, '{}output_{:02d}.npy'.format(self.prefix, t + 1)), batch)
        pass

class MeanBinaryIOUxImage():
    def __init__(self, name=''):
        self.name = name
        self.all_scores = None
        self.score = 0.

    def __call__(self, y_true, y_pred, *args, **kwargs):
        self.update_state(y_true, y_pred)

    def eval_iou(self, a, b):
        c = a * b  # intersection
        # print('and', c)
        d = np.clip(a + b, 0, 1)  # union
        # print('or', d)
        iou = np.count_nonzero(c) / np.count_nonzero(d)  # iou
        # print('iou', iou)
        return iou

    def update_state(self, y_true, y_pred):
        new_data = None
        for Y, Yp in zip(y_true, y_pred):
            this_score = self.eval_iou(Y, Yp)
            this_score = np.expand_dims(np.array(this_score), axis=0)
            if new_data is None:
                new_data = this_score
            else:
                new_data = np.concatenate([new_data, this_score], axis=0)
        # add to all scores
        if self.all_scores is None:
            self.all_scores = new_data
        else:
            self.all_scores = np.concatenate([self.all_scores, new_data], axis=0)
        # update scores
        self.update_scores()

    def update_scores(self):
        self.score = np.mean(self.all_scores)

    def result(self):
        self.update_scores()
        return self.score

    def reset_states(self):
        self.__init__(self.name)

    def __repr__(self):
        return str(self.score)