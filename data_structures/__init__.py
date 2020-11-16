import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

def _control_len(f):
    def wrapper(self, *args, **kwargs):
        res = f(self, *args, **kwargs)
        ok = False
        while not ok:
            if self.__len__() > self.__max_len__:
                self.pop(0)
            else:
                ok = True
        return res
    return wrapper

class LimitedList(list):
    @_control_len
    def __init__(self, L : int, base_list: list = ()):
        super(LimitedList, self).__init__(base_list)
        self.__max_len__ = L

    @_control_len
    def append(self, p_object):
        super(LimitedList, self).append(p_object)

    def __add__(self, other):
        return self.__class__(self.__max_len__, base_list=super(LimitedList, self).__add__(other))

    def __iadd__(self, other):
        return self.__add__(other)

    def __getitem__(self, item):
        if type(item) is int:
            return super(LimitedList, self).__getitem__(item)
        else:
            return self.__class__(self.__max_len__, base_list=super(LimitedList, self).__getitem__(item))


class EarlyStoppingCheck(LimitedList):
    def __init__(
            self, monitor, patience, base_list=(), delta=1e-3,
            best_score = None, patience_counter = 0,
            mode='auto'
    ):
        super(EarlyStoppingCheck, self).__init__(patience, base_list=base_list)
        self.min_delta = None
        self.delta = delta
        self.best_score = best_score
        self.patience = patience
        self.patience_counter = patience_counter
        self.monitor=monitor
        self.mode=mode

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'loss' in self.monitor:
                self.monitor_op = np.less
            else:
                self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        for el in self: self._check_new_best(el)

    def _eval_mean_change(self):
        change = []
        for i in range(len(self) - 1):
            diff = self[i + 1] - self[i]
            change.append(diff)
        mc = sum(change) / len(change)
        return mc

    def _map_change_to_delta(self):
        mapping = []
        for i in range(len(self) - 1):
            diff = self[i + 1] - self[i]
            if diff < -self.delta:
               mapping.append(0)
            else:
                mapping.append(1)
        return mapping

    def change_is_ok(self):
        # eval mean
        if sum(self._map_change_to_delta()) < (len(self)-1):
            return True
        else:
            return False

    def _check_new_best(self, new_el):
        coeff = 1. if self.should_decrease else -1.
        if self.best_score is None:
            self.best_score = new_el
        elif coeff * (self.best_score - new_el) > self.delta:
            self.best_score = new_el    # set new best
            self.patience_counter = 0   # reset counter
        else:
            self.patience_counter += 1  # increase counter

    def is_done(self):
        if self.patience_counter > self.patience:
            return True
        else:
            return False

    def append(self, p_object):
        super(EarlyStoppingCheck, self).append(p_object)
        self._check_new_best(p_object)

    def __repr__(self):
        s = self.repr_patience()
        s += ' ['
        s += ','.join([str(x) for x in self])
        s += ']'
        return s

    def __add__(self, other):
        new_list = self.__class__(
            patience=self.patience,
            base_list=super(EarlyStoppingCheck, self).__add__(other),
            delta=self.delta,
            best_score=self.best_score, patience_counter=self.patience_counter
        )
        return new_list

    def is_full(self):
        if self.__len__() == self.__max_len__:
            return True
        else:
            return False

    def repr_patience(self):
        return '{}/{}'.format(self.patience_counter, self.patience)


class NewEarlyStopping(EarlyStopping):
    """
    Modified early stopping callback built on top of the existing early stopping.
    """
    def __init__(self, monitor='loss_val', min_delta=0, patience=0, mode='auto', verbose=0):
        super(NewEarlyStopping, self).__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=None,
            restore_best_weights=False
        )
        self.stop_training = False
        self.stopped_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.current_is_best(current):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True


    def current_is_best(self, current):
        return self.monitor_op(current - self.min_delta, self.best)

    def __repr__(self):
        return '{}/{}'.format(self.wait, self.patience)
