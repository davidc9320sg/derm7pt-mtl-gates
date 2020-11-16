import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TARGET_SIZE = (256, 256)

class Derm7pt():
    def _extract_indeces(self, df_path):
        df = pd.read_csv(df_path)
        ans = df.values.flatten().astype(int)
        return ans

    def _extract_data_from_indeces(self, idx_list):
        return self.df_og.iloc[idx_list]

    def __init__(self, base_dir="/data/datasets/derm7pt-data", binary_diagnosis=False):
        self.target_size = TARGET_SIZE
        self.csv_path = base_dir + '/derm_data.csv'
        self.images_root = base_dir + '/images'
        # load df
        self.df_og = pd.read_csv(self.csv_path)
        # for task diagnosis numeric, change the labels in binary "melanoma" v "non melanoma"
        self.binary_diagnosis = binary_diagnosis
        if binary_diagnosis:
            self.df_og['diagnosis_numeric'] = self.df_og['diagnosis_numeric'].apply(lambda x: 1 if x==2 else 0)
        self.unique_classes = 24 if not binary_diagnosis else 21
        # extract indeces
        self.train_idx_dir = base_dir + '/meta/train_indexes.csv'
        self.val_idx_dir = base_dir + '/meta/valid_indexes.csv'
        self.test_idx_dir = base_dir + '/meta/test_indexes.csv'
        self.train_idx = self._extract_indeces(self.train_idx_dir)
        self.val_idx = self._extract_indeces(self.val_idx_dir)
        self.test_idx = self._extract_indeces(self.test_idx_dir)
        # extract data
        self.train = self._extract_data_from_indeces(self.train_idx)
        self.val = self._extract_data_from_indeces(self.val_idx)
        self.test = self._extract_data_from_indeces(self.test_idx)
        # tasks
        self.tasks  = [
            'diagnosis_numeric',
            'pigment_network_numeric',
            'blue_whitish_veil_numeric',
            'vascular_structures_numeric',
            'pigmentation_numeric',
            'streaks_numeric',
            'dots_and_globules_numeric',
            'regression_structures_numeric'
        ]
        # define one-hot encoders
        encoders = {}
        for t in self.tasks:
            tmp = self.df_og[t].values.reshape((-1, 1))
            ohe = OneHotEncoder(sparse=False, dtype=np.float32).fit(tmp)
            encoders[t] = ohe
            # DEBUG
            # tmp_x = self.df_og[t].iloc[:5].values.reshape((-1, 1))
            # ans = ohe.transform(tmp_x)
            # print(ans)
        self.one_hot_encoders = encoders
        # criteria scores
        self.criteria_scores = {
            'pigment_network_numeric' : tf.constant([0., 0., 2.], dtype=tf.float32),
            'blue_whitish_veil_numeric' : tf.constant([0., 2.], dtype=tf.float32),
            'vascular_structures_numeric' : tf.constant([0., 0., 2.], dtype=tf.float32),
            'pigmentation_numeric' : tf.constant([0., 0., 1.], dtype=tf.float32),
            'streaks_numeric' : tf.constant([0., 0., 1.], dtype=tf.float32),
            'dots_and_globules_numeric' : tf.constant([0., 0., 1.], dtype=tf.float32),
            'regression_structures_numeric': tf.constant([0, 1.], dtype=tf.float32)
        }
        self.criteria_scores = [self.criteria_scores[t] for t in self.tasks if t is not 'diagnosis_numeric']
        self.tasks_output_shapes = {
            'diagnosis_numeric' : 5 if not self.binary_diagnosis else 2,
            'pigment_network_numeric' : 3,
            'blue_whitish_veil_numeric' : 2,
            'vascular_structures_numeric' : 3,
            'pigmentation_numeric' : 3,
            'streaks_numeric' : 3,
            'dots_and_globules_numeric' : 3,
            'regression_structures_numeric' : 2
        }

    def get_dataset(
            self, train_frac = 0.4, test_frac = 0.4, batch_size = 1, classification_label='diagnosis', reshuffle_training=True,
            random_state = 0, augment_train = True, random_rotations = True
    ):
        """
        Get dataset in TF dataset format. Data is randomly split in training, testing, validation.
        """
        self.batch_size = batch_size
        self.classification_label = classification_label

        # read csv

        # separate train-val-test
        focus_col = 'diagnosis_numeric'

        # augmentation
        if augment_train:
            self.train_og = self.train.copy(deep=True)
            diag_labels_replication_factors = [5, 0, 0, 3, 5]
            for idx, replication_fact in enumerate(diag_labels_replication_factors):
                if replication_fact > 0:
                    to_replicate = self.train[self.train['diagnosis_numeric'] == idx]
                    concat = pd.concat([self.train, *[to_replicate]*replication_fact])
                    # print('augmenting train df')
                    # print(concat.diagnosis.value_counts() / len(concat))
                    self.train = concat.sample(frac=1., random_state=random_state)
            print('AUG TRAIN')
            print_split(self.train, self.tasks)

        # make tf dataset - train
        train_ds = self.make_tf_dataset(
            self.train, batch_size=batch_size, shuffle=True, reshuffle_each_iteration=reshuffle_training
        )
        if random_rotations:
            train_ds = train_ds.map(map_augment)
        # make tf dataset - val
        val_ds = self.make_tf_dataset(self.val, batch_size=batch_size, shuffle=False)
        # make tf dataset - test
        test_ds = self.make_tf_dataset(self.test, batch_size=batch_size, shuffle=False)

        return train_ds, val_ds, test_ds

    def get_balanced_dataset(self, k, random_state=0, random_rotations=False, reshuffle_training=False, max_iter_train=20):
        """
        Get datasets for training, validation and test.
        The training dataset is balanced according to the sampling method of Kawahara et al.
        """
        train_and_val = pd.concat([self.train, self.val])
        batch_size = self.unique_classes * k
        # training ---------------------------
        train_ds = Derm7ptBalancedIterator(
            data=self.train, tasks=self.tasks, images_root=self.images_root, k=k,
            random_state=random_state, random_rotations=random_rotations,
            reshuffle_each_iteration=reshuffle_training, max_iter_train=max_iter_train,
            encoders=self.one_hot_encoders
        )
        # validation ---------------------
        val_ds = Derm7ptNormalIterator(
            data=self.val, tasks=self.tasks, images_root=self.images_root, batch_size=batch_size,
            encoders=self.one_hot_encoders, augment=False, name='validation_iterator'
        )
        # test --------------------------
        test_ds = Derm7ptNormalIterator(
            data=self.test, tasks=self.tasks, images_root=self.images_root, batch_size=batch_size,
            encoders=self.one_hot_encoders, augment=False, name='validation_iterator'
        )
        return  train_ds, val_ds, test_ds

    def make_tf_dataset(self, df, batch_size=None, shuffle=False, seed=0, reshuffle_each_iteration=False):
        # labels
        image_path = self.images_root + '/' + df['derm']
        image_path = image_path
        labels = []
        for t in self.tasks:
            tmp = df[t]
            tmp = pd.get_dummies(tmp)
            tmp = tmp.values
            tmp = tmp.astype(np.float32)
            labels.append(tmp)

        tf_ds = tf.data.Dataset.from_tensor_slices((image_path, *labels))
        if shuffle:
            tf_ds = tf_ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
        if batch_size is not None:
            tf_ds = tf_ds.batch(batch_size)
        tf_ds = tf_ds.map(map_input_and_labels)
        return tf_ds


def map_input_and_labels(*input_tuple):
    x_path = input_tuple[0]
    im = tf.map_fn(load_jpg, x_path, dtype=tf.float32)
    return im, input_tuple[1:]

def map_augment(*input_tuple):
    im = input_tuple[0]
    im = tf.map_fn(augment, im, dtype=tf.float32)
    return im, input_tuple[1]

def augment(im):
    _im = tf.keras.preprocessing.image.random_rotation(im, 360, channel_axis=-1)
    _im = tf.keras.preprocessing.image.random_zoom(_im, (0.5, 2.), channel_axis=-1)
    _im = tf.keras.preprocessing.image.random_shift(_im, 0.3, 0.3, channel_axis=-1)
    _im = tf.image.random_flip_up_down(im)
    _im = tf.image.random_flip_left_right(_im)
    return _im

def load_jpg(img_path, resize=True, resize_shape=(512, 512)):
    im = tf.io.read_file(img_path)
    im = tf.image.decode_jpeg(im, channels=3)
    im = tf.cast(im, dtype=tf.float32)
    # im = (im - 127.5) / 127.5  # standard tf processing
    im = im / 255.
    if resize:
        im = tf.image.resize(im, resize_shape)
    return im

class Derm7ptNormalIterator:
    def __init__(
            self,
            data: pd.DataFrame,
            tasks: list,
            images_root: str,
            batch_size : int,
            encoders : dict,
            reshuffle=False,
            augment=False,
            name=''
    ):
        self.data = data
        self.tasks = tasks
        self.images_root = images_root
        self.batch_size = batch_size
        self.index = 0
        self.encoders = encoders
        self.reshuffle = reshuffle
        if self.reshuffle:
            self.data = self.data.sample(frac=1.)
        self.augment = augment
        self.datagen = ImageDataGenerator(
            rotation_range=45, width_shift_range=0.25, height_shift_range=0.25,
            zoom_range=0.25, horizontal_flip=True, vertical_flip=True,
            dtype=np.float32, data_format='channels_last'
        )
        self.name = name

    def __iter__(self):
        return self

    def __next__(self):
        i0 = self.index
        i1 = min(self.index + self.batch_size, len(self.data))
        if i0 == i1:
            self.index = 0
            if self.reshuffle:
                self.data = self.data.sample(frac=1.)
            raise StopIteration
        data_to_return = self.data.iloc[i0:i1]
        x_and_y = self.load_image_and_features(data_to_return)
        self.index = i1
        return x_and_y


    def load_image_and_features(self, df):
        # image
        image_path = self.images_root + '/' + df['derm']
        image_path = image_path.values
        # load images
        with Pool(20) as p:
            X = np.array(p.map(load_image_fn, image_path))
        # apply transformations
        if self.augment:
            for X in self.datagen.flow(
                    X, batch_size=len(X), shuffle=False,
                    # save_to_dir='/data/projects/isic-cvpr-mtl-derm7pt/aug/',
            ):
                break
        # labels
        labels = []
        for t in self.tasks:
            tmp = df[t].values.reshape((-1, 1))
            tmp = self.encoders[t].transform(tmp)
            labels.append(tmp)
        return X, labels


class Derm7ptBalancedIterator():
    """
    Dataset iterator taking samples according to the sampling method from Kawahara et al.
    """
    def __init__(
            self,
            data : pd.DataFrame,
            tasks : list,
            images_root : str,
            encoders : dict,
            k=1,
            random_state=0,
            random_rotations=True,
            reshuffle_each_iteration=False,
            max_iter_train=20
    ):
        self.data = data
        self.tasks = tasks
        self.encoders = encoders
        self.index = 0
        self.k = k
        self.initial_random_state = random_state
        self.random_state = random_state
        self.images_root = images_root
        self.random_rotations = random_rotations
        self.reshuffle_each_iteration = reshuffle_each_iteration
        # compute
        self.batch_size = 24 * k
        self.DEFAULT_MAX_ITER =  max_iter_train # max(max_iter_train, len(self.data) / self.batch_size)
        self.max_iter = self.DEFAULT_MAX_ITER
        #
        self.iterated_once = []
        # transformer; i.e. keras image generator
        rotation_range = 360 if random_rotations else 0
        self.datagen = ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.10, height_shift_range=0.10,
            zoom_range=0.10, horizontal_flip=True, vertical_flip=True,
            shear_range=0.1, #brightness_range=(0.5, 1.), rescale=1/255.,
            dtype=np.float32, data_format='channels_last'
        )

    def take(self, i):
        if i > 0:
            self.max_iter = i
        elif i == 0 or i < -1:
            raise Exception('i must be > 0 or -1')
        return self.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        self.current_seed = self.random_state + self.index
        if len(self.iterated_once) >= len(self.data) or self.index > self.max_iter:
            print('self.index', self.index)
            # reset
            # print(self.iterated_once) # DEBUG
            # with open('/home/coppolad/Desktop/TEMP/iter/tmp{}.txt'.format('sd'), 'a') as f:
            #     f.write(str(self.iterated_once)+'\n')
            self.iterated_once = []
            if self.reshuffle_each_iteration:
                self.random_state = self.index + self.random_state
            else:
                self.random_state = self.initial_random_state
            self.index = 0
            self.max_iter = self.DEFAULT_MAX_ITER
            raise StopIteration
        data_to_return = None
        for t in self.tasks:
            unique_labels = self.data[t].unique()
            data_for_this_task = None
            for ul in unique_labels:
                data_for_ul = self.data[self.data[t] == ul]
                data_for_ul = data_for_ul.sample(n=self.k, random_state=self.current_seed)
                # add to data for this task
                if data_for_this_task is None:
                    data_for_this_task = data_for_ul.copy()
                else:
                    data_for_this_task = data_for_this_task.append(data_for_ul, sort=False, ignore_index=True)
            # add to sampled data
            if data_to_return is None:
                data_to_return = data_for_this_task.copy()
            else:
                data_to_return = data_to_return.append(data_for_this_task, sort=False, ignore_index=True)
        #
        data_to_return = data_to_return.sample(frac=1., random_state=self.current_seed)
        self.batch_size = len(data_to_return)
        self.add_to_iterated(values = data_to_return['case_num'].values)
        # print(data_to_return['case_num'].values)  # DEBUG <<<<<<--------------------
        # extract features
        x_and_y = self.load_image_and_features(data_to_return)
        self.index += 1
        return x_and_y

    def load_image_and_features(self, df):
        # image
        image_path = self.images_root + '/' + df['derm']
        image_path = image_path.values
        # load images
        with Pool(15) as p:
            all_im = np.array(p.map(load_image_fn, image_path))
        # apply transformations
        X = None
        for X in self.datagen.flow(
                all_im, seed=self.current_seed, batch_size=len(all_im), shuffle=False,
                ):
            break

        # labels
        labels = []
        for t in self.tasks:
            tmp = df[t].values.reshape((-1, 1))
            tmp = self.encoders[t].transform(tmp)
            labels.append(tmp)
        return X, labels

    def add_to_iterated(self, values):
        for v in values:
            if v not in self.iterated_once:
                self.iterated_once.append(v)

def load_image_fn(im_pth):
    im = tf.keras.preprocessing.image.load_img(im_pth, target_size=TARGET_SIZE, interpolation='bicubic')
    im = np.array(im, dtype=np.float32)
    im = im / 255.
    return im

def load_n_aug(im_pth, random_rotations=True):
    im = tf.keras.preprocessing.image.load_img(im_pth, target_size=(512, 512))
    im = np.array(im, dtype=np.float32) / 255.
    if random_rotations:
        im = tf.keras.preprocessing.image.random_zoom(im, (0.5, 1.5), row_axis=0, col_axis=1, channel_axis=2)
        # plt.imshow(im), plt.show()
        im = tf.keras.preprocessing.image.random_rotation(im, 360, row_axis=0, col_axis=1, channel_axis=2)
        # plt.imshow(im), plt.show()
        im = tf.keras.preprocessing.image.random_shift(im, 0.25, 0.25, row_axis=0, col_axis=1, channel_axis=2)
        # plt.imshow(im), plt.show()
    # im = tf.constant(im, dtype=tf.float32)
    # im = np.expand_dims(im, 0)
    return im


def print_split(_df, tasks):
    s = ''
    for t in tasks:
        s += '{}\t'.format(t)
        vc = _df[t].value_counts().sort_index()
        # vc
        for k, v in vc.items():
            s += '({}, {})\t'.format(k, v)
        s += '\n'
    print(s)