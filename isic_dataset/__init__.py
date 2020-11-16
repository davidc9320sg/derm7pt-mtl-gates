import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

class ISIC2018():
    def __init__(
            self,
            root="/data/isic_2018/",
            csv_name="task3_gt.csv"
    ):
        #load dataset
        df = pd.read_csv(os.path.join(root, csv_name))
        tmp = pd.get_dummies(df, columns=['diagnosis'])
        #
        self.root = root
        self.csv_name = csv_name
        self.df = df
        self.tasks = None

    def get_dataset(
            self,
            tasks_in="all",
            test_frac=0.3,
            val_frac=0.1,
            batch_size=20,
            seed=0,
            random_jpg_quality = False,
            **kwargs
    ):
        # sample dataset tmp/test
        _tmp_df = test_df = train_df = val_df = None
        ssf = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        for _tmp_idx, test_idx in ssf.split(self.df['index'], self.df['diagnosis']):
            _tmp_df = self.df.iloc[_tmp_idx]
            test_df = self.df.iloc[test_idx]
        assert _tmp_df is not None and test_df is not None

        # from tmp, sample train/val split
        ssf = StratifiedShuffleSplit(n_splits=1, test_size=val_frac/(1-test_frac), random_state=seed)
        for train_idx, val_idx in ssf.split(_tmp_df['index'], _tmp_df['diagnosis']):
            train_df = _tmp_df.iloc[train_idx]
            val_df = _tmp_df.iloc[val_idx]
        assert val_df is not None and train_df is not None

        # create dummies
        train_df = pd.get_dummies(train_df, columns=['diagnosis'])
        test_df = pd.get_dummies(test_df, columns=['diagnosis'])
        val_df = pd.get_dummies(val_df, columns=['diagnosis'])

        self.train = train_df
        self.test = test_df
        self.val = val_df

        # define tasks
        if tasks_in == "all":
            tasks = [c for c in train_df.columns if 'diagnosis' in c]
        else:
            tasks = ['diagnosis_' + c for c in tasks_in]
        self.tasks = tasks

        # make idx list
        idx = []
        for df in [train_df, val_df, test_df]:
            idx.append(df['index'].values)
        train_idx, val_idx, test_idx = idx

        # create labels
        train_labels = []
        test_labels = []
        val_labels = []
        task_modes = []
        for t in tasks:
            select_mode = True
            for arr, df in zip(
                    [train_labels, val_labels, test_labels],
                    [train_df, val_df, test_df]
            ):
                # classification task
                y = df[t]
                y = pd.get_dummies(y)
                arr.append(y.values.astype(np.float32))
                if select_mode: task_modes.append('class')

        # TF Dataset
        train = tf.data.Dataset.from_tensor_slices((train_idx, *train_labels))
        val = tf.data.Dataset.from_tensor_slices((val_idx, *val_labels))
        test = tf.data.Dataset.from_tensor_slices((test_idx, *test_labels))

        # mapping functions definitions - ---
        root = self.root

        def _map_fn(im_id, *label):
            # read image
            im_folder = root + 'data/ham10k/images/'
            img_path = im_folder + im_id + '.jpg'
            im = _load_image(img_path)
            return im, tuple(label)

        def _preprocess(x, y):
            _im = x
            if random_jpg_quality:
                _im = tf.image.random_jpeg_quality(_im, 25, 100, seed=0)
            return _im, y

        # apply mapping functions
        train = train.map(_map_fn).map(_preprocess).batch(batch_size)
        val = val.map(_map_fn).batch(batch_size)
        test = test.map(_map_fn).batch(batch_size)

        return (train, val, test), task_modes

    def save_split(self, savedir):
        # save train_val_test_indeces
        k = pd.merge(
            self.train['index'].reset_index(drop=True).rename('train'),
            self.val['index'].reset_index(drop=True).rename('val'),
            how='outer', left_index=True, right_index=True
        )
        kk = pd.merge(
            k,
            self.test['index'].reset_index(drop=True).rename('test'),
            how='outer', left_index=True, right_index=True
        )
        kk.to_csv(os.path.join(savedir, 'image_id.csv'), index=False)

def _load_image(img_path):
    im = tf.io.read_file(img_path)
    im = tf.image.decode_jpeg(im, channels=3)
    im = tf.cast(im, dtype=tf.float32)
    im = im / 255.  # standard tf processing
    im = tf.image.resize(im, (224, 224))
    return im


# task 1-2
class ISIC2018_tasks_1_2():
    def __init__(self, random_state=111):
        self.root = './isic_18'
        self.jpg_root = self.root + '/images_all_res'
        self.segm_root = self.root + '/masks_segm_res'
        self.attr_root = self.root + '/masks_detect_res'
        self.csv_path = self.root + '/attributes_mtl.csv'
        self.batch_size = 1
        self.random_state = random_state
        # self.root = '/data/isic_2018/data'
        # self.jpg_root =  self.root + '/images_all'
        # self.segm_root = self.root + '/masks_segm'
        # self.attr_root = self.root + '/masks_detect'

    def get_dataset(self, batch_size = 1, classification_label='diagnosis', reshuffle_training=True):
        def lambda_f(x):
            if np.random.random() < 0.3 or x['diagnosis'] != 'nevus':
                return True
            return False

        def remove_sebker(x):
            if x['diagnosis'] == 'seborrheic keratosis':
                return False
            return True

        def apply_weight(x):
            return scores[x['diagnosis']]

        def change_to_categorical(x):
            if x == 'nevus':
                return 0
            if x == 'melanoma':
                return 1
            if x == 'seborrheic keratosis':
                return 2
            raise Exception()

        self.batch_size = batch_size
        self.classification_label = classification_label
        # read csv
        df_og = pd.read_csv(self.csv_path)
        idx_to_keep = df_og.apply(remove_sebker, axis=1)
        df = df_og[idx_to_keep]
        df = df.reset_index(drop=True)
        df['diagnosis'] = df['diagnosis'].apply(change_to_categorical)
        print(df.diagnosis.value_counts())
        scores = 1 - df.diagnosis.value_counts() / len(df)
        # scores = scores / sum(scores)
        df['weight'] = df.apply(apply_weight, axis=1)
        # separate train-val-test
        ssf_ma_te = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=self.random_state) # main-test
        ssf_tr_va = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=self.random_state) # train-validation
        # split all in main-test
        train_df = val_df = test_df = None
        for main_idx, test_idx in ssf_ma_te.split(df['id'], df[[classification_label, 'attribute_pigment_network']]):
            main_df = df.iloc[main_idx]
            test_df = df.iloc[test_idx]
            # split main in train-validation
            for train_idx, val_idx in ssf_tr_va.split(main_df['id'], main_df[[classification_label, 'attribute_pigment_network']]):
                train_df = main_df.iloc[train_idx]
                val_df = main_df.iloc[val_idx]
        # DATA AUGMENTATION by multiplying melanoma instances <---------
        melanoma = train_df[train_df['diagnosis'] == 1]
        concat = pd.concat([train_df, melanoma])
        print('augmenting train df')
        print(concat.diagnosis.value_counts() / len(concat))
        train_df = concat.sample(frac=1., random_state=self.random_state)
        # extract relevant features
        train_tuple = self.extract_features_and_labels(train_df)
        val_tuple = self.extract_features_and_labels(val_df)
        test_tuple = self.extract_features_and_labels(test_df)
        # print
        print('TRAIN'.center(15, '-'))
        print(train_df.diagnosis.value_counts())
        print('VAL'.center(15, '-'))
        print(val_df.diagnosis.value_counts())
        print('TEST'.center(15, '-'))
        print(test_df.diagnosis.value_counts())
        # make tf dataset
        train = self.make_tf_dataset(train_tuple, training=reshuffle_training)
        test = self.make_tf_dataset(test_tuple)
        val = self.make_tf_dataset(val_tuple)
        return train, test, val

    def extract_features_and_labels(self, complete_df):
        # make class labels
        class_labels = complete_df['diagnosis']
        print(class_labels.value_counts())
        print(complete_df['attribute_pigment_network'].value_counts()/len(complete_df))
        class_labels = class_labels.to_list()
        # generate path lists
        ids = complete_df['id']
        jpg_paths = self.jpg_root + '/' + complete_df['id'] + '.jpg'
        jpg_paths = jpg_paths.to_list()
        segm_paths = self.segm_root + '/' + complete_df['id'] + '_segmentation.png'
        segm_paths = segm_paths.to_list()
        # attributes list
        attribute_endings = [
            '_attribute_globules.png',
            '_attribute_milia_like_cyst.png',
            '_attribute_negative_network.png',
            '_attribute_pigment_network.png',
            '_attribute_streaks.png'
        ]
        attributes_paths = []
        for suffix in attribute_endings:
            tmp = self.attr_root + '/' + complete_df['id'] + suffix
            tmp = tmp.to_list()
            attributes_paths.append(tmp)
        # return
        ids_to_return = ids.to_list()
        weights = complete_df['weight'].to_list()
        return jpg_paths, class_labels, segm_paths, attributes_paths[3], weights

    def make_tf_dataset(self, ds_tuple, training=False):
        ds = tf.data.Dataset.from_tensor_slices(ds_tuple)
        # if training:
        #     ds = ds.shuffle(buffer_size=1000, seed=self.random_state, reshuffle_each_iteration=True)
        ds = ds.map(_map_features_and_labels)
        ds = ds.batch(self.batch_size)
        return ds

def _map_features_and_labels(jpg_path, label, segm_path, pigm_net_path, weights):
    input_img = _load_jpg(jpg_path)
    segm_label = _load_png(segm_path)
    segm_label = tf.reshape(segm_label, [-1])
    segm_label = tf.one_hot(segm_label, 2, axis=-1)
    pigm_net_label = _load_png(pigm_net_path)
    pigm_net_label = tf.reshape(pigm_net_label, [-1])
    pigm_net_label = tf.one_hot(pigm_net_label, 2, axis=-1)
    # TODO: revert back to depth = 3 for 'diagnosis'
    # TODO: revert back to depth = 2 for 'melanocytic' or 'nevus VS melanoma'
    label_one_hot = tf.one_hot(label, 2)
    return input_img, label_one_hot, segm_label, pigm_net_label, weights

def _preprocess_image(im):
    _im = tf.image.random_flip_left_right(_im)
    _im = tf.image.random_flip_up_down(_im)
    return _im


def _load_jpg(img_path):
    im = tf.io.read_file(img_path)
    im = tf.image.decode_jpeg(im, channels=3)
    im = tf.cast(im, dtype=tf.float32)
    # im = (im - 127.5) / 127.5  # standard tf processing
    im = im / 255.
    im = tf.image.resize(im, (512, 512))
    return im

def _load_png(img_path):
    im = tf.io.read_file(img_path)
    im = tf.image.decode_png(im, channels=0, dtype=tf.uint8)
    im = tf.image.resize(im, (309, 309))
    im = tf.cast(im, dtype=tf.int32)
    # im = im / 255.  # standard tf processing
    im = tf.clip_by_value(im, 0, 1)
    return im

def change_label(x):
    if x == 'nevus':
        return 0
    elif x == 'melanoma':
        return 1
    elif x == 'seborrheic keratosis':
        return 2
    else:
        raise Exception('unknown label encountered '+x)


def resize_and_save(directory):
    shapes = []
    listdir = os.listdir(directory)
    s = ''
    for i, filename in enumerate(listdir):
        if filename.endswith('.png'):
            filepath = directory + '/' + filename
            im = tf.io.read_file(filepath)
            im = tf.image.decode_png(im, channels=1)
            # im = tf.cast(im, dtype=tf.float32)
            # im = im / 255.
            im = tf.image.resize(im, (512, 512))
            im = tf.cast(im, dtype=tf.uint8)
            im = tf.image.encode_png(im)
            tf.io.write_file('./isic_resized_masks/'+filename, im)
            shapes.append(im.shape.as_list())
        # print
        print('\b'*len(s), end='')
        s = '{:03d}/{:03d}'.format(i, len(listdir))
        print(s, end='')
    print('')
    # return shapes