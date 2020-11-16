import os
import pandas as pd
import numpy as np
from datetime import datetime
from optparse import OptionParser
import re
import json


def get_path_to_dataset(filename):
    '''Get path to dataset based on OS.
    Location is hardcoded.'''
    if os.name == 'nt':
        # windows
        dataset_path = 'D:\\Data\\isic_2018\\{}'.format(filename)
    elif os.name == 'posix':
        # linux
        # dataset_path = '../isic_2018/{}'.format(filename) # workstation ntu
        dataset_path = '/data/isic_2018/{}'.format(filename)    # workstation bii
    else:
        raise Exception('Run on windows or linux.')
    return dataset_path


def get_path_to_image(id: str, type :str, return_folder: bool = False,
                      pattern: str = None, patch_id: str = None,
                      folder_name:str =None):
    ''' Get path to image.
    Types: rgb; mask; mask_attr; rgb_patch; mask_patch.
    If PATCH specify the patch id. '''
    if type == 'rgb':
        folder = 'images_all'
        suffix = ''
        extension = '.jpg'
    elif type == 'mask':
        folder = 'masks_segm'
        suffix = '_segmentation'
        extension = '.png'
    elif type == 'mask_attr':
        if pattern is not None:
            folder = 'masks_detect'
            suffix = '_'+pattern
            extension = '.png'
        else:
            raise Exception('Specify lesion pattern (pigment network, streaks, ...).')
    elif type == 'rgb_patch':
        if patch_id is not None:
            folder = 'rgb_patch'
            suffix = '_'+patch_id
            extension = '.jpg'
        else:
            raise Exception('Specify patch number (usually 00000, 00001, 00002, ...).')
    elif type == 'mask_patch':
        if patch_id is not None:
            folder = 'mask_patch'
            suffix = '_mask_'+patch_id
            extension = '.png'
        else:
            raise Exception('Specify patch number (usually 00000, 00001, 00002, ...).')
    else:
        raise NameError

    if folder_name is not None:
        folder = folder_name

    if os.name == 'nt':
        # windows
        path_prefix = 'D:\\Data\\isic_2018\\data\\{}\\'.format(folder)
    else:
        # linux
        # path_prefix = '../isic_2018/data/{}/'.format(folder) # workstation lab
        path_prefix = '/data/isic_2018/data/{}/'.format(folder) # workstation bii
    path = path_prefix + id + suffix + extension
    if not return_folder:
        return path
    else:
        return path, path_prefix


def make_run_name_as_timestamp():
    '''
    Make string for current run based on datetime using the format "{y}{m}{d}_{h}{mt}".
    :return: formatted string.
    '''
    now = datetime.now()
    run_name = '{:0>2}{:0>2}{:0>2}_{:0>2}{:0>2}'.format(
        str(now.year)[-2:], now.month, now.day, now.hour, now.minute
    )
    return run_name


def make_run_name(options_input):
    if not options_input:
        string = make_run_name_as_timestamp()
    else:
        regex = re.compile(r"^\d{6}_\d{4}")
        findings = regex.findall(options_input)
        if len(findings) > 0:
            # means that a timestamp is already present
            string = options_input
        else:
            # join timestamp and arg name
            string = '_'.join([make_run_name_as_timestamp(), options_input])
    return string

def usual_script_options(return_unparsed_object=False):
    parser = OptionParser()
    parser.add_option('-n', '--run_name', dest='run_name', type='string')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=10, type='int')
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int')
    parser.add_option('-d', '--dataset', dest='dataset')
    parser.add_option('--lr', dest='lr', default='0.001', type='float')
    parser.add_option('--comment', dest='comment', default='', type='string')
    parser.add_option('-t', '--tasks', dest='tasks', type='str')
    parser.add_option('--off', dest='mtl_off', action='store_true', default=False)
    parser.add_option('-g', '--gene', dest='genome', type='string', default=None)
    if return_unparsed_object:
        return parser
    else:
        return parse_script_options(parser)


def parse_script_options(parser):
    options, _ = parser.parse_args()
    BATCH_SIZE = options.batch_size
    EPOCHS = options.epochs
    RUN_NAME = make_run_name(options.run_name)
    LEARNING_RATE = options.lr

    if options.tasks:
        tasks_str = options.tasks
        TASKS = parse_tasks(tasks_str)
    else:
        TASKS = None

    return {
        'BATCH_SIZE': BATCH_SIZE, 'EPOCHS': EPOCHS,
        'RUN_NAME': RUN_NAME, 'LR': LEARNING_RATE,
        'DATASET': options.dataset if options.dataset else None,
        'COMMENT': options.comment, 'TASKS' : TASKS,
        'MTL_OFF' : options.mtl_off, 'GENOME' : options.genome
    }


def parse_tasks(tasks_str):
    if tasks_str != "all":
        tmp_tasks = tasks_str.split(';')
        tmp_tasks = [t.strip() for t in tmp_tasks]
        tasks_list = []
        for t in tmp_tasks:
            if '[' in t and ']' in t:
                tmp = t.split(',')
                tmp = [tt.strip('[').strip(']').strip() for tt in tmp]
            else:
                tmp = t.strip()
            tasks_list.append(tmp)
    else:
        tasks_list = tasks_str
    return tasks_list