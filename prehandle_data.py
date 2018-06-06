# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import re
import os
from os import getcwd
from os.path import exists, isdir, isfile, join
import shutil
import numpy as np
import pandas as pd


class ImageString(object):
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'XR_(\w+)')

    def __init__(self, img_filename):
        self.img_filename = img_filename
        self.patient = self._parse_patient()
        self.study = self._parse_study()
        self.image_num = self._parse_image()
        self.study_type = self._parse_study_type()
        self.image = self._parse_image()
        self.normal = self._parse_normal()

    def flat_file_name(self):
        return "{}_{}_patient{}_study{}_image{}.png".format(self.normal, self.study_type, self.patient, self.study,
                                                            self.image, self.normal)

    def _parse_patient(self):
        return int(self._patient_re.search(self.img_filename).group(1))

    def _parse_study(self):
        return int(self._study_re.search(self.img_filename).group(1))

    def _parse_image(self):
        return int(self._image_re.search(self.img_filename).group(1))

    def _parse_study_type(self):
        return self._study_type_re.search(self.img_filename).group(1)

    def _parse_normal(self):
        return "normal" if ("negative" in self.img_filename) else "abnormal"


# processed
# data
# ├── train
# │   ├── abnormal
# │   └── normal
# └── val
#     ├── abnormal
#     └── normal

newpath = './data'
if not os.path.exists(newpath):
    os.makedirs(newpath)

newpath = './data/train'
if not os.path.exists(newpath):
    os.makedirs(newpath)

newpath = './data/val'
if not os.path.exists(newpath):
    os.makedirs(newpath)  

proc_data_dir = join(getcwd(), 'data')
proc_train_dir = join(proc_data_dir, 'train')
proc_val_dir = join(proc_data_dir, 'val')

train_csv = './MURA-v1.1/train_image_paths.csv'
val_csv = './MURA-v1.1/valid_image_paths.csv'
assert exists(train_csv) and isfile(train_csv) and exists(val_csv) and isfile(val_csv)

df = pd.read_csv(train_csv, names=['img_name'], header=None)
for img in df.img_name:
#     assert ("negative" in img) is (label is 0)
    enc = ImageString(img)
    cat_dir = join(proc_train_dir, enc.normal)
    if not os.path.exists(cat_dir):
        os.mkdir(cat_dir)
    shutil.copy2(enc.img_filename, join(cat_dir, enc.flat_file_name()))

df = pd.read_csv(val_csv, names=['img_name'], header=None)
for img in df.img_name:
#     assert ("negative" in img) is (label is 0)
    enc = ImageString(img)
    cat_dir = join(proc_val_dir, enc.normal)
    if not os.path.exists(cat_dir):
        os.mkdir(cat_dir)
    shutil.copy2(enc.img_filename, join(cat_dir, enc.flat_file_name()))
