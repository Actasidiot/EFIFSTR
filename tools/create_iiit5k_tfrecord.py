import logging
import os
import random
import PIL.Image

import tensorflow as tf
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from aster.utils import dataset_util
from aster.core import standard_fields as fields

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw SynthText dataset.')
FLAGS = flags.FLAGS


def create_iiit5k_subset(output_path, train_subset=True, lexicon_index=None):
  writer = tf.python_io.TFRecordWriter(output_path)

  mat_file_name = 'traindata.mat' if train_subset else 'testdata.mat'
  data_key = 'traindata' if train_subset else 'testdata'
  groundtruth_mat_path = os.path.join(FLAGS.data_dir, mat_file_name)

  mat_dict = sio.loadmat(groundtruth_mat_path)
  entries = mat_dict[data_key].flatten()
  for entry in tqdm(entries):
    image_rel_path = str(entry[0][0])
    groundtruth_text = str(entry[1][0])
    if not train_subset:
      lexicon = [str(t[0]) for t in entry[lexicon_index].flatten()]

    image_path = os.path.join(FLAGS.data_dir, image_rel_path)
    with open(image_path, 'rb') as f:
      image_jpeg = f.read()

    example = tf.train.Example(features=tf.train.Features(feature={
      fields.TfExampleFields.image_encoded: \
        dataset_util.bytes_feature(image_jpeg),
      fields.TfExampleFields.image_format: \
        dataset_util.bytes_feature('jpeg'.encode('utf-8')),
      fields.TfExampleFields.filename: \
        dataset_util.bytes_feature(image_rel_path.encode('utf-8')),
      fields.TfExampleFields.channels: \
        dataset_util.int64_feature(3),
      fields.TfExampleFields.colorspace: \
        dataset_util.bytes_feature('rgb'.encode('utf-8')),
      fields.TfExampleFields.transcript: \
        dataset_util.bytes_feature(groundtruth_text.encode('utf-8')),
      fields.TfExampleFields.lexicon: \
        dataset_util.bytes_feature(('\t'.join(lexicon)).encode('utf-8'))
    }))
    writer.write(example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  # create_iiit5k_subset('data/iiit5k_train.tfrecord', train_subset=True)
  create_iiit5k_subset('data/iiit5k_test_50.tfrecord', train_subset=False, lexicon_index=2)
  # create_iiit5k_subset('data/iiit5k_test_1k.tfrecord', train_subset=False, lexicon_index=3)
