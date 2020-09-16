# -*- coding: utf-8 -*-

import os
import io
import random
import re
import glob

from PIL import Image
import tensorflow as tf

from aster.utils import dataset_util
from aster.core import standard_fields as fields

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/data/wangyz/Projects/Dataset/', 'Root directory to raw SynthText dataset.')
flags.DEFINE_bool('exclude_difficult', False, 'Excluding non-alphanumeric examples.')
flags.DEFINE_string('output_path', './ic1315coco_train_all.tfrecord', 'Output tfrecord path.')
FLAGS = flags.FLAGS

def _is_difficult(word):
  assert isinstance(word, str)
  return not re.match('^[\w]+$', word)

def char_check(word):
  if not word.isalnum():
    return False
  else:
    for char in word:
      if char < ' ' or char > '~':
        return False
  return True

def create_ic15(output_path):
  writer = tf.python_io.TFRecordWriter(output_path)
  data_dirs = ['/data/wangyz/Projects/Dataset/IC13_WordRecognition/Challenge2_Training_Task3_Images_GT/',
  '/data/wangyz/Projects/Dataset/IC15Inc_WordRecognition/ch4_training_word_images_gt/',
  '/data/wangyz/Projects/Dataset/COCO_WordRecognition/train_words/',
  '/data/wangyz/Projects/Dataset/COCO_WordRecognition/val_words/']
  groundtruth_file_pathes = ['/data/wangyz/Projects/Dataset/IC13_WordRecognition/Challenge2_Training_Task3_Images_GT/gt.txt',
  '/data/wangyz/Projects/Dataset/IC15Inc_WordRecognition/ch4_training_word_images_gt/gt.txt',
  '/data/wangyz/Projects/Dataset/COCO_WordRecognition/train_words_gt.txt',
  '/data/wangyz/Projects/Dataset/COCO_WordRecognition/val_words_gt.txt',
  ]
  dataset_names = ['IC13' ,'IC15' , 'COCO-train','COCO-val']
  count = 0
  for datasetid in range(4):
    #datasetid = 0
    groundtruth_file_path = groundtruth_file_pathes[datasetid]

    with open(groundtruth_file_path, 'r') as f:
      lines = f.readlines()
      img_gts = [line.strip() for line in lines]
      for img_gt in img_gts:
        if datasetid == 0:
          content=img_gt.split(',')
          img_path=data_dirs[datasetid] + content[0]
          gt = content[1][2:-1]
        if datasetid == 1:
          content=img_gt.split(',')
          img_path=data_dirs[datasetid] + content[0]
          gt = content[1][2:-1]
        if datasetid == 2:
          content=img_gt.split(',')
          img_path=data_dirs[datasetid]+content[0]+'.jpg'
          ll=len(content[0])
          gt = img_gt[ll+1:]
        if datasetid == 3:
          content=img_gt.split(',')
          img_path=data_dirs[datasetid]+content[0]+'.jpg'
          ll=len(content[0])
          gt = img_gt[ll+1:]
        if FLAGS.exclude_difficult and not char_check(gt):
          continue
        #img_path = os.path.join(FLAGS.data_dir, img_rel_path)


        img = Image.open(img_path)
        img=img.convert('RGB')
        img_buff = io.BytesIO()
        img.save(img_buff, format='jpeg')
        word_crop_jpeg = img_buff.getvalue()
        crop_name = dataset_names[datasetid] + '_' + os.path.basename(img_path)
        gt = gt.replace(' ', '')
        #print(img_gt)
        #print(crop_name)
        #print(gt)
        #input()
        example = tf.train.Example(features=tf.train.Features(feature={
          fields.TfExampleFields.image_encoded: \
            dataset_util.bytes_feature(word_crop_jpeg),
          fields.TfExampleFields.image_format: \
            dataset_util.bytes_feature('jpeg'.encode('utf-8')),
          fields.TfExampleFields.filename: \
            dataset_util.bytes_feature(crop_name.encode('utf-8')),
          fields.TfExampleFields.channels: \
            dataset_util.int64_feature(3),
          fields.TfExampleFields.colorspace: \
            dataset_util.bytes_feature('rgb'.encode('utf-8')),
          fields.TfExampleFields.transcript: \
            dataset_util.bytes_feature(gt.encode('utf-8')),
        }))
        writer.write(example.SerializeToString())
        count += 1
        if count % 1000 == 0:
          print(count)
  writer.close()
  print('{} examples created'.format(count))

if __name__ == '__main__':
  create_ic15(FLAGS.output_path)
