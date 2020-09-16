import functools

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d
from tensorflow.contrib.framework import arg_scope

from utils import shape_utils


class FeatureExtractor(object):
  def __init__(self,
               convnet=None,
               brnn_fn_list=[],
               summarize_activations=False,
               is_training=True):
    self._convnet = convnet
    self._brnn_fn_list = brnn_fn_list
    self._summarize_activations = summarize_activations
    self._is_training = is_training

  def preprocess(self, resized_inputs, scope=None):
    with tf.variable_scope(scope, 'FeatureExtractorPreprocess', [resized_inputs]) as preproc_scope:
      preprocessed_inputs = self._convnet.preprocess(resized_inputs, preproc_scope)
    return preprocessed_inputs

  def extract_features(self, preprocessed_inputs, scope=None, reuse = None):
    with tf.variable_scope(scope, 'FeatureExtractor', [preprocessed_inputs], reuse = reuse):
      cnn_fmaps_mulscale = self._convnet.extract_features(preprocessed_inputs)
      cnn_fmaps_lastscale = cnn_fmaps_mulscale[-1]
      #### max pooling via vertical direction
      batch_size, feature_h, feature_w, feature_c = cnn_fmaps_lastscale.get_shape().as_list()
      cnn_fmaps_pool = tf.nn.max_pool(cnn_fmaps_lastscale,
                         ksize=[1, feature_h, 1, 1],
                         strides=[1, feature_h, 1, 1],
                         padding='SAME',
                         name='pool')
      cnn_fmaps_pool_trans = tf.transpose(cnn_fmaps_pool, [0, 2, 1, 3])  # [batch_size, feature_w, feature_h, FLAGS.out_channels]
      cnn_fmaps_pool_trans = tf.reshape(cnn_fmaps_pool_trans, [batch_size, feature_w, feature_c])
      #print('lstm encoder input shape: {}'.format(cnn_fmaps_pool_trans.get_shape().as_list()))

      ### a two layer LSTM
      cell = tf.nn.rnn_cell.LSTMCell(512, state_is_tuple=True)
      if self._is_training:
          cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)

      cell1 = tf.nn.rnn_cell.LSTMCell(512, state_is_tuple=True)
      if self._is_training:
          cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=0.5)

      stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)
      initial_state = stack.zero_state(batch_size, dtype=tf.float32)
      

      le_output, le_output_states = tf.nn.dynamic_rnn(
          cell=stack,
          inputs=cnn_fmaps_pool_trans,
          sequence_length=tf.fill([batch_size], feature_w),
          initial_state=initial_state,
          dtype=tf.float32,
          time_major=False
      )
    if self._summarize_activations:
      tf.summary.image('ResizedImage1', (preprocessed_inputs[:4] + 1.0) * 127.5, max_outputs=4)
      tf.summary.image('ResizedImage2', (preprocessed_inputs[-4:] + 1.0) * 127.5, max_outputs=4)
    
    return cnn_fmaps_mulscale, le_output_states
