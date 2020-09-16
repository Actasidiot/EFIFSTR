import logging
import functools

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.contrib.layers import conv2d, max_pool2d, fully_connected, batch_norm, conv2d_transpose
from core import predictor
from core import sync_attention_wrapper
from core import loss
from utils import shape_utils
from c_ops import ops
import numpy as np
class AttentionPredictor(predictor.Predictor):
  """Attention decoder based on tf.contrib.seq2seq"""

  def __init__(self,
               rnn_cell=None,
               rnn_regularizer=None,
               num_attention_units=None,
               max_num_steps=None,
               multi_attention=False,
               beam_width=None,
               reverse=False,
               label_map=None,
               loss=None,
               sync=False,
               lm_rnn_cell=None,
               is_training=True):
    super(AttentionPredictor, self).__init__(is_training)
    self._rnn_cell = rnn_cell
    self._rnn_regularizer = rnn_regularizer
    self._num_attention_units = num_attention_units
    self._max_num_steps = max_num_steps
    self._multi_attention = multi_attention
    self._beam_width = beam_width
    self._reverse = reverse
    self._label_map = label_map
    self._sync = sync
    self._lm_rnn_cell = lm_rnn_cell
    self._loss = loss
    self._embeddings = tf.get_variable("Embedding", [325, 1, 1, 128], tf.float32, tf.random_normal_initializer(stddev=0.01), trainable = True)


    if not self._is_training and not self._beam_width > 0:
      raise ValueError('Beam width must be > 0 during inference')

  @property
  def start_label(self):
    return 0

  @property
  def end_label(self):
    return 1

  @property
  def num_classes(self):
    return self._label_map.num_classes + 2


  def compute_att_2d(self, ld_output, cnn_fmaps_mulscale, d):
    with tf.variable_scope('decoder/compute2d_attention_layer'):
      cnn_fmaps_s1 = cnn_fmaps_mulscale[0] # 24 * 80
      cnn_fmaps_s2 = cnn_fmaps_mulscale[1] # 12 * 40
      cnn_fmaps_s3 = cnn_fmaps_mulscale[2] # 6 * 40
      cnn_fmaps_lastscale = cnn_fmaps_mulscale[-1] # 6 * 40
      batch_size, batch_len, depth = shape_utils.combined_static_and_dynamic_shape(ld_output)
      _, feature_h, feature_w, feature_c = cnn_fmaps_lastscale.get_shape().as_list()
      ## for lstm outputs
      ld_output = tf.reshape(ld_output, (batch_size * batch_len, depth))
      ld_output_reshape = tf.reshape(ld_output, [batch_size * batch_len, 1, 1, depth])
      ld_output_conv = conv2d(ld_output_reshape, d, 1, activation_fn = None ,  normalizer_fn=None, scope ='hs_conv')
      ld_output_conv_tile = tf.tile(ld_output_conv,[1,feature_h,feature_w,1])

      cnn_fmap_conv = conv2d(cnn_fmaps_lastscale, d, 3, activation_fn = None , normalizer_fn=None, scope='fmap_conv')
      cnn_fmap_tile = tf.expand_dims(cnn_fmap_conv, 1)
      cnn_fmap_tile = tf.tile(cnn_fmap_tile,[1,batch_len,1,1,1])
      cnn_fmap_tile = tf.reshape(cnn_fmap_tile, [batch_size * batch_len, feature_h, feature_w, feature_c])

      g = tf.nn.tanh(tf.add(cnn_fmap_tile, ld_output_conv_tile))
      g = tf.nn.dropout(g, 0.5)
      g_conv = conv2d(g, 1, 1, scope='g_conv',  activation_fn = None ,  normalizer_fn=None)
      g_conv_reshape = tf.reshape(g_conv, [batch_size * batch_len, feature_w * feature_h])
      g_conv_reshape_softmax = tf.nn.softmax(g_conv_reshape)
      mask = tf.reshape(g_conv_reshape_softmax, [batch_size * batch_len, feature_h , feature_w, 1])
      tf.summary.image('Mask1', (mask[:20]) , max_outputs=20)

      g_tmp = tf.tile(tf.reshape(g_conv_reshape_softmax, [batch_size * batch_len, feature_h , feature_w, 1]),[1,1,1,feature_c])
      glimpse = tf.reduce_sum(tf.multiply(cnn_fmap_tile, g_tmp), [1,2])

      _, cnn_fmap_s1_h, cnn_fmap_s1_w, cnn_fmap_s1_c = cnn_fmaps_s1.get_shape().as_list()
      _, cnn_fmap_s2_h, cnn_fmap_s2_w, cnn_fmap_s2_c = cnn_fmaps_s2.get_shape().as_list()
      _, cnn_fmap_s3_h, cnn_fmap_s3_w, cnn_fmap_s3_c = cnn_fmaps_s3.get_shape().as_list()
      
      mask_s3 = tf.tile(mask,[1,1,1,cnn_fmap_s3_c]) #bs_bl, 6 ,40 ,1
      mask_s2 = tf.tile(tf.image.resize_bilinear(mask,[cnn_fmap_s2_h,cnn_fmap_s2_w]),[1,1,1,cnn_fmap_s2_c])
      mask_s1 = tf.tile(tf.image.resize_bilinear(mask,[cnn_fmap_s1_h,cnn_fmap_s1_w]),[1,1,1,cnn_fmap_s1_c])
      # cnn_fmaps_s1 ( bs * 24 * 80 * c )
      
      cnn_fmap_s1_tile = tf.expand_dims(cnn_fmaps_s1, 1)
      cnn_fmap_s1_tile = tf.tile(cnn_fmap_s1_tile,[1,batch_len,1,1,1])
      cnn_fmap_s1_tile = tf.reshape(cnn_fmap_s1_tile, [batch_size * batch_len, cnn_fmap_s1_h, cnn_fmap_s1_w, cnn_fmap_s1_c])
      glimpse_s1 = tf.multiply(cnn_fmap_s1_tile, mask_s1)
      
      cnn_fmap_s2_tile = tf.expand_dims(cnn_fmaps_s2, 1)
      cnn_fmap_s2_tile = tf.tile(cnn_fmap_s2_tile,[1,batch_len,1,1,1])
      cnn_fmap_s2_tile = tf.reshape(cnn_fmap_s2_tile, [batch_size * batch_len, cnn_fmap_s2_h, cnn_fmap_s2_w, cnn_fmap_s2_c])
      glimpse_s2 = tf.multiply(cnn_fmap_s2_tile, mask_s2)
      
      cnn_fmap_s3_tile = tf.expand_dims(cnn_fmaps_s3, 1)
      cnn_fmap_s3_tile = tf.tile(cnn_fmap_s3_tile,[1,batch_len,1,1,1])
      cnn_fmap_s3_tile = tf.reshape(cnn_fmap_s3_tile, [batch_size * batch_len, cnn_fmap_s3_h, cnn_fmap_s3_w, cnn_fmap_s3_c])
      glimpse_s3 = tf.multiply(cnn_fmap_s3_tile, mask_s3)
      
      
      glimpse_s1_reshape = tf.reshape(glimpse_s1, [batch_size * batch_len, cnn_fmap_s1_h * cnn_fmap_s1_w, cnn_fmap_s1_c])
      glimpse_s1_reshape = tf.reshape(glimpse_s1_reshape, [batch_size * batch_len, cnn_fmap_s1_c, cnn_fmap_s1_h * cnn_fmap_s1_w])
      
      glimpse_s2_reshape = tf.reshape(glimpse_s2, [batch_size * batch_len, cnn_fmap_s2_h * cnn_fmap_s2_w, cnn_fmap_s2_c])
      glimpse_s2_reshape = tf.reshape(glimpse_s2_reshape, [batch_size * batch_len,  cnn_fmap_s2_c, cnn_fmap_s2_h * cnn_fmap_s2_w])
      
      glimpse_s3_reshape = tf.reshape(glimpse_s3, [batch_size * batch_len, cnn_fmap_s3_h * cnn_fmap_s3_w, cnn_fmap_s3_c])
      glimpse_s3_reshape = tf.reshape(glimpse_s3_reshape, [batch_size * batch_len,  cnn_fmap_s3_c, cnn_fmap_s3_h * cnn_fmap_s3_w])
      
      glimpse_s1_resize_ = fully_connected(glimpse_s1_reshape, 16 * 16)
      glimpse_s2_resize_ = fully_connected(glimpse_s2_reshape, 8 * 8)
      glimpse_s3_resize_ = fully_connected(glimpse_s3_reshape, 4 * 4)

      glimpse_s1_resize = tf.reshape(glimpse_s1_resize_, [batch_size * batch_len, 16 * 16, cnn_fmap_s1_c])
      glimpse_s1_resize = tf.reshape(glimpse_s1_resize, [batch_size * batch_len, 16, 16, cnn_fmap_s1_c])
      
      glimpse_s2_resize = tf.reshape(glimpse_s2_resize_, [batch_size * batch_len, 8 * 8, cnn_fmap_s2_c])
      glimpse_s2_resize = tf.reshape(glimpse_s2_resize, [batch_size * batch_len,  8, 8, cnn_fmap_s2_c])
      
      glimpse_s3_resize = tf.reshape(glimpse_s3_resize_, [batch_size * batch_len, 4 * 4, cnn_fmap_s3_c])
      glimpse_s3_resize = tf.reshape(glimpse_s3_resize, [batch_size * batch_len,  4, 4, cnn_fmap_s3_c])


      embeddings_ids = tf.random_uniform([batch_size * batch_len], minval = 0, maxval = 325, dtype= tf.int64)
      embeddings_fordeconv = tf.gather(self._embeddings, embeddings_ids)

      glimpse_fordeconv = tf.reshape(glimpse, [batch_size * batch_len, 1, 1, depth])
      concat_feat = tf.concat([glimpse_fordeconv, embeddings_fordeconv],axis=-1)
      d1 = conv2d_transpose(concat_feat, 128, [2,  2], [2, 2], normalizer_fn = batch_norm, scope='gly_deconv_1')
      d2 = conv2d_transpose(d1, 64, [3, 3],  [2, 2], normalizer_fn = batch_norm, scope='gly_deconv_2')
      d3 = conv2d_transpose(tf.concat([d2,glimpse_s3_resize],axis=-1), 32, [3, 3],  [2, 2], normalizer_fn = batch_norm, scope='gly_deconv_3')
      d4 = conv2d_transpose(tf.concat([d3,glimpse_s2_resize],axis=-1), 16, [3, 3],  [2, 2], normalizer_fn = batch_norm, scope='gly_deconv_4')
      d5 = conv2d_transpose(tf.concat([d4,glimpse_s1_resize],axis=-1), 1, [3, 3],  [2, 2], activation_fn = tf.nn.tanh, scope='gly_deconv_5')

      glyph = d5 # batch_size * batchlen , 32, 32, 1
      glyph = tf.reshape(glyph,[batch_size * batch_len, 32 * 32] ) # batch_size * batchlen , 32 * 32
      glyph_for_visual = tf.reshape((glyph + 1.0) * 127.5, [batch_size * batch_len, 32 , 32, 1])
      tf.summary.image('glyph1', (glyph_for_visual[:20]) , max_outputs=20)
      
      glyph_output = tf.reshape(glyph,[batch_size, batch_len, 32 * 32] ) # batch_size , batchlen , 32 * 32

      glimpse = tf.reshape(glimpse,[batch_size,batch_len, depth ] )
      c_h_concat = tf.concat([glimpse,tf.reshape(ld_output,[batch_size, batch_len, depth])],axis=-1)
      rnn_output = tf.layers.dense(c_h_concat, self.num_classes, name = 'output_w')

      
    return rnn_output, glyph_output, embeddings_ids
  

  def predict(self, cnn_fmaps_mulscale, lstm_holistic_features, scope=None):
    with tf.variable_scope(scope, 'Predict'):
      predict = []
      ### a two layer LSTM
      cell0 = tf.nn.rnn_cell.LSTMCell(512, state_is_tuple=True)

      if self._is_training:
          cell0 = tf.nn.rnn_cell.DropoutWrapper(cell=cell0, output_keep_prob=0.5)
      cell1 = tf.nn.rnn_cell.LSTMCell(512, state_is_tuple=True)
      if self._is_training:
          cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=0.5)
      lstm_cell = tf.nn.rnn_cell.MultiRNNCell([cell0, cell1], state_is_tuple=True)
      char_embedding_array = tf.constant(np.identity(self.num_classes, dtype=np.float32))
      with tf.variable_scope('decoder') as scope:
        ld_output, ld_output_states = tf.nn.dynamic_rnn(
            cell=lstm_cell,
            inputs=tf.nn.embedding_lookup(char_embedding_array, self._groundtruth_dict['decoder_inputs']),
            #sequence_length=tf.fill([batch_size], feature_w),
            initial_state=lstm_holistic_features,
            dtype=tf.float32,
            time_major=False,
            scope = scope
        )
        
      
      batch_size, batch_len, depth = shape_utils.combined_static_and_dynamic_shape(ld_output)
      rnn_output, glyphs, embeddings_ids = self.compute_att_2d(ld_output, cnn_fmaps_mulscale, 512)

      sample_id = tf.argmax(rnn_output, 2)
      if self._is_training:
        #assert isinstance(outputs, seq2seq.BasicDecoderOutput)
        outputs_dict = {
          'labels': sample_id,
          'logits': rnn_output,
          'glyphs': glyphs,
          'embedding_ids': embeddings_ids
        }
      else:
        outputs_dict = {
          'labels': sample_id,
          'scores': res_score,
          #'lengths': prediction_lengths
        }
    return outputs_dict

  def RecognitionLoss(self, predictions_dict, scope=None):
    assert 'logits' or 'glyphs' in predictions_dict
    with tf.variable_scope(scope, 'Loss', list(predictions_dict.values())):
      loss_tensor = self._loss(
        predictions_dict['logits'],
        self._groundtruth_dict['decoder_targets'],
        self._groundtruth_dict['decoder_lengths']
      )
    return loss_tensor
  def GenerationLoss(self, predictions_dict, scope=None):
    assert 'logits' or 'glyphs' in predictions_dict
    with tf.variable_scope(scope, 'Loss', list(predictions_dict.values())):
      glyphs = predictions_dict['glyphs']
      ref_glyphs = tf.constant(np.load('data/glyphs-325-fonts.npy'),dtype=tf.float32) # 96 , 325, 32*32
      #ref_glyphs_reshape = tf.reshape(ref_glyphs, [96*325, 32*32])
      labels = self._groundtruth_dict['decoder_targets']
      lengths = self._groundtruth_dict['decoder_lengths']
      batch_size, batch_len = shape_utils.combined_static_and_dynamic_shape(labels)
      labels_indexes = tf.reshape(labels, [batch_size * batch_len]) +  96 * predictions_dict['embedding_ids']

      targets = tf.gather(ref_glyphs, labels_indexes) # batch_size * batch_len, 8, 32*32
      targets_for_visual = tf.reshape((targets + 1.0) * 127.5, [batch_size * batch_len, 32 , 32, 1])
      tf.summary.image('target_glyph1', (targets_for_visual[:20]) , max_outputs=20)
      targets = tf.reshape(targets, [batch_size, batch_len, 32*32])

      with tf.name_scope(scope, 'WeightedL1Loss'):
        raw_losses = tf.reduce_mean(tf.abs(glyphs - targets), axis=[2])
        batch_size, max_time = shape_utils.combined_static_and_dynamic_shape(labels)
        mask = tf.less(
          tf.tile([tf.range(max_time)], [batch_size, 1]),
          tf.expand_dims(lengths, 1),
          name='mask'
        )
        masked_losses = tf.multiply(
          raw_losses,
          tf.cast(mask, tf.float32),
          name='masked_losses'
        )
        row_losses = tf.reduce_sum(masked_losses, 1, name='row_losses')

        losses_tmp = tf.truediv(row_losses, tf.cast(lengths, tf.float32))
        loss_for_compare = tf.reduce_mean(losses_tmp)
        tf.summary.scalar('averged_L1_loss', loss_for_compare)
 
        loss = tf.reduce_sum(row_losses)
        loss = tf.truediv(
          loss,
          tf.cast(tf.maximum(batch_size, 1), tf.float32))
        l1_loss_tensor = loss * 0.5
    return l1_loss_tensor

  def provide_groundtruth(self, groundtruth_text, scope=None):
    with tf.name_scope(scope, 'ProvideGroundtruth', [groundtruth_text]):
      batch_size = shape_utils.combined_static_and_dynamic_shape(groundtruth_text)[0]
      if self._reverse:
        groundtruth_text = ops.string_reverse(groundtruth_text)
      text_labels, text_lengths = self._label_map.text_to_labels(
        groundtruth_text,
        pad_value=self.end_label,
        return_lengths=True)
      start_labels = tf.fill([batch_size, 1], tf.constant(self.start_label, tf.int64))
      end_labels = tf.fill([batch_size, 1], tf.constant(self.end_label, tf.int64))
      if not self._sync:
        decoder_inputs = tf.concat([start_labels, start_labels, text_labels], axis=1)
        decoder_targets = tf.concat([start_labels, text_labels, end_labels], axis=1)
        decoder_lengths = text_lengths + 2
      else:
        decoder_inputs = tf.concat([start_labels, text_labels], axis=1)
        decoder_targets = tf.concat([text_labels, end_labels], axis=1)
        decoder_lengths = text_lengths + 1

      # set maximum lengths
      decoder_inputs = decoder_inputs[:,:self._max_num_steps]
      decoder_targets = decoder_targets[:,:self._max_num_steps]
      decoder_lengths = tf.minimum(decoder_lengths, self._max_num_steps)
      
      self._groundtruth_dict['decoder_inputs'] = decoder_inputs
      self._groundtruth_dict['decoder_targets'] = decoder_targets
      self._groundtruth_dict['decoder_lengths'] = decoder_lengths

  def postprocess(self, predictions_dict, scope=None):
    assert 'scores' in predictions_dict
    with tf.variable_scope(scope, 'Postprocess', list(predictions_dict.values())):
      text = self._label_map.labels_to_text(predictions_dict['labels'])
      if self._reverse:
        text = ops.string_reverse(text)
      scores = predictions_dict['scores']
    return {'text': text, 'scores': scores}

  def _build_decoder_cell(self, feature_maps):
    attention_mechanism = self._build_attention_mechanism(feature_maps)
    wrapper_class = seq2seq.AttentionWrapper if not self._sync else sync_attention_wrapper.SyncAttentionWrapper
    attention_cell = wrapper_class(
      self._rnn_cell,
      attention_mechanism,
      output_attention=False)
    if not self._lm_rnn_cell:
      decoder_cell = attention_cell
    else:
      decoder_cell = ConcatOutputMultiRNNCell([attention_cell, self._lm_rnn_cell])

    return decoder_cell

  def _build_attention_mechanism(self, feature_maps):
    """Build (possibly multiple) attention mechanisms."""
    def _build_single_attention_mechanism(memory):
      if not self._is_training:
        memory = seq2seq.tile_batch(memory, multiplier=self._beam_width)
      return seq2seq.BahdanauAttention(
        self._num_attention_units,
        memory,
        memory_sequence_length=None
      )
    
    feature_sequences = [tf.squeeze(map, axis=1) for map in feature_maps]
    if self._multi_attention:
      attention_mechanism = []
      for i, feature_sequence in enumerate(feature_sequences):
        memory = feature_sequence
        attention_mechanism.append(_build_single_attention_mechanism(memory))
    else:
      memory = tf.concat(feature_sequences, axis=1)
      attention_mechanism = _build_single_attention_mechanism(memory)
    return attention_mechanism

  def _build_decoder(self, decoder_cell, batch_size):
    embedding_fn = functools.partial(tf.one_hot, depth=self.num_classes)
    output_layer = tf.layers.Dense(
      self.num_classes,
      activation=None,
      use_bias=True,
      kernel_initializer=tf.variance_scaling_initializer(),
      bias_initializer=tf.zeros_initializer())
    if self._is_training:
      train_helper = seq2seq.TrainingHelper(
        embedding_fn(self._groundtruth_dict['decoder_inputs']),
        sequence_length=self._groundtruth_dict['decoder_lengths'],
        time_major=False)
      decoder = seq2seq.BasicDecoder(
        cell=decoder_cell,
        helper=train_helper,
        initial_state=decoder_cell.zero_state(batch_size, tf.float32),
        output_layer=output_layer)
    else:
      decoder = seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding_fn,
        start_tokens=tf.fill([batch_size], self.start_label),
        end_token=self.end_label,
        initial_state=decoder_cell.zero_state(batch_size * self._beam_width, tf.float32),
        beam_width=self._beam_width,
        output_layer=output_layer,
        length_penalty_weight=0.0)
    return decoder


class ConcatOutputMultiRNNCell(rnn.MultiRNNCell):
  """RNN cell composed of multiple RNN cells whose outputs are concatenated along depth."""

  @property
  def output_size(self):
    return sum([cell.output_size for cell in self._cells])

  def call(self, inputs, state):
    cur_state_pos = 0
    outputs = []
    new_states = []
    for i, cell in enumerate(self._cells):
      with vs.variable_scope("cell_%d" % i):
        if self._state_is_tuple:
          if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s" %
                (len(self.state_size), state))
          cur_state = state[i]
        else:
          cur_state = array_ops.slice(state, [0, cur_state_pos],
                                      [-1, cell.state_size])
          cur_state_pos += cell.state_size
        cur_output, new_state = cell(inputs, cur_state)
        new_states.append(new_state)
        outputs.append(cur_output)

    new_states = (tuple(new_states) if self._state_is_tuple else
                  array_ops.concat(new_states, 1))
    output = tf.concat(outputs, -1)

    return output, new_states
