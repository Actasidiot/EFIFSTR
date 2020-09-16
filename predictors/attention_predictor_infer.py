import logging
import functools

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape
from core import predictor
from core import sync_attention_wrapper
from core import loss
from utils import shape_utils
from c_ops import ops
from tensorflow.contrib.layers import conv2d, max_pool2d, fully_connected

class Compute2dAttentionLayer(tf.keras.layers.Layer):
  def __init__(self, embedding_dim, output_dim, cnn_fmap, **kwargs):
    #tf.keras.layers.Layer.__init__(self)
    self.embedding_dim = embedding_dim
    self.output_dim = output_dim
    self.cnn_fmap = cnn_fmap
    super(Compute2dAttentionLayer, self).__init__( **kwargs)
  def build(self, input_shape):
    super(Compute2dAttentionLayer, self).build(input_shape)
  
  def compute_output_shape(self, input_shape):
    #print('compute_output_shape')
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    return input_shape[:-1].concatenate(self.output_dim)
  
  def call(self, input):
    cnn_fmaps_lastscale = self.cnn_fmap[-1]
    batch_size, batch_len, depth = shape_utils.combined_static_and_dynamic_shape(input)
    _, feature_h, feature_w, feature_c = cnn_fmaps_lastscale.get_shape().as_list()
    ld_output = tf.reshape(input, (batch_size * batch_len, depth))
    ld_output_reshape = tf.reshape(ld_output, [batch_size * batch_len, 1, 1, depth])
    ld_output_conv = conv2d(ld_output_reshape, self.embedding_dim, 1, activation_fn = None ,  normalizer_fn=None, scope ='compute2d_attention_layer/hs_conv',reuse=tf.AUTO_REUSE)
    ld_output_conv_tile = tf.tile(ld_output_conv,[1,feature_h,feature_w,1])
    cnn_fmap_conv = conv2d(cnn_fmaps_lastscale, self.embedding_dim, 3, activation_fn = None , normalizer_fn=None, scope='compute2d_attention_layer/fmap_conv',reuse=tf.AUTO_REUSE)
    cnn_fmap_tile = tf.expand_dims(cnn_fmap_conv, 1)
    cnn_fmap_tile = tf.tile(cnn_fmap_tile,[1,batch_len,1,1,1])
    cnn_fmap_tile = tf.reshape(cnn_fmap_tile, [batch_size * batch_len, feature_h, feature_w, feature_c] )
    g = tf.nn.tanh(tf.add(cnn_fmap_tile, ld_output_conv_tile))
    g_conv = conv2d(g, 1, 1, scope='compute2d_attention_layer/g_conv',  activation_fn = None ,  normalizer_fn=None, reuse=tf.AUTO_REUSE)
    g_conv_reshape = tf.reshape(g_conv, [batch_size * batch_len, feature_w * feature_h])
    g_conv_reshape_softmax = tf.nn.softmax(g_conv_reshape)
    mask = tf.reshape(g_conv_reshape_softmax, [batch_size * batch_len, feature_h , feature_w, 1])
    g_tmp = tf.tile(tf.reshape(g_conv_reshape_softmax, [batch_size * batch_len, feature_h , feature_w, 1]),[1,1,1,feature_c])
    glimpse = tf.reduce_sum(tf.multiply(cnn_fmap_tile, g_tmp), [1,2])
    glimpse = tf.reshape(glimpse,[batch_size,batch_len, depth ] )
    c_h_concat = tf.concat([glimpse,tf.reshape(ld_output,[batch_size, batch_len, depth])],axis=-1)
    rnn_output = tf.layers.dense(c_h_concat, self.output_dim, name = 'compute2d_attention_layer/output_w',reuse=tf.AUTO_REUSE)
    output = tf.reshape(rnn_output,[1, -1, self.output_dim])
    return output

class AttentionPredictorInfer(predictor.Predictor):
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
    super(AttentionPredictorInfer, self).__init__(is_training)
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

  def predict(self, cnn_fmap, lstm_holistic_features, scope=None):
    '''
    if not isinstance(feature_maps, (list, tuple)):
      raise ValueError('`feature_maps` must be list of tuple')
    '''
    with tf.variable_scope(scope, 'Predict'):
      cnn_fmaps_lastscale = cnn_fmap[-1]
      batch_size = shape_utils.combined_static_and_dynamic_shape(cnn_fmaps_lastscale)[0]
      decoder_cell = self._build_decoder_cell()
      decoder = self._build_decoder(decoder_cell, batch_size, lstm_holistic_features, cnn_fmap)
      outputs, _, output_lengths = seq2seq.dynamic_decode(
        decoder=decoder,
        output_time_major=False,
        impute_finished=False,
        maximum_iterations=self._max_num_steps
      )
      # apply regularizer
      filter_weights = lambda vars : [x for x in vars if x.op.name.endswith('kernel')]
      tf.contrib.layers.apply_regularization(
        self._rnn_regularizer,
        filter_weights(decoder_cell.trainable_weights))

      outputs_dict = None
      if self._is_training:
        assert isinstance(outputs, seq2seq.BasicDecoderOutput)
        outputs_dict = {
          'labels': outputs.sample_id,
          'logits': outputs.rnn_output,
        }
      else:
        assert isinstance(outputs, seq2seq.FinalBeamSearchDecoderOutput)
        prediction_labels = outputs.predicted_ids[:,:,0]
        prediction_lengths = output_lengths[:,0]
        prediction_scores = tf.gather_nd(
          outputs.beam_search_decoder_output.scores[:,:,0],
          tf.stack([tf.range(batch_size), prediction_lengths-1], axis=1)
        )
        outputs_dict = {
          'labels': prediction_labels,
          'scores': prediction_scores,
          'lengths': prediction_lengths
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
      ref_glyphs = tf.constant(np.load('data/glyphs-325-fonts.npy'),dtype=tf.float32) # 96 ,325, 32*32
      labels = self._groundtruth_dict['decoder_targets']
      lengths = self._groundtruth_dict['decoder_lengths']
      batch_size, batch_len = shape_utils.combined_static_and_dynamic_shape(labels)
      labels_indexes = tf.reshape(labels, [batch_size * batch_len])
      targets = tf.gather(ref_glyphs, labels_indexes) # batch_size * batch_len, 32*32
      targets_for_visual = tf.reshape((targets + 1.0) * 127.5, [batch_size * batch_len, 32 , 32, 1])
      tf.summary.image('target_glyph1', (targets_for_visual[:20]) , max_outputs=20)
      targets = tf.reshape(targets, [batch_size, batch_len, 32*32])

      with tf.name_scope(scope, 'WeightedL1Loss'):
        raw_losses = tf.reduce_sum(tf.abs(glyphs - targets), axis=[2])
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

  def _build_decoder_cell(self):
    cell0 = tf.nn.rnn_cell.LSTMCell(512, state_is_tuple=True)
    if self._is_training:
        cell0 = tf.nn.rnn_cell.DropoutWrapper(cell=cell0, output_keep_prob=0.5)
    cell1 = tf.nn.rnn_cell.LSTMCell(512, state_is_tuple=True)
    if self._is_training:
        cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=0.5)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([cell0, cell1], state_is_tuple=True)
    return lstm_cell

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

  def _build_decoder(self, decoder_cell, batch_size, lstm_holistic_features, cnn_fmap):
    embedding_fn = functools.partial(tf.one_hot, depth=self.num_classes)
    output_layer = Compute2dAttentionLayer(512, self.num_classes, cnn_fmap)
    if self._is_training:
      train_helper = seq2seq.TrainingHelper(
        embedding_fn(self._groundtruth_dict['decoder_inputs']),
        sequence_length=self._groundtruth_dict['decoder_lengths'],
        time_major=False)
      decoder = seq2seq.BasicDecoder(
        cell=decoder_cell,
        helper=train_helper,
        initial_state=lstm_holistic_features,
        output_layer=output_layer)
    else:
      lstm0_state_tile = tf.nn.rnn_cell.LSTMStateTuple(tf.tile(lstm_holistic_features[0].c,[self._beam_width,1]), tf.tile(lstm_holistic_features[0].h,[self._beam_width,1]))
      lstm1_state_tile= tf.nn.rnn_cell.LSTMStateTuple(tf.tile(lstm_holistic_features[1].c,[self._beam_width,1]),tf.tile( lstm_holistic_features[1].h,[self._beam_width,1]))
      lstm_holistic_features_tile = (lstm0_state_tile, lstm1_state_tile)
      decoder = seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding_fn,
        start_tokens=tf.fill([batch_size], self.start_label),
        end_token=self.end_label,
        initial_state = lstm_holistic_features_tile,
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
