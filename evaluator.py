import os
import logging
import tensorflow as tf
import numpy as np
import editdistance

from core import preprocessor
from core import prefetcher
from core import standard_fields as fields
from builders import preprocessor_builder
import eval_util


EVAL_METRICS_FN_DICT = {
  'recognition_metrics': eval_util.evaluate_recognition_results,
}


def _extract_prediction_tensors(model,
                                create_input_dict_fn,
                                data_preprocessing_steps,
                                ignore_groundtruth=False,
                                evaluate_with_lexicon=False):
  # input queue
  input_dict = create_input_dict_fn()
  prefetch_queue = prefetcher.prefetch(input_dict, capacity=500)
  input_dict = prefetch_queue.dequeue()
  original_image = tf.to_float(input_dict[fields.InputDataFields.image])
  original_image_shape = tf.shape(original_image)
  input_dict[fields.InputDataFields.image] = original_image

  # data preprocessing
  preprocessed_input_dict = preprocessor.preprocess(input_dict, data_preprocessing_steps)

  # model inference
  preprocessed_image = preprocessed_input_dict[fields.InputDataFields.image]
  preprocessed_image_shape = tf.shape(preprocessed_image)
  predictions_dict = model.predict(tf.expand_dims(preprocessed_image, 0))
  recognitions = model.postprocess(predictions_dict)

  def _lexicon_search(lexicon, word):
    edit_distances = []
    for lex_word in lexicon:
      edit_distances.append(editdistance.eval(lex_word.lower(), word.lower()))
    edit_distances = np.asarray(edit_distances, dtype=np.int)
    argmin = np.argmin(edit_distances)
    return lexicon[argmin]

  if evaluate_with_lexicon:
    lexicon = input_dict[fields.InputDataFields.lexicon]
    recognition_text = tf.py_func(
      _lexicon_search,
      [lexicon, recognitions['text'][0]],
      tf.string,
      stateful=False,
    )
  else:
    recognition_text = recognitions['text'][0]

  tensor_dict = {
    'original_image': original_image,
    'original_image_shape': original_image_shape,
    'preprocessed_image_shape': preprocessed_image_shape,
    'filename': preprocessed_input_dict[fields.InputDataFields.filename],
    'groundtruth_text': input_dict[fields.InputDataFields.groundtruth_text],
    'recognition_text': recognition_text,
  }
  if 'control_points' in predictions_dict:
    tensor_dict.update({
      'control_points': predictions_dict['control_points'],
      'rectified_images': predictions_dict['rectified_images'],
      'generated_images': predictions_dict['generated_images']
    })

  return tensor_dict


def evaluate(create_input_dict_fn, create_model_fn, eval_config,
             checkpoint_dir, eval_dir,
             repeat_evaluation=True):
  model = create_model_fn()
  data_preprocessing_steps = [
      preprocessor_builder.build(step)
      for step in eval_config.data_preprocessing_steps]

  tensor_dict = _extract_prediction_tensors(
      model=model,
      create_input_dict_fn=create_input_dict_fn,
      data_preprocessing_steps=data_preprocessing_steps,
      ignore_groundtruth=eval_config.ignore_groundtruth,
      evaluate_with_lexicon=eval_config.eval_with_lexicon)

  summary_writer = tf.summary.FileWriter(eval_dir)

  def _process_batch(tensor_dict, sess, batch_index, counters, update_op):
    if batch_index >= eval_config.num_visualizations:
      if 'original_image' in tensor_dict:
        tensor_dict = {k: v for (k, v) in tensor_dict.items()
                       if k != 'original_image'}
    try:
      (result_dict, _, glyphs) = sess.run([tensor_dict, update_op, tf.get_collection('glyph')])
      counters['success'] += 1
    except tf.errors.InvalidArgumentError:
      logging.info('Skipping image')
      counters['skipped'] += 1
      return {}
    global_step = tf.train.global_step(sess, tf.train.get_global_step())
    if batch_index < eval_config.num_visualizations:
      eval_util.visualize_recognition_results(
          result_dict,
          'Recognition_{}'.format(batch_index),
          global_step,
          summary_dir=eval_dir,
          export_dir=os.path.join(eval_dir, 'vis'),
          summary_writer=summary_writer,
          only_visualize_incorrect=eval_config.only_visualize_incorrect)

    return result_dict

  def _process_aggregated_results(result_lists):
    eval_metric_fn_key = eval_config.metrics_set
    if eval_metric_fn_key not in EVAL_METRICS_FN_DICT:
      raise ValueError('Metric not found: {}'.format(eval_metric_fn_key))
    return EVAL_METRICS_FN_DICT[eval_metric_fn_key](result_lists)

  variables_to_restore = tf.global_variables()
  #variables_to_restore = tf.trainable_variables()
  global_step = tf.train.get_or_create_global_step()
  variables_to_restore.append(global_step)
  if eval_config.use_moving_averages:
    variable_averages = tf.train.ExponentialMovingAverage(0.0)
    variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)
  def _restore_latest_checkpoint(sess):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, latest_checkpoint)

  eval_util.repeated_checkpoint_run(
      tensor_dict=tensor_dict,
      update_op=tf.no_op(),
      summary_dir=eval_dir,
      aggregated_result_processor=_process_aggregated_results,
      batch_processor=_process_batch,
      checkpoint_dirs=[checkpoint_dir],
      variables_to_restore=None,
      restore_fn=_restore_latest_checkpoint,
      num_batches=eval_config.num_examples,
      eval_interval_secs=eval_config.eval_interval_secs,
      max_number_of_evaluations=(
          1 if eval_config.ignore_groundtruth else
          eval_config.max_evals if eval_config.max_evals else
          None if repeat_evaluation else 1),
      master=eval_config.eval_master,
      save_graph=eval_config.save_graph,
      save_graph_dir=(eval_dir if eval_config.save_graph else ''))

  summary_writer.close()
