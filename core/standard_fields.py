class InputDataFields(object):
  image = 'image'
  original_image = 'original_image'
  key = 'key'
  source_id = 'source_id'
  filename = 'filename'
  groundtruth_text = 'groundtruth_text'
  groundtruth_glyphs = 'groundtruth_glpyphs'
  groundtruth_keypoints = 'groundtruth_keypoints'
  lexicon = 'lexicon'


class TfExampleFields(object):
  image_encoded = 'image/encoded'
  target_image_encoded = 'image/target'
  image_format = 'image/format'  # format is reserved keyword
  target_image_format = 'image/target_format'  # format is reserved keyword
  filename = 'image/filename'
  channels = 'image/channels'
  colorspace = 'image/colorspace'
  height = 'image/height'
  width = 'image/width'
  source_id = 'image/source_id'
  transcript = 'image/transcript'
  lexicon = 'image/lexicon'
  keypoints = 'image/keypoints'
